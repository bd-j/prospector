#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from prospect.sources import CSPSpecBasis
from prospect.models import SpecModel, templates
from prospect.observation import Spectrum, Photometry, Lines


@pytest.fixture(scope="module")
def build_sps():
    sps = CSPSpecBasis(zcontinuous=1)
    return sps


def build_model(add_neb=False):
    model_params = templates.TemplateLibrary["parametric_sfh"]
    if add_neb:
        model_params.update(templates.TemplateLibrary["nebular"])
    return SpecModel(model_params)


def build_obs(multispec=True, add_lines=False):
    N = 1500 * (2 - multispec)
    wmax = 7000
    wsplit = wmax - N * multispec

    fnames = list([f"sdss_{b}0" for b in "ugriz"])
    Nf = len(fnames)
    phot = [Photometry(filters=fnames, flux=np.ones(Nf), uncertainty=np.ones(Nf)/10)]
    spec = [Spectrum(wavelength=np.linspace(4000, wsplit, N),
                     flux=np.ones(N), uncertainty=np.ones(N) / 10,
                     mask=slice(None))]

    if multispec:
        spec += [Spectrum(wavelength=np.linspace(wsplit+1, wmax, N),
                          flux=np.ones(N), uncertainty=np.ones(N) / 10,
                          mask=slice(None))]

    obs = spec + phot

    if add_lines:
        line_ind = [59, 62, 74, 73, 75] # index of line in FSPS table
        line_name = ["Hb", "OIII-5007", "Ha", "NII-6548", "NII-6584"]
        line_wave = [4861, 5007, 6563, 6548, 6584] # can be approximate
        n_line = len(line_ind)
        lines = Lines(line_ind=line_ind,
                      flux=np.ones(n_line), # erg/s/cm^2
                      uncertainty=np.ones(n_line)/10,
                      line_names=line_name,  # optional
                      wavelength=line_wave, # optional
                    )
        obs += [lines]

    [ob.rectify() for ob in obs]
    for ob in obs:
        assert ob.ndof > 0

    return obs


def test_prediction_nodata(build_sps):
    sps = build_sps
    model = build_model(add_neb=True)
    sobs, pobs = build_obs(multispec=False)
    pobs.flux = None
    pobs.uncertainty = None
    sobs.wavelength = None
    sobs.flux = None
    sobs.uncertainty = None
    pred, mfrac = model.predict(model.theta, observations=[sobs, pobs], sps=sps)
    assert len(pred[0]) == len(sps.wavelengths)
    assert np.any(np.isfinite(pred[0]))
    assert len(pred[1]) == len(pobs.filterset)
    assert np.any(np.isfinite(pred[1]))


def test_multispec(build_sps, plot=False):
    sps = build_sps

    obslist_single = build_obs(multispec=False)
    obslist_multi = build_obs(multispec=True)
    model = build_model(add_neb=True)

    preds_single, mfrac = model.predict(model.theta, observations=obslist_single, sps=sps)
    preds_multi, mfrac = model.predict(model.theta, observations=obslist_multi, sps=sps)

    assert len(preds_single) == 2
    assert len(preds_multi) == 3
    assert np.allclose(preds_single[-1], preds_multi[-1])

    # TODO: turn this plot into an actual test
    if plot:
        import matplotlib.pyplot as pl
        fig, ax = pl.subplots()
        ax.plot(obslist_single[0].wavelength, preds_single[0])
        for p, o in zip(preds_multi, obslist_multi):
           if o.kind == "photometry":
               ax.plot(o.wavelength, p, "o")
           else:
               ax.plot(o.wavelength, p)


def test_line(build_sps, plot=False):

    obs = build_obs(multispec=False, add_lines=True)
    model = build_model(add_neb=True)

    sps = build_sps
    preds, mfrac = model.predict(model.theta, observations=obs, sps=sps)
    (spec, phot, lines) = preds
    assert np.all(lines) > 0

    #print(f"log(OIII[5007]/Hb)={np.log10(lines[1]/lines[0])}")
    #print(f"log(NII/Ha)={np.log10(lines[-2:]/lines[2])}")


def lnlike_testing(build_sps):
    # testing lnprobfn

    observations = build_obs()
    model = build_model(add_neb=True)
    sps = build_sps

    from prospect.likelihood.likelihood import compute_lnlike
    from prospect.fitting import lnprobfn

    predictions, x = model.predict(model.theta, observations, sps=sps)
    lnp_data = [compute_lnlike(pred, obs, vectors={}) for pred, obs
                in zip(predictions, observations)]
    assert np.all([np.isscalar(p) for p in lnp_data])
    assert len(lnp_data) == len(observations)

    lnp = lnprobfn(model.theta, model=model, observations=observations, sps=sps)

    assert np.isscalar(lnp)

    # %timeit model.prior_product(model.theta)
    # arr = np.zeros(model.ndim)
    # arr[-1] = 1
    # theta = model.theta.copy()
    # %timeit predictions, x = model.predict(theta + np.random.uniform(-0.1, 0.1) * arr, observations=observations, sps=sps)
    # %timeit lnp_data = [compute_lnlike(pred, obs, vectors={}) for pred, obs in zip(predictions, observations)]
    # %timeit lnp = lnprobfn(theta + np.random.uniform(0, 3) * arr, model=model, observations=observations, sps=sps)
