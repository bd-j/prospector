#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np

from sedpy.observate import load_filters
from prospect.sources import CSPSpecBasis
from prospect.models import SpecModel, templates
from prospect.data import Spectrum, Photometry


def build_model(add_neb=False):
    model_params = templates.TemplateLibrary["parametric_sfh"]
    if add_neb:
        model_params.update(templates.TemplateLibrary["nebular"])
    return SpecModel(model_params)


def build_obs(multispec=True):
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

    obslist = spec + phot
    [obs.rectify() for obs in obslist]
    return obslist


def build_sps():
    sps = CSPSpecBasis(zcontinuous=1)
    return sps


def test_multispec():
    obslist_single = build_obs(multispec=False)
    obslist = build_obs()
    model = build_model(add_neb=True)
    sps = build_sps()

    predictions_single, mfrac = model.predict(model.theta, observations=obslist_single, sps=sps)
    predictions, mfrac = model.predict(model.theta, observations=obslist, sps=sps)

    assert len(predictions_single) == 2
    assert len(predictions) == 3
    assert np.allclose(predictions_single[-1], predictions[-1])
    # TODO: turn this plot into an actual test
    #import matplotlib.pyplot as pl
    #fig, ax = pl.subplots()
    #ax.plot(obslist_single[0].wavelength, predictions_single[0])
    #for p, o in zip(predictions, obslist):
    #    if o.kind == "photometry":
    #        ax.plot(o.wavelength, p, "o")
    #    else:
    #        ax.plot(o.wavelength, p)


def lnlike_testing():

    # testing lnprobfn
    observations = build_obs()
    model = build_model(add_neb=True)
    from prospect.likelihood.likelihood import compute_lnlike
    from prospect.fitting import lnprobfn

    lnp = lnprobfn(model.theta, model=model, observations=obslist, sps=sps)

    #%timeit model.prior_product(model.theta)
    #%timeit predictions, x = model.predict(model.theta + np.random.uniform(0, 3) * arr, observations=obslist, sps=sps)
    #%timeit lnp_data = [compute_lnlike(pred, obs, vectors={}) for pred, obs in zip(predictions, observations)]
    #%timeit lnp = lnprobfn(model.theta + np.random.uniform(0, 3) * arr, model=model, observations=obslist, sps=sps)