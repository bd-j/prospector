#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np

import pytest

from prospect.sources import CSPSpecBasis
from prospect.models import SpecModel, templates
from prospect.observation import Spectrum, Photometry
from prospect.likelihood import NoiseModel
from prospect.likelihood.likelihood import compute_lnlike
from prospect.fitting import lnprobfn


@pytest.fixture
def get_sps():
    sps = CSPSpecBasis(zcontinuous=1)
    return sps


def build_model(add_neb=False, add_outlier=False):
    model_params = templates.TemplateLibrary["parametric_sfh"]
    if add_neb:
        model_params.update(templates.TemplateLibrary["nebular"])
    if add_outlier:
        model_params.update(templates.TemplateLibrary["outlier_model"])
        model_params["f_outlier_phot"]["isfree"] = True
        model_params["f_outlier_phot"]["init"] = 0.05
    return SpecModel(model_params)


def build_obs(multispec=True, add_outlier=True):
    N = 1500 * (2 - multispec)
    wmax = 7000
    wsplit = wmax - N * multispec

    fnames = list([f"sdss_{b}0" for b in "ugriz"])
    Nf = len(fnames)
    phot = [Photometry(filters=fnames, flux=np.ones(Nf), uncertainty=np.ones(Nf)/10)]
    spec = [Spectrum(wavelength=np.linspace(4000, wsplit, N),
                     flux=np.ones(N), uncertainty=np.ones(N) / 10,
                     mask=slice(None))]

    if add_outlier:
        phot[0].noise = NoiseModel(frac_out_name='f_outlier_phot',
                                   nsigma_out_name='nsigma_outlier_phot')

    if multispec:
        spec += [Spectrum(wavelength=np.linspace(wsplit+1, wmax, N),
                          flux=np.ones(N), uncertainty=np.ones(N) / 10,
                          mask=slice(None))]

    obslist = spec + phot
    [obs.rectify() for obs in obslist]
    return obslist


def test_lnlike_shape(get_sps):
    # testing lnprobfn
    sps = get_sps

    for add_out in [True, False]:
        observations = build_obs(add_outlier=add_out)
        model = build_model(add_neb=add_out, add_outlier=add_out)

        model.set_parameters(model.theta)
        [obs.noise.update(**model.params) for obs in observations
        if obs.noise is not None]
        predictions, x = model.predict(model.theta, observations, sps=sps)

        # check you get a scalar lnp for each observation
        lnp_data = [compute_lnlike(pred, obs, vectors={}) for pred, obs
                    in zip(predictions, observations)]
        assert np.all([np.isscalar(p) for p in lnp_data]), f"failed for add_outlier={add_out}"
        assert len(lnp_data) == len(observations), f"failed for add_outlier={add_out}"

        # check lnprobfn returns scalar
        lnp = lnprobfn(model.theta, model=model, observations=observations, sps=sps)

        assert np.isscalar(lnp), f"failed for add_outlier={add_out}"

    # %timeit model.prior_product(model.theta)
    # arr = np.zeros(model.ndim)
    # arr[-1] = 1
    # theta = model.theta.copy()
    # %timeit predictions, x = model.predict(theta + np.random.uniform(-0.1, 0.1) * arr, observations=observations, sps=sps)
    # %timeit lnp_data = [compute_lnlike(pred, obs, vectors={}) for pred, obs in zip(predictions, observations)]
    # %timeit lnp = lnprobfn(theta + np.random.uniform(0, 3) * arr, model=model, observations=observations, sps=sps)
