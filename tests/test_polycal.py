#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from prospect.sources import CSPSpecBasis
from prospect.models import SpecModel, templates
from prospect.observation import Spectrum, Photometry, PolyOptCal


class PolySpectrum(PolyOptCal, Spectrum):
    pass



@pytest.fixture
def get_sps():
    sps = CSPSpecBasis(zcontinuous=1)
    return sps


def build_model(add_neb=False):
    model_params = templates.TemplateLibrary["parametric_sfh"]
    if add_neb:
        model_params.update(templates.TemplateLibrary["nebular"])
    return SpecModel(model_params)


def build_obs(multispec=False):
    N = 1500 * (2 - multispec)
    wmax = 7000
    wsplit = wmax - N * multispec

    fnames = list([f"sdss_{b}0" for b in "ugriz"])
    Nf = len(fnames)
    phot = [Photometry(filters=fnames,
                       flux=np.ones(Nf),
                       uncertainty=np.ones(Nf)/10)]
    spec = [PolySpectrum(wavelength=np.linspace(4000, wsplit, N),
                         flux=np.ones(N),
                         uncertainty=np.ones(N) / 10,
                         mask=slice(None),
                         polynomial_order=5)
            ]

    if multispec:
        spec += [Spectrum(wavelength=np.linspace(wsplit+1, wmax, N),
                          flux=np.ones(N), uncertainty=np.ones(N) / 10,
                          mask=slice(None))]

    obslist = spec + phot
    [obs.rectify() for obs in obslist]
    return obslist


def test_polycal(plot=False):
    """Make sure the polynomial optimization works
    """
    sps = get_sps
    observations = build_obs()
    model = build_model()

    preds, extra = model.predict(model.theta, observations=observations, sps=sps)
    obs = observations[0]

    assert np.any(obs.response != 0)

    if plot:
        import matplotlib.pyplot as pl
        fig, axes = pl.subplots(3, 1, sharex=True)
        ax = axes[0]
        ax.plot(obs.wavelength, obs.flux, label="obseved flux (ones)")
        ax.plot(obs.wavelength, preds[0], label="model flux (times response)")
        ax = axes[1]
        ax.plot(obs.wavelength, obs.response, label="instrumental response (polynomial)")
        ax = axes[2]
        ax.plot(obs.wavelength, preds[0]/ obs.response, label="intrinsic model spectrum")
        ax.set_xlabel("wavelength")
        [ax.legend() for ax in axes]