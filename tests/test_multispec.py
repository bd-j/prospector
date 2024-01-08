#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np

import pytest

from sedpy.observate import load_filters
from prospect.sources import CSPSpecBasis
from prospect.models import SpecModel, templates
from prospect.observation import Spectrum, Photometry


@pytest.fixture(scope="module")
def build_sps():
    sps = CSPSpecBasis(zcontinuous=1)
    return sps


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
    assert len(pred[1]) == len(pobs.filterset)


def test_multispec(build_sps):
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
    #import matplotlib.pyplot as pl
    #fig, ax = pl.subplots()
    #ax.plot(obslist_single[0].wavelength, predictions_single[0])
    #for p, o in zip(predictions, obslist):
    #    if o.kind == "photometry":
    #        ax.plot(o.wavelength, p, "o")
    #    else:
    #        ax.plot(o.wavelength, p)


def test_multires():
    # Test the smoothing of multiple spectra to different resolutions
    # - give the same wavelength array different instrumental resolutions, assert similar but different, and that smoothing by the difference gives the right answer
    # Test the use of two differernt smoothings, physical and instrumental
    # - give an obs with no instrument smoothing and one with, make sure they are different
    pass


def test_multinoise():
    pass


def test_multical():
    pass