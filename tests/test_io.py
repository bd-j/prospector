#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

import h5py

from prospect.models import SpecModel, templates
from prospect.sources import CSPSpecBasis
from prospect.observation import Photometry, Spectrum
from prospect.io.write_results import write_obs_to_h5
from prospect.io.read_results import obs_from_h5


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


def test_observation_io(build_sps, plot=False):
    sps = build_sps

    obslist = build_obs(multispec=True)
    model = build_model(add_neb=True)

    # obs writing
    with h5py.File("test.h5", "w") as hf:
        write_obs_to_h5(hf, obslist)
    with h5py.File("test.h5", "r") as hf:
        obsr = obs_from_h5(hf["observations"])
