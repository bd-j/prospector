#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
#import pytest

from prospect.sources import CSPSpecBasis
from prospect.models import SpecModel, templates
from prospect.observation import Spectrum, UndersampledSpectrum

#@pytest.fixture(scope="module")
def build_sps():
    sps = CSPSpecBasis(zcontinuous=1)
    return sps


def build_model(add_neb=False, sigma_v=200):
    model_params = templates.TemplateLibrary["ssp"]
    model_params.update(templates.TemplateLibrary["spectral_smoothing"])
    model_params["sigma_smooth"] = dict(N=1, isfree=False, init=sigma_v)
    if add_neb:
        model_params.update(templates.TemplateLibrary["nebular"])
        model_params["nebemlineinspec"]["init"] = np.array([False])
        model_params["eline_sigma"] = dict(N=1, isfree=False, init=sigma_v)

    model_params["tage"]["init"] = 0.2

    return SpecModel(model_params)


def build_obs(undersampling=4):

    wmin, wmax = 4000, 7000
    fwhm = 5  # instrumental LSF FWHM in AA
    dl_s = fwhm / 3 # well sampled spectrum
    dl_u = fwhm / 2 * undersampling # horribly undersampled spectrum

    wave = np.arange(wmin, wmax, dl_s)
    resolution = (fwhm/2.355) / wave * 2.998e5  # in km/s
    full = Spectrum(wavelength=wave.copy(),
                    flux=np.ones(len(wave)),
                    uncertainty=np.ones(len(wave)) / 10,
                    resolution=resolution,
                    mask=slice(None),
                    name="Oversampled no pixelization")

    full_pix = UndersampledSpectrum(wavelength=wave.copy(),
                    flux=np.ones(len(wave)),
                    uncertainty=np.ones(len(wave)) / 10,
                    resolution=resolution,
                    mask=slice(None),
                    name="Oversampled but pixelized")

    wave = np.arange(wmin, wmax, dl_u)
    resolution = (fwhm/2.355) / wave * 2.998e5  # in km/s
    under_pix = UndersampledSpectrum(wavelength=wave.copy(),
                                 flux=np.ones(len(wave)),
                                 uncertainty=np.ones(len(wave)) / 10,
                                 resolution=resolution,
                                 mask=slice(None),
                                 name="Undersampled and pixelized")
    under = Spectrum(wavelength=wave.copy(),
                                 flux=np.ones(len(wave)),
                                 uncertainty=np.ones(len(wave)) / 10,
                                 resolution=resolution,
                                 mask=slice(None),
                                 name="Undersampled no pixelization")

    obslist = [full, full_pix, under, under_pix]
    [obs.rectify() for obs in obslist]
    return obslist


#def test_undersample(build_sps, plot=False):
if __name__ == "__main__":
    plot = True

    sps = build_sps()
    obslist = build_obs()
    model = build_model(add_neb=True)

    preds, x = model.predict(model.theta, observations=obslist, sps=sps)

    # TODO: turn this plot into an actual test
    if plot:
        import matplotlib.pyplot as pl
        fig, axes = pl.subplots(2, 1, sharex=True)
        ax = axes[0]
        ax.plot(model.observed_wave(model._wave), model._smooth_spec, label="LOSVD only")
        for p, o in zip(preds, obslist):
           if o.kind == "photometry":
               ax.plot(o.wavelength, p, "o")
           else:
               ax.step(o.wavelength, p, where="mid", label=o.name)
        ax.legend()

        ax = axes[1]
        ax.plot(obslist[0].wavelength, preds[0]/preds[1], label="Oversampled")
        ax.plot(obslist[2].wavelength, preds[2]/preds[3], label="Undersampled")
        ax.set_ylabel("Ratio of pixel-unconvolved to pixel-convolved flux")
        ax.legend()

        ax.set_xlim(o.wavelength.min(), o.wavelength.max())