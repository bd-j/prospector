#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sedpy.observate import load_filters
from prospect.sources import CSPSpecBasis
from prospect.models import SpecModel, templates
from prospect.utils.observation import Spectrum, Photometry


def build_model():
    model_params = templates.TemplateLibrary["parametric_sfh"]
    return SpecModel(model_params)


def build_obs(multispec=True):
    N = 1500 * (2 - multispec)
    wmax = 7000
    wsplit = wmax - N * multispec

    filterlist = load_filters([f"sdss_{b}0" for b in "ugriz"])
    Nf = len(filterlist)
    phot = [Photometry(filters=filterlist, flux=np.ones(Nf), uncertainty=np.ones(Nf)/10)]
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


if __name__ == "__main__":
    obslist_single = build_obs(multispec=False)
    obslist = build_obs()
    model = build_model()
    sps = build_sps()

    #sys.exit()
    predictions_single, mfrac = model.predict(model.theta, obslist=obslist_single, sps=sps)
    #sys.exit()
    predictions, mfrac = model.predict(model.theta, obslist=obslist, sps=sps)

    import matplotlib.pyplot as pl
    fig, ax = pl.subplots()
    ax.plot(obslist_single[0].wavelength, predictions_single[0])
    for p, o in zip(predictions, obslist):
        if o.kind == "photometry":
            ax.plot(o.wavelength, p, "o")
        else:
            ax.plot(o.wavelength, p)

