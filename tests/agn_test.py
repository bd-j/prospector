#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl

from sedpy import observate
from prospect.utils.obsutils import fix_obs
from prospect.models.sedmodel import AGNSpecModel
from prospect.models.templates import TemplateLibrary
from prospect.sources import CSPSpecBasis
from prospect.models import priors


_agn_eline_ = {}
_agn_eline_["agn_elum"] = dict(N=1, isfree=False, init=1e-4,
                               prior=priors.Uniform(mini=1e-6, maxi=1e-2))
_agn_eline_["agn_eline_sigma"] = dict(N=1, isfree=False, init=100.0,
                                      prior=priors.Uniform(mini=50, maxi=500))


if __name__ == "__main__":

    #
    sps = CSPSpecBasis(zcontinuous=1)

    #obs
    fnames = [f"sdss_{b}0" for b in "ugriz"]
    filts = observate.load_filters(fnames)

    obs = dict(filters=filts,
               wavelength=np.linspace(3000, 9000, 1000),
               spectrum=np.ones(1000),
               unc=np.ones(1000)*0.1)
    obs = fix_obs(obs)

    # model
    model_pars = TemplateLibrary["parametric_sfh"]
    model_pars.update(TemplateLibrary["nebular"])
    # add an emission line template for AGN
    model_pars.update(TemplateLibrary["agn_eline"])
    # make it old
    model_pars["tau"]["init"] = 0.05
    model_pars["tage"]["init"] = 3
    model = AGNSpecModel(model_pars)

    model.params["agn_elum"] = 1e-4
    spec, phot, x = model.predict(model.theta, obs, sps)
    #aind = model.theta_index["agne_elum"]
    model.params["agn_elum"] = 1e-6
    spec1, phot1, x1 = model.predict(model.theta, obs, sps)

    model.params["agn_elum"] = 1e-4
    model.params["agn_eline_sigma"] = 400.0
    spec2, phot2, x2 = model.predict(model.theta, obs, sps)


    pl.ion()
    fig, ax = pl.subplots()
    ax.plot(obs["wavelength"], spec, label="agn_elum=1e-4")
    ax.plot(obs["wavelength"], spec1, label="agn_elum=1e-6")
    ax.plot(obs["wavelength"], spec2, label="agn_elum=1e-4, agn_sigma=400km/s")
    ax.legend()
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("F_nu (maggies)")

    print("Change in magnitude for faint AGN:\n",-2.5*np.log10(phot1/phot))
    print("Change in magnitude for broad-line AGN:\n",-2.5*np.log10(phot2/phot))
