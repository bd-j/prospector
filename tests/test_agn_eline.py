#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
import numpy as np

from sedpy import observate
from prospect.utils.obsutils import fix_obs
from prospect.models.sedmodel import AGNSpecModel
from prospect.models.templates import TemplateLibrary
from prospect.sources import CSPSpecBasis


def test_agn_elines():
    # --- sps ---
    sps = CSPSpecBasis(zcontinuous=1)

    # --- simple obs ---
    fnames = [f"sdss_{b}0" for b in "ugriz"]
    filts = observate.load_filters(fnames)

    obs = dict(filters=filts,
               wavelength=np.linspace(3000, 9000, 1000),
               spectrum=np.ones(1000),
               unc=np.ones(1000)*0.1)
    obs = fix_obs(obs)

    # --- model ---
    model_pars = TemplateLibrary["parametric_sfh"]
    model_pars.update(TemplateLibrary["nebular"])
    # add an emission line template for AGN
    model_pars.update(TemplateLibrary["agn_eline"])
    # make it old
    model_pars["tau"]["init"] = 0.05
    model_pars["tage"]["init"] = 3
    model = AGNSpecModel(model_pars)

    model.params["agn_elum"] = 1e-4
    spec0, phot0, x0 = model.predict(model.theta, obs, sps)
    model.params["agn_elum"] = 1e-6
    spec1, phot1, x1 = model.predict(model.theta, obs, sps)
    assert (not np.allclose(spec1, spec0)), "changing AGN luminosity had no effect"

    model.params["agn_elum"] = 1e-4
    model.params["agn_eline_sigma"] = 400.0
    spec2, phot2, x2 = model.predict(model.theta, obs, sps)
    assert (not np.allclose(spec2, spec0)), "broadening lines had no effect on the spectrum"

    assert np.allclose(phot2, phot0), "broadening lines changed the photometry"

    # do a check for phot-only obs
    pobs = dict(filters=filts,
                maggies=np.ones(len(filts)),
                maggies_unc=0.1 * np.ones(len(filts)),
                wavelength=np.linspace(3000, 9000, 1000),
                spectrum=None)
    pobs = fix_obs(pobs)
    spec3, phot3, x2 = model.predict(model.theta, obs=pobs, sps=sps)
    assert np.allclose(phot3, phot2), "Phot-only obs did not add AGn lines correctly"

    if False:
        import matplotlib.pyplot as pl
        pl.ion()
        fig, ax = pl.subplots()
        ax.plot(obs["wavelength"], spec0, label="agn_elum=1e-4")
        ax.plot(obs["wavelength"], spec1, label="agn_elum=1e-6")
        ax.plot(obs["wavelength"], spec2, label="agn_elum=1e-4, agn_sigma=400km/s")
        ax.legend()
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("F_nu (maggies)")

        print("Change in magnitude for faint AGN:\n", -2.5*np.log10(phot1/phot0))
        print("Change in magnitude for broader-line AGN:\n", -2.5*np.log10(phot2/phot0))
