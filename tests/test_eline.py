#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sedpy import observate

from prospect import prospect_args
from prospect.data import Photometry, Spectrum, from_oldstyle
from prospect.models.templates import TemplateLibrary
from prospect.models.sedmodel import SpecModel

from prospect.sources import CSPSpecBasis


# test nebular line specification
def test_eline_parsing():
    model_pars = TemplateLibrary["parametric_sfh"]
    model_pars.update(TemplateLibrary["nebular"])

    # test ignoring a line
    lya = "Ly alpha 1216"
    model_pars["elines_to_ignore"] = dict(init=lya, isfree=False)
    model = SpecModel(model_pars)
    model.parse_elines()
    assert not np.isin(lya, model.emline_info["name"][model._use_eline])
    assert np.isin("H alpha 6563", model.emline_info["name"][model._use_eline])
    assert np.all(model._fix_eline)
    assert model._use_eline.sum() == len(model._use_eline) - 1
    assert len(model._use_eline) == len(model.emline_info)

    # test fitting all the non-ignored lines
    model_pars["marginalize_elines"] = dict(init=True)
    model = SpecModel(model_pars)
    model.parse_elines()
    assert model._fit_eline.sum() == len(model._use_eline)

    # test fitting just a line or two
    fit_lines = ["[OIII]5007"]
    model_pars["elines_to_fit"] = dict(init=fit_lines)
    model = SpecModel(model_pars)
    model.parse_elines()
    assert model._fit_eline.sum() == len(fit_lines)
    assert model._fix_eline.sum() == (len(model._use_eline) - len(fit_lines))

    # test fixing a line
    _ = model_pars.pop("elines_to_fit")
    fix_lines = ["H beta 4861"]
    model_pars["elines_to_fix"] = dict(init=fit_lines)
    model = SpecModel(model_pars)
    model.parse_elines()
    assert model._fix_eline.sum() == len(fix_lines)
    assert model._fit_eline.sum() == (len(model._use_eline) - len(fix_lines))


def test_nebline_phot_addition():
    fnames = [f"sdss_{b}0" for b in "ugriz"]
    filts = observate.load_filters(fnames)

    obs = dict(filters=filts,
               wavelength=np.linspace(3000, 9000, 1000),
               spectrum=np.ones(1000),
               unc=np.ones(1000)*0.1)
    obslist = from_oldstyle(obs)

    sps = CSPSpecBasis(zcontinuous=1)

    # Make emission lines more prominent
    zred = 1.0

    # add nebline photometry in FSPS
    model_pars = TemplateLibrary["parametric_sfh"]
    model_pars["zred"]["init"] = zred
    model_pars.update(TemplateLibrary["nebular"])
    m1 = SpecModel(model_pars)

    # adding nebline photometry by hand
    model_pars = TemplateLibrary["parametric_sfh"]
    model_pars["zred"]["init"] = zred
    model_pars.update(TemplateLibrary["nebular"])
    model_pars["nebemlineinspec"]["init"] = False
    m2 = SpecModel(model_pars)

    (s1, p1), _ = m1.predict(m1.theta, obslist, sps)
    (s2, p2), _ = m2.predict(m2.theta, obslist, sps)

    # make sure some of the lines were important
    p1n = m1.nebline_photometry(filts)
    assert np.any(p1n / p1[1] > 0.05)

    # make sure you got the same answer
    assert np.all(np.abs(p1 - p2) / p1 < 1e-3)


def test_filtersets():
    """This test no longer relevant.....
    """
    fnames = [f"sdss_{b}0" for b in "ugriz"]
    flist = observate.load_filters(fnames)

    obs = dict(wavelength=np.linspace(3000, 9000, 1000),
               spectrum=np.ones(1000),
               unc=np.ones(1000)*0.1,
               filters=fnames)
    obslist = from_oldstyle(obs)

    sps = CSPSpecBasis(zcontinuous=1)

    # Make emission lines more prominent
    zred = 0.5
    models = []

    # test SpecModel no nebular emission
    model_pars = TemplateLibrary["parametric_sfh"]
    model_pars["zred"]["init"] = zred
    models.append(SpecModel(model_pars))

    # test SpecModel with nebular emission added by SpecModel
    model_pars = TemplateLibrary["parametric_sfh"]
    model_pars["zred"]["init"] = zred
    model_pars.update(TemplateLibrary["nebular"])
    model_pars["nebemlineinspec"]["init"] = False
    models.append(SpecModel(model_pars))

    for i, model in enumerate(models):
        (_, pset), _ = model.predict(model.theta, obslist, sps)

        # make sure some of the filters are affected by lines
        # ( nebular flux > 10% of total flux)
        if i == 1:
            nebphot = model.nebline_photometry(flist)
            assert np.any(nebphot / pset > 0.1)

        # make sure photometry is consistent
        # make sure we actually used different filter types
        # We always use filtersets now


def test_eline_implementation():

    test_eline_parsing()

    filters = observate.load_filters([f"sdss_{b}0" for b in "ugriz"])
    obs = dict(filters=filters,
               wavelength=np.linspace(3000, 9000, 1000),
               spectrum=np.ones(1000),
               unc=np.ones(1000)*0.1,
               maggies=np.ones(len(filters))*1e-7,
               maggies_unc=np.ones(len(filters))*1e-8)
    obslist = from_oldstyle(obs)

    model_pars = TemplateLibrary["parametric_sfh"]
    model_pars.update(TemplateLibrary["nebular"])
    model_pars["nebemlineinspec"]["init"] = False
    model_pars["eline_sigma"] = dict(init=500)
    model_pars["zred"]["init"] = 4
    model = SpecModel(model_pars)

    sps = CSPSpecBasis(zcontinuous=1)

    # generate with all fixed lines added
    (spec, phot), mfrac = model.predict(model.theta, obslist, sps=sps)

    # test ignoring a line
    lya = "Ly alpha 1216"
    model_pars["elines_to_ignore"] = dict(init=lya, isfree=False)
    model = SpecModel(model_pars)
    (spec_nolya, phot_nolya), mfrac = model.predict(model.theta, obslist, sps=sps)
    assert np.any((phot - phot_nolya) / phot != 0.0)
    lint = np.trapz(spec - spec_nolya, obs["wavelength"])
    assert lint > 0

    # test igoring a line, phot only
    model = SpecModel(model_pars)
    (phot_nolya_2,), mfrac = model.predict(model.theta, [obslist[1]], sps=sps)
    obs["spectrum"] = obs_spec
    assert np.all(phot_nolya == phot_nolya_2)

    #import matplotlib.pyplot as pl
    #pl.ion()
    #fig, ax = pl.subplots()
    #ax.plot(obs["wavelength"], spec)
    #ax.plot(obs["wavelength"], spec_nolya)


#def test_marginalizing_lines():
    # test marginalizing over lines
    #model_pars["marginalize_elines"] = dict(init=True)
    #model = SpecModel(model_pars)
