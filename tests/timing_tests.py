#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
from sedpy import observate

from prospect import prospect_args
from prospect.utils.obsutils import fix_obs
from prospect.models.templates import TemplateLibrary
from prospect.models.sedmodel import SpecModel

from prospect.sources import CSPSpecBasis, FastStepBasis


def build_model(nebular=True):
    model_pars = TemplateLibrary["continuity_sfh"]
    if nebular:
        model_pars.update(TemplateLibrary["nebular"])
    Basis = FastStepBasis
    return SpecModel(model_pars), Basis(zcontinuous=1)


def build_obs(spec=False):
    fnames = [f"sdss_{b}0" for b in "ugriz"]
    filts = observate.load_filters(fnames, gridded=True)
    obs = dict(filters=filts,
               maggies=np.zeros(len(fnames)),
               maggies_unc=np.ones(len(fnames)),
               wavelength=np.linspace(3000, 9000, 1000),
               spectrum=np.ones(1000),
               unc=np.ones(1000)*0.1)
    if not spec:
        obs["spectrum"] = None
    obs = fix_obs(obs)

    try:
        from prospect.observation import from_oldstyle
        obs = from_oldstyle(obs)
    except(ImportError):
        pass

    return obs


def predict(model, obs, sps):
    z = np.random.uniform(-1, 0)
    model.params["logzsol"] = np.array([z])
    preds, x = model.predict(model.theta, observations=obs, sps=sps)
    return preds


if __name__ == "__main__":

    obs = build_obs()
    model, sps = build_model(nebular=False)

    # make sure models get cached
    logzsol_prior = model.config_dict["logzsol"]['prior']
    lo, hi = logzsol_prior.range
    logzsol_grid = np.around(np.arange(lo, hi, step=0.1), decimals=2)
    for logzsol in logzsol_grid:
        model.params["logzsol"] = np.array([logzsol])
        _ = model.predict(model.theta, observations=obs, sps=sps)

    N = 100
    tic = time.time()
    for i in range(N):
        _ = predict(model, obs, sps)
    toc = time.time()
    print((toc-tic) / N)