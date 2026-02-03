#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np

#import pytest

from prospect.utils.prospect_args import get_parser
from prospect.sources import CSPSpecBasis
from prospect.models import SpecModel, templates
from prospect.observation import Photometry
from prospect.fitting import fit_model
from prospect.fitting.fitting import run_nested
from prospect.io.write_results import write_hdf5
from prospect.io.read_results import results_from

#@pytest.fixture
def get_sps(**kwargs):
    sps = CSPSpecBasis(zcontinuous=1)
    return sps


def build_model(add_neb=False, add_outlier=False, **kwargs):
    model_params = templates.TemplateLibrary["parametric_sfh"]
    model_params["logzsol"]["isfree"] = False  # built for speed
    if add_neb: # skip for speed
        model_params.update(templates.TemplateLibrary["nebular"])
    if add_outlier:
        model_params.update(templates.TemplateLibrary["outlier_model"])
        model_params["f_outlier_phot"]["isfree"] = True
        model_params["f_outlier_phot"]["init"] = 0.05
    return SpecModel(model_params)


def build_obs(**kwargs):

    from astroquery.sdss import SDSS
    from astropy.coordinates import SkyCoord
    bands = "ugriz"
    mcol = [f"cModelMag_{b}" for b in bands]
    ecol = [f"cModelMagErr_{b}" for b in bands]
    cat = SDSS.query_crossid(SkyCoord(ra=204.46376, dec=35.79883, unit="deg"),
                             data_release=16,
                             photoobj_fields=mcol + ecol + ["specObjID"])
    shdus = SDSS.get_spectra(plate=2101, mjd=53858, fiberID=220)[0]
    assert int(shdus[2].data["SpecObjID"][0]) == cat[0]["specObjID"]

    fnames = [f"sdss_{b}0" for b in bands]
    maggies = np.array([10**(-0.4 * cat[0][f"cModelMag_{b}"]) for b in bands])
    magerr = np.array([cat[0][f"cModelMagErr_{b}"] for b in bands])
    magerr = np.hypot(magerr, 0.05)

    phot = Photometry(filters=fnames, flux=maggies, uncertainty=magerr*maggies/1.086,
                      name=f'sdss_phot_specobjID{cat[0]["specObjID"]}')

    obslist = [phot]
    [obs.rectify() for obs in obslist]
    return obslist


if __name__ == "__main__":

    parser = get_parser()
    parser.set_defaults(nested_target_n_effective=256,
                        nested_nlive=512,
                        verbose=0)
    args = parser.parse_args()
    run_params = vars(args)
    run_params["param_file"] = __file__

    # build stuff
    model = build_model()
    obs = build_obs()
    sps = get_sps()

    # do a first model caching
    (phot,), mfrac = model.predict(model.theta, obs, sps=sps)
    print(model)

    # test passing sampler-specific kwargs
    sampler = "nautilus"
    run_params["nested_sampler"] = sampler
    run_params["n_like_max"] = 100 # speed
    # just to make sure we will get out exactly n_like_max points
    assert run_params["nested_nlive"] > run_params["n_like_max"]
    out = run_nested(obs, model, sps, return_sampler_result_object=True, **run_params)
    _ = run_params.pop("n_like_max")
    res = out["sampler_result_object"]
    assert res.n_like == 100
    assert len(out["points"]) == 100

    # loop over samplers
    results = {}
    samplers = ["nautilus", "ultranest", "dynesty"]
    for sampler in samplers:
        print(sampler)
        run_params["nested_sampler"] = sampler
        out = fit_model(obs, model, sps, **run_params)
        results[sampler] = out["sampling"]
        hfile = f"./{sampler}_test.h5"
        write_hdf5(hfile,
                   run_params,
                   model,
                   obs,
                   results[sampler],
                   None,
                   sps=sps)
        ires, iobs, im = results_from(hfile)
        assert (im is not None)

    # compare runtime
    for sampler in samplers:
        print(sampler, results[sampler]["duration"])

    # compare posteriors
    colors = ["royalblue", "darkorange", "firebrick"]
    import matplotlib.pyplot as pl
    from prospect.plotting import corner
    ndim = model.ndim
    cfig, axes = pl.subplots(ndim, ndim, figsize=(10,9))
    for sampler, color in zip(samplers, colors):
        out = results[sampler]
        axes = corner.allcorner(out["points"].T,
                                model.theta_labels(),
                                axes,
                                color=color,
                                weights= np.exp(out["log_weight"]),
                                show_titles=True)
    cfig.savefig("sampler_test_corner.png")
