#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" write_results.py - Methods for writing prospector ingredients and outputs
to HDF5 files as well as to pickles.
"""

import os, time, warnings
from copy import deepcopy
import pickle, json, base64
import numpy as np
try:
    import h5py
    _has_h5py_ = True
except(ImportError):
    _has_h5py_ = False

from .. import NumpyEncoder

__all__ = ["githash", "write_hdf5",
           "chain_to_struct"]


unserial = json.dumps('Unserializable')


def pick(obj):
    """create a serialized object that can go into hdf5 in py2 and py3, and can be read by both
    """
    return np.void(pickle.dumps(obj, 0))


def githash(**extras):
    """Pull out the git hash history for Prospector here.
    """
    try:
        from .._version import __version__#, __githash__
        __githash__ = None
        bgh = __version__, __githash__
    except(ImportError):
        warnings.warn("Could not obtain prospector version info", RuntimeWarning)
        bgh = "Can't get version number."

    return bgh


def paramfile_string(param_file=None, **extras):
    try:
        with open(param_file, "r") as pfile:
            pstr = pfile.read()
    except:
        warnings.warn("Could not store paramfile text", RuntimeWarning)
        pstr = ''
    return pstr


def write_hdf5(hfile, run_params, model, obs,
               sampler=None,
               optimize_result_list=None,
               tsample=0.0, toptimize=0.0,
               sampling_initial_center=[],
               sps=None, **extras):
    """Write output and information to an HDF5 file object (or
    group).

    :param hfile:
        File to which results will be written.  Can be a string name or an
        `h5py.File` object handle.

    :param run_params:
        The dictionary of arguments used to build and fit a model.

    :param model:
        The `prospect.models.SedModel` object.

    :param obs:
        The dictionary of observations that were fit.

    :param sampler:
        The `emcee` or `dynesty` sampler object used to draw posterior samples.
        Can be `None` if only optimization was performed.

    :param optimize_result_list:
        A list of `scipy.optimize.OptimizationResult` objects generated during
        the optimization stage.  Can be `None` if no optimization is performed

    param sps: (optional, default: None)
        If a `prospect.sources.SSPBasis` object is supplied, it will be used to
        generate and store
    """
    # If ``hfile`` is not a file object, assume it is a filename and open
    if type(hfile) is str:
        hf = h5py.File(hfile, "w")
    else:
        hf = hfile

    # ----------------------
    # Sampling info
    if run_params.get("emcee", False):
        chain, extras = emcee_to_struct(sampler, model)
    elif run_params.get("dynesty", False):
        chain, extras = dynesty_to_struct(sampler, model)
    else:
        chain, extras = None, None
    write_sampling_h5(hf, chain, extras)
    hf.flush()

    # ----------------------
    # High level parameter and version info
    meta = metadata(run_params, model)
    for k, v in meta.items():
        hf.attrs[k] = k
    hf.flush()

    # -----------------
    # Optimizer info
    hf.attrs['optimizer_duration'] = json.dumps(toptimize)
    if optimize_result_list is not None:
        out = optresultlist_to_ndarray(optimize_result_list)
        mgroup = hf.create_group('optimization')
        mdat = mgroup.create_dataset('optimizer_results', data=out)

    # ----------------------
    # Observational data
    write_obs_to_h5(hf, obs)
    hf.flush()

    # ---------------
    # Best fitting model in space of data
    if sps is not None:
        if "sampling/chain" in hf:
            from ..plotting.utils import best_sample
            pbest = best_sample(hf["sampling"])
            spec, phot, mfrac = model.predict(pbest, obs=obs, sps=sps)
            best = hf.create_group("bestfit")
            best.create_dataset("spectrum", data=spec)
            best.create_dataset("photometry", data=phot)
            best.create_dataset("parameter", data=pbest)
            best.attrs["mfrac"] = mfrac
            if obs["wavelength"] is None:
                best.create_dataset("restframe_wavelengths", data=sps.wavelengths)

    # Store the githash last after flushing since getting it might cause an
    # uncatchable crash
    bgh = githash(**run_params)
    hf.attrs['prospector_version'] = json.dumps(bgh)
    hf.close()


def metadata(run_params, model):
    meta = dict(run_params=run_params,
                paramfile_text=paramfile_string(**run_params),
                model_params=deepcopy(model.params)
                )
    for k, v in list(meta.items()):
        try:
            meta[k] = json.dumps(v, cls=NumpyEncoder)
        except(TypeError):
            meta[k] = pick(v)
        except:
            meta[k] = unserial

    return meta


def emcee_to_struct(sampler, model):
    # preamble
    samples = sampler.get_chain(flat=True)
    lnprior = model.prior_product(samples)

    # chaincat & extras
    chaincat = chain_to_struct(samples, model=model)
    extras = dict(weights=None,
                  lnprobability=sampler.get_log_prob(flat=True),
                  lnlike=sampler.get_log_prob(flat=True) - lnprior,
                  acceptance=sampler.acceptance_fraction,
                  rstate=sampler.random_state)

    return chaincat, extras


def dynesty_to_struct(dyout, model):
    # preamble
    lnprior = model.prior_product(dyout['samples'])

    # chaincat & extras
    chaincat = chain_to_struct(dyout["samples"], model=model)
    extras = dict(weights=np.exp(dyout['logwt']-dyout['logz'][-1]),
                  lnprobability=dyout['logl'] + lnprior,
                  lnlike=dyout['logl'],
                  efficiency=np.atleast_1d(dyout['eff']),
                  logz=np.atleast_1d(dyout['logz']),
                  ncall=json.dumps(dyout['ncall'].tolist())
                 )
    return chaincat, extras


def write_sampling_h5(hf, chain, extras):
    try:
        sdat = hf['sampling']
    except(KeyError):
        sdat = hf.create_group('sampling')

    sdat.create_dataset('chain', data=chain)
    for k, v in extras.items():
        try:
            sdat.create_dataset(k, data=v)
        except:
            sdat.attrs[k] = v


def write_obs_to_h5(hf, obslist):
    """Write observational data to the hdf5 file
    """
    try:
        odat = hf.create_group('obs')
    except(ValueError):
        # We already have an 'obs' group
        return
    for obs in obslist:
        obs.to_h5_dataset(odat)
    hf.flush()


def optresultlist_to_ndarray(results):
    npar, nout = len(results[0].x), len(results[0].fun)
    dt = [("success", np.bool), ("message", "S50"), ("nfev", np.int),
          ("x", (np.float, npar)), ("fun", (np.float, nout))]
    out = np.zeros(len(results), dtype=np.dtype(dt))
    for i, r in enumerate(results):
        for f in out.dtype.names:
            out[i][f] = r[f]

    return out


def chain_to_struct(chain, model=None, names=None, **extras):
    """Given a (flat)chain (or parameter dictionary) and a model, convert the
    chain to a structured array

    Parameters
    ----------
    chain : ndarry of shape (nsamples, ndim)
        A chain or a dictionary of parameters, values of which are numpy
        datatypes.

    model : A ProspectorParams instance

    names : list of strings

    extras : optional
        Extra keyword arguments are assumed to be 1d ndarrays of type np.float64
        and shape (nsamples,) that will be added as additional fields of the
        output structure

    Returns
    -------
    struct :
        A structured ndarray of parameter values.
    """
    indict = type(chain) == dict
    if indict:
        return dict_to_struct(chain)
    else:
        n = np.prod(chain.shape[:-1])
        assert model.ndim == chain.shape[-1]

    if model is not None:
        model.set_parameters(chain[0])
        names = model.free_params
        dt = [(p, model.params[p].dtype, model.params[p].shape)
              for p in names]
    else:
        dt = [(str(p), "<f8", (1,)) for p in names]

    dt += [(str(k), "<f8") for k in extras.keys()]

    struct = np.zeros(n, dtype=np.dtype(dt))
    for i, p in enumerate(names):
        if model is not None:
            inds = model.theta_index[p]
        else:
            inds = slice(i, i+1, None)
        struct[p] = chain[..., inds].reshape(-1, model.params[p].shape[0])

    for k, v in extras.items():
        try:
            struct[k] = v
        except(ValueError, IndexError):
            pass

    return struct


def dict_to_struct(indict):
    dt = [(p, indict[p].dtype.descr[0][1], indict[p].shape)
          for p in indict.keys()]
    struct = np.zeros(1, dtype=np.dtype(dt))
    for i, p in enumerate(indict.keys()):
        struct[p] = indict[p]
    return struct
