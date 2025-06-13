#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" write_results.py - Methods for writing prospector ingredients and outputs
to HDF5 files as well as to pickles.
"""

import warnings
import pickle, json
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured

try:
    import h5py
    _has_h5py_ = True
except(ImportError):
    _has_h5py_ = False

__all__ = ["githash", "write_hdf5",
           "chain_to_struct"]


unserial = json.dumps('Unserializable')


class NumpyEncoder(json.JSONEncoder):
    """
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, type):
            return str(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)

        return json.JSONEncoder.default(self, obj)


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


def write_hdf5(hfile,
               config={},
               model=None,
               obs=None,
               sampling_result=None,
               optimize_result_tuple=None,
               write_model_params=True,
               sps=None,
               **extras):
    """Write output and information to an HDF5 file object (or
    group).

    hfile : string or `h5py.File`
        File to which results will be written.  Can be a string name or an
        `h5py.File` object handle.

    run_params : dict-like
        The dictionary of arguments used to build and fit a model.

    model : Instance of :py:class:`prospect.models.SpecModel`
        The  object.

    obs : list of Observation() instances
        The observations that were fit.

    sampling_result : EnsembleSampler() or dict
        The `emcee` sampler used to draw posterior samples or nested sampler
        output. Can be `None` if only optimization was performed.

    optimize_result_tuple : 2-tuple of (list, float)
        A list of `scipy.optimize.OptimizationResult` objects generated during
        the optimization stage, and a float giving the duration of sampling. Can
        be `None` if no optimization is performed

    sps : instance of :py:class:`prospect.sources.SSPBasis` (optional, default: None)
        If a `prospect.sources.SSPBasis` object is supplied, it will be used to
        generate and store best fit values (not implemented)
    """
    # If ``hfile`` is not a file object, assume it is a filename and open
    if isinstance(hfile, str):
        hf = h5py.File(hfile, "w")
    else:
        hf = hfile

    assert (model is not None), "Must pass a prospector model"
    run_params = config

    # ----------------------
    # Sampling info
    if run_params.get("emcee", False):
        chain, extras = emcee_to_struct(sampling_result, model)
    elif bool(run_params.get("nested_sampler", False)):
        chain, extras = nested_to_struct(sampling_result, model)
    else:
        chain, extras = None, None
    write_sampling_h5(hf, chain, extras)
    hf.flush()

    # ----------------------
    # Observational data
    if obs is not None:
        write_obs_to_h5(hf, obs)
        hf.flush()

    # ----------------------
    # High level parameter and version info
    meta = metadata(run_params, model, write_model_params=write_model_params)
    for k, v in meta.items():
        hf.attrs[k] = v
    hf.flush()

    # -----------------
    # Optimizer info
    if optimize_result_tuple is not None:
        optimize_list, toptimize = optimize_result_tuple
        optarr = optresultlist_to_ndarray(optimize_list)
        opt = hf.create_group('optimization')
        _ = opt.create_dataset('optimizer_results', data=optarr)
        opt.attrs["optimizer_duration"] = json.dumps(toptimize)

    # ---------------
    # Best fitting model in space of data
    if sps is not None:
        if "sampling/chain" in hf:
            pass
            #from ..plotting.utils import best_sample
            #pbest = best_sample(hf["sampling"])
            #predictions, mfrac = model.predict(pbest, obs=obs, sps=sps)
            #best = hf.create_group("bestfit")
            #best.create_dataset("spectrum", data=spec)
            #best.create_dataset("photometry", data=phot)
            #best.create_dataset("parameter", data=pbest)
            #best.attrs["mfrac"] = mfrac
            #if obs["wavelength"] is None:
            #    best.create_dataset("restframe_wavelengths", data=sps.wavelengths)

    # Store the githash last after flushing since getting it might cause an
    # uncatchable crash
    bgh = githash(**run_params)
    hf.attrs['prospector_version'] = json.dumps(bgh)
    hf.close()


def metadata(run_params, model, write_model_params=True):
    """Generate a metadata dictionary, with serialized entries.
    """
    meta = dict(run_params=run_params,
                paramfile_text=paramfile_string(**run_params))
    if write_model_params:
        from copy import deepcopy
        meta["model_params"] = deepcopy(model.params)
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
    lnpost = sampler.get_log_prob(flat=True)

    # chaincat & extras
    chaincat = chain_to_struct(samples, model=model)
    extras = dict(weights=None,
                  lnprobability=lnpost,
                  lnlike=lnpost - lnprior,
                  acceptance=sampler.acceptance_fraction,
                  rstate=sampler.random_state,
                  duration=sampler.getattr("duration", 0.0))

    return chaincat, extras


def nested_to_struct(nested_out, model):
    # preamble
    lnprior = model.prior_product(nested_out['points'])

    # chaincat & extras
    chaincat = chain_to_struct(nested_out['points'], model=model)
    extras = dict(weights=np.exp(nested_out['log_weight']),
                  lnprobability=nested_out['log_like'] + lnprior,
                  lnlike=nested_out['log_like'],
                  duration=nested_out.get("duration", 0.0)
                  )
    return chaincat, extras


def write_sampling_h5(hf, chain, extras):
    try:
        sdat = hf['sampling']
    except(KeyError):
        sdat = hf.create_group('sampling')

    sdat.create_dataset('chain', data=chain)
    try:
        uchain = structured_to_unstructured(chain)
        sdat.create_dataset("unstructured_chain", data=uchain)
    except:
        pass
    for k, v in extras.items():
        try:
            sdat.create_dataset(k, data=v)
        except:
            sdat.attrs[k] = v


def write_obs_to_h5(hf, obslist):
    """Write observational data to the hdf5 file
    """
    try:
        odat = hf.create_group('observations')
    except(ValueError):
        # We already have an 'obs' group
        return
    for obs in obslist:
        obs.to_h5_dataset(odat)
    hf.flush()


def optresultlist_to_ndarray(results):
    npar, nout = len(results[0].x), len(results[0].fun)
    dt = [("success", bool), ("message", "U50"), ("nfev", int),
          ("x", (float, npar)), ("fun", (float, nout))]
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
    if isinstance(chain, dict):
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

    # TODO: replace with unstructured_to_structured
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
