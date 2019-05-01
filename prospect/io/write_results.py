import os, time, warnings
import pickle, json, base64
import numpy as np
try:
    import h5py
    _has_h5py_ = True
except(ImportError):
    _has_h5py_ = False


__all__ = ["run_command", "githash", "write_pickles", "write_hdf5"]


unserial = json.dumps('Unserializable')


def pick(obj):
    """create a serialized object that can go into hdf5 in py2 and py3, and can be read by both
    """
    return np.void(pickle.dumps(obj, 0))


#def run_command(cmd):
#    """Open a child process, and return its exit status and stdout.
#    """
#    import subprocess
#    child = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE,
#                             stdin=subprocess.PIPE, stdout=subprocess.PIPE)
#    out = [s for s in child.stdout]
#    w = child.wait()
#    return os.WEXITSTATUS(w), out


def githash(**extras):
    """Pull out the git hash history for Prospector here.
    """
    try:
        from .._version import __version__, __githash__
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


def write_hdf5(hfile, run_params, model, obs, sampler, optimize_result_list,
               tsample=0.0, toptimize=0.0, sampling_initial_center=[],
               **extras):
    """Write output and information to an HDF5 file object (or
    group).
    """
    try:
        # If ``hfile`` is not a file object, assume it is a filename and open
        hf = h5py.File(hfile, "a")
    except(AttributeError,TypeError):
        hf = hfile
    except(NameError):
        warnings.warn("HDF5 file could not be opened, as h5py could not be imported.")
        return

    # ----------------------
    # Sampling info
    try:
        # emcee
        a = sampler.acceptance_fraction
        write_emcee_h5(hf, sampler, model, sampling_initial_center, tsample)
    except(AttributeError):
        # dynesty or nestle
        if sampler is None:
            sdat = hf.create_group('sampling')
        elif 'eff' in sampler:
            write_dynesty_h5(hf, sampler, model, tsample)
        else:
            write_nestle_h5(hf, sampler, model, tsample)

    # -----------------
    # Optimizer info
    if optimize_result_list is not None:
        out = optresultlist_to_ndarray(optimize_result_list)
        mdat = hf.create_dataset('optimization', data=out)
        
    
    # ----------------------
    # High level parameter and version info
    write_h5_header(hf, run_params, model)
    hf.attrs['optimizer_duration'] = json.dumps(toptimize)
    hf.flush()

    # ----------------------
    # Observational data
    write_obs_to_h5(hf, obs)

    # Store the githash last after flushing since getting it might cause an
    # uncatchable crash
    bgh = githash(**run_params)
    hf.attrs['prospector_version'] = json.dumps(bgh)
    hf.close()


def write_emcee_h5(hf, sampler, model, sampling_initial_center, tsample):
    """Write emcee information to the provided HDF5 file in the `sampling`
    group.
    """
    try:
        sdat = hf['sampling']
    except(KeyError):
        sdat = hf.create_group('sampling')
    if 'chain' not in sdat:
        sdat.create_dataset('chain',
                            data=sampler.chain)
        sdat.create_dataset('lnprobability',
                            data=sampler.lnprobability)
    sdat.create_dataset('acceptance',
                        data=sampler.acceptance_fraction)
    sdat.create_dataset('sampling_initial_center',
                        data=sampling_initial_center)
    sdat.create_dataset('initial_theta',
                        data=model.initial_theta.copy())
    # JSON Attrs
    sdat.attrs['rstate'] = pick(sampler.random_state)
    sdat.attrs['sampling_duration'] = json.dumps(tsample)
    sdat.attrs['theta_labels'] = json.dumps(list(model.theta_labels()))

    hf.flush()


def write_nestle_h5(hf, nestle_out, model, tsample):
    """Write nestle results to the provided HDF5 file in the `sampling` group.
    """
    try:
        sdat = hf['sampling']
    except(KeyError):
        sdat = hf.create_group('sampling')
    sdat.create_dataset('chain',
                        data=nestle_out['samples'])
    sdat.create_dataset('weights',
                        data=nestle_out['weights'])
    sdat.create_dataset('lnlikelihood',
                        data=nestle_out['logl'])
    sdat.create_dataset('lnprobability',
                        data=(nestle_out['logl'] +
                              model.prior_product(nestle_out['samples'])))
    sdat.create_dataset('logvol',
                        data=nestle_out['logvol'])
    sdat.create_dataset('logz',
                        data=np.atleast_1d(nestle_out['logz']))
    sdat.create_dataset('logzerr',
                        data=np.atleast_1d(nestle_out['logzerr']))
    sdat.create_dataset('h_information',
                        data=np.atleast_1d(nestle_out['h']))

    # JSON Attrs
    for p in ['niter', 'ncall']:
        sdat.attrs[p] = json.dumps(nestle_out[p])
    sdat.attrs['theta_labels'] = json.dumps(list(model.theta_labels()))
    sdat.attrs['sampling_duration'] = json.dumps(tsample)

    hf.flush()

def write_dynesty_h5(hf, dynesty_out, model, tsample):
    """Write nestle results to the provided HDF5 file in the `sampling` group.
    """
    try:
        sdat = hf['sampling']
    except(KeyError):
        sdat = hf.create_group('sampling')

    sdat.create_dataset('chain',
                        data=dynesty_out['samples'])
    sdat.create_dataset('weights',
                        data=np.exp(dynesty_out['logwt']-dynesty_out['logz'][-1]))
    sdat.create_dataset('logvol',
                        data=dynesty_out['logvol'])
    sdat.create_dataset('logz',
                        data=np.atleast_1d(dynesty_out['logz']))
    sdat.create_dataset('logzerr',
                        data=np.atleast_1d(dynesty_out['logzerr']))
    sdat.create_dataset('information',
                        data=np.atleast_1d(dynesty_out['information']))
    sdat.create_dataset('lnlikelihood',
                        data=dynesty_out['logl'])
    sdat.create_dataset('lnprobability',
                        data=(dynesty_out['logl'] +
                              model.prior_product(dynesty_out['samples'].flatten())))
    sdat.create_dataset('efficiency',
                        data=np.atleast_1d(dynesty_out['eff']))
    sdat.create_dataset('niter',
                        data=np.atleast_1d(dynesty_out['niter']))
    sdat.create_dataset('samples_id',
                        data=np.atleast_1d(dynesty_out['samples_id']))

    # JSON Attrs
    sdat.attrs['ncall'] = json.dumps(dynesty_out['ncall'].tolist())
    sdat.attrs['theta_labels'] = json.dumps(list(model.theta_labels()))
    sdat.attrs['sampling_duration'] = json.dumps(tsample)

    hf.flush()


def write_h5_header(hf, run_params, model):
    """Write header information about the run.
    """
    serialize = {'run_params': run_params,
                 'model_params': [functions_to_names(p.copy())
                                  for p in model.config_list],
                 'paramfile_text': paramfile_string(**run_params)}
    for k, v in list(serialize.items()):
        try:
            hf.attrs[k] = json.dumps(v)  #, cls=NumpyEncoder)
        except(TypeError):
            # Should this fall back to pickle.dumps?
            hf.attrs[k] = pick(v)
            warnings.warn("Could not JSON serialize {}, pickled instead".format(k),
                          RuntimeWarning)
        except:
            hf.attrs[k] = unserial
            warnings.warn("Could not serialize {}".format(k), RuntimeWarning)
    hf.flush()


def write_obs_to_h5(hf, obs):
    """Write observational data to the hdf5 file
    """
    try:
        odat = hf.create_group('obs')
    except(ValueError):
        # We already have an 'obs' group
        return
    for k, v in list(obs.items()):
        if k == 'filters':
            try:
                v = [f.name for f in v]
            except:
                pass
        if isinstance(v, np.ndarray):
            odat.create_dataset(k, data=v)
        else:
            try:
                odat.attrs[k] = json.dumps(v)  #, cls=NumpyEncoder)
            except(TypeError):
                # Should this fall back to pickle.dumps?
                odat.attrs[k] = pick(v)
                warnings.warn("Could not JSON serialize {}, pickled instead".format(k))
            except:
                odat.attrs[k] = unserial
                warnings.warn("Could not serialize {}".format(k))

    hf.flush()


def optresultlist_to_ndarray(results):
    npar, nout = len(results[0].x), len(results[0].fun)
    dt = [("success", np.bool), ("message", "S50"), ("nfev", np.int), ("x", (np.float, npar)), ("fun", (np.float, nout))]
    out = np.zeros(len(results), dtype=np.dtype(dt))
    for i, r in enumerate(results):
        for f in out.dtype.names:
            out[i][f] = r[f]

    return out

        
def write_pickles(run_params, model, obs, sampler, powell_results,
                  outroot=None, tsample=None, toptimize=None,
                  post_burnin_center=None, post_burnin_prob=None,
                  sampling_initial_center=None, simpleout=False, **extras):
    """Write results to two different pickle files.  One (``*_mcmc``) contains
    only lists, dictionaries, and numpy arrays and is therefore robust to
    changes in object definitions.  The other (``*_model``) contains the actual
    model object (and minimization result objects) and is therefore more
    fragile.
    """

    if outroot is None:
        tt = int(time.time())
        outroot = '{1}_{0}'.format(tt, run_params['outfile'])
    bgh = githash(**run_params)
    paramfile_text = paramfile_string(**run_params)

    write_model_pickle(outroot + '_model', model, bgh=bgh, powell=powell_results,
                       paramfile_text=paramfile_text)

    if simpleout and _has_h5py_:
        return

    # write out a simple chain as a pickle.  This isn't really necessary since
    # the hd5 usually works
    results = {}

    # Useful global info and parameters
    results['run_params'] = run_params
    results['obs'] = obs
    results['model_params'] = [functions_to_names(p.copy()) for p in model.config_list]
    results['theta_labels'] = list(model.theta_labels())

    # Parameter value at variopus phases
    results['initial_theta'] = model.initial_theta
    results['sampling_initial_center'] = sampling_initial_center
    results['post_burnin_center'] = post_burnin_center
    results['post_burnin_prob'] = post_burnin_prob

    # Chain and ancillary sampling info
    results['chain'] = sampler.chain
    results['lnprobability'] = sampler.lnprobability
    results['acceptance'] = sampler.acceptance_fraction
    results['rstate'] = sampler.random_state
    results['sampling_duration'] = tsample
    results['optimizer_duration'] = toptimize

    results['prospector_version'] = bgh
    results['paramfile_text'] = paramfile_text

    with open(outroot + '_mcmc', "wb") as out:
        pickle.dump(results, out)


def write_model_pickle(outname, model, bgh=None, powell=None, **kwargs):
    model_store = {}
    model_store['powell'] = powell
    model_store['model'] = model
    model_store['prospector_version'] = bgh
    for k, v in kwargs.items():
        try:
            model_store[k] = v
        except:
            pass
    with open(outname, "wb") as out:
        pickle.dump(model_store, out)


def functions_to_names(p):
    """Replace prior and dust functions (or objects) with the names of those
    functions (or pickles).
    """
    for k, v in list(p.items()):
        if callable(v):
            try:
                p[k] = [v.__name__, v.__module__]
            except(AttributeError):
                p[k] = pickle.dumps(v, protocol=2)
    return p
