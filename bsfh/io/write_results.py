import os, subprocess, time
import pickle, json
import numpy as np
from ..models.parameters import functions_to_names, plist_to_pdict
try:
    import h5py
except:
    pass

__all__ = ["run_command", "githash", "write_pickles", "write_hdf5"]

def run_command(cmd):
    """Open a child process, and return its exit status and stdout.
    """
    child = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out = [s for s in child.stdout]
    w = child.wait()
    return os.WEXITSTATUS(w), out


def githash(nofork=False, **extras):
    """Pull out the git hash history for bsfh here.

    :param nofork: (optional, default: False)
        If ``True``, do *not* get the githash, since this involves creating a
        fork, which can cause a problem on some MPI implementations (in a way
        that cannot be caught niceley)
    """
    cmd = 'cd {0}; git log --format="format:%h"'
    if not nofork:
        try:
            bsfh_dir = os.path.dirname(__file__)
            bgh = run_command(cmd.format(bsfh_dir))[1]
        except:
            print("Couldn't get Prospector git hash")
            bgh = "Can't get hash for some reason"
    else:
        bgh = "Can't check hash (nofork=True)."

    return bgh


def write_pickles(run_params, model, obs, sampler, powell_results,
                  outroot=None, tsample=None, toptimize=None,
                  post_burnin_center=None, post_burnin_prob=None,
                  sampling_initial_center=None, **extras):
    """Write results to two different pickle files.  One (``*_mcmc``) contains
    only lists, dictionaries, and numpy arrays and is therefore robust to
    changes in object definitions.  The other (``*_model``) contains the actual
    model object (and minimization result objects) and is therefore more
    fragile.
    """

    bgh = githash(**run_params)

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

    results['bsfh_version'] = bgh
    results['paramfile_text'] = paramfile_string(**run_params)
    
    if outroot is None:
        tt = int(time.time())
        outroot = '{1}_{0}'.format(tt, run_params['outfile'])
    
    write_model_pickle(outroot + '_model', model, bgh=bgh, powell=powell_results,
                       paramfile_text=results['paramfile_text'])
    with open(outroot + '_mcmc', "wb") as out:
        pickle.dump(results, out)


def write_model_pickle(outname, model, bgh=None, powell=None, **kwargs):
    model_store = {}
    model_store['powell'] = powell
    model_store['model'] = model
    model_store['bsfh_version'] = bgh
    for k, v in kwargs.items():
        try:
            model_store[k] = v
        except:
            pass
    with open(outname, "wb") as out:
        pickle.dump(model_store, out)


def paramfile_string(param_file=None, **extras):
    pstr = ''
    try:
        with open(param_file, "r") as pfile:
            pstr = pfile.read()
    except:
        pass
    return pstr

        
def write_hdf5(hfile, run_params, model, obs, sampler, powell_results,
               tsample=0.0, toptimize=0.0, sampling_initial_center=None,
               mfile=None, **extras):
    """Write output and information to an HDF5 file object (or
    group).  
    """
    try:
        # If ``hfile`` is not a file object, assume it is a filename and open
        hf = h5py.File(hfile, "a")
    except(AttributeError):
        hf = hfile

    unserial = json.dumps('Unserializable')
    # ----------------------
    # High level parameter and version info
    serialize = {'run_params': run_params,
                 'model_params': [functions_to_names(p.copy()) for p in model.config_list],
                 'paramfile_text': paramfile_string(**run_params)}
    for k, v in list(serialize.items()):
        try:
            hf.attrs[k] = json.dumps(v)
        except(TypeError):
            hf.attrs[k] = unserial
            print("Could not serialize {}".format(k))
    hf.attrs['optimizer_duration'] = json.dumps(toptimize)
    hf.flush()

    # ----------------------
    # Sampling info
    try:
        sdat = hf['sampling']
    except(KeyError):
        sdat = hf.create_group('sampling')
        sdat.create_dataset('chain', data=sampler.chain)
        sdat.create_dataset('lnprobability', data=sampler.lnprobability)
    sdat.create_dataset('acceptance', data=sampler.acceptance_fraction)
    sdat.create_dataset('sampling_initial_center', data=sampling_initial_center)
    sdat.create_dataset('initial_theta', data=model.initial_theta.copy())
    # JSON Attrs    
    sdat.attrs['rstate'] = pickle.dumps(sampler.random_state)
    sdat.attrs['sampling_duration'] = json.dumps(tsample)
    sdat.attrs['theta_labels'] = json.dumps(list(model.theta_labels()))
    hf.flush()

    # ----------------------
    # Observational data
    odat = hf.create_group('obs')
    for k, v in list(obs.items()):
        if k == 'filters':
            try:
                v = [f.name for f in v]
            except:
                pass
        if type(v) is np.ndarray:
            odat.create_dataset(k, data=v)
        else:
            try:
                odat.attrs[k] = json.dumps(v)
            except(TypeError):
                odat.attrs[k] = unserial
                print("Could not serialize {}".format(k))
    hf.flush()
    # Store the githash last after flushing since getting it might cause an
    # uncatchable crash
    bgh = githash(**run_params)
    hf.attrs['bsfh_version'] = json.dumps(bgh)
    hf.close()

    #if mfile is None:
    #    mfile = hf.name.replace('.h5', '_model')
    #write_model_pickle(outroot + '_model', model, bgh=bgh, powell=powell_results)
