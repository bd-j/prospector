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
    """Pull out the git hash for bsfh here.
    """
    if not nofork:
        try:
            bsfh_dir = os.path.dirname(__file__)
            bgh = run_command('cd {0}\n git rev-parse HEAD'.format(bsfh_dir)
                          )[1][0].replace('\n', '')
        except:
            print("Couldn't get Prospector git hash")
            bgh = "Can't get hash for some reason"
    else:
        bgh = "Can't check hash (nofork=True)."

    return bgh


def write_pickles(run_params, model, obs, sampler, powell_results,
                  tsample=None, toptimize=None,
                  sampling_initial_center=None,
                  post_burnin_center=None, post_burnin_prob=None):
    """Write results to two different pickle files.  One (``*_mcmc``) contains
    only lists, dictionaries, and numpy arrays and is therefore robust to
    changes in object definitions.  The other (``*_model``) contains the actual
    model object (and minimization result objects) and is therefore more
    fragile.
    """

    bgh = githash(**run_params)

    results, model_store = {}, {}

    results['run_params'] = run_params
    results['obs'] = obs
    results['model_params'] = [functions_to_names(p) for p in model.config_list]
    results['model_params_dict'] = plist_to_pdict([functions_to_names(p)
                                                   for p in model.config_list])
    results['initial_theta'] = model.initial_theta
    results['sampling_initial_center'] = sampling_initial_center
    results['post_burnin_center'] = post_burnin_center
    results['post_burnin_prob'] = post_burnin_prob

    results['chain'] = sampler.chain
    results['lnprobability'] = sampler.lnprobability
    results['acceptance'] = sampler.acceptance_fraction
    results['rstate'] = sampler.random_state
    results['sampling_duration'] = tsample
    results['optimizer_duration'] = toptimize
    results['bsfh_version'] = bgh

    model_store['powell'] = powell_results
    model_store['model'] = model
    model_store['bsfh_version'] = bgh

    # prospectr_dir =
    # cgh = run_command('git rev-parse HEAD')[1][0].replace('\n','')
    # results['cetus_version'] = cgh

    tt = int(time.time())
    with open('{1}_{0}_mcmc'.format(tt, run_params['outfile']), 'wb') as out:
        pickle.dump(results, out)

    with open('{1}_{0}_model'.format(tt, run_params['outfile']), 'wb') as out:
        pickle.dump(model_store, out)


def write_hdf5(hf, run_params, model, obs, sampler, powell_results,
               tsample=0.0, toptimize=0.0, sampling_initial_center=None):
    """Write output and information to an already open HDF5 file object (or
    group)
    """

    # ----------------------
    # High level parameter and version info
    bgh = githash(**run_params)
    serialize = {'run_params': run_params,
                 'model_params': [functions_to_names(p) for p in model.config_list],
                 'bsfh_version': bgh,
                 }

    for k, v in list(serialize.items()):
        try:
            hf.attrs[k] = json.dumps(v)
        except(TypeError):
            hf.attrs[k] = 'Unserializable'
            print("Could not serialize {}".format(k))

    hf.attrs['optimizer_duration'] = toptimize
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
    sdat.attrs['rstate'] = sampler.random_state
    sdat.attrs['sampling_duration'] = tsample
    hf.flush()

    # ----------------------
    # Observational data
    dnames = ['wavelength', 'spectrum', 'unc', 'mask',
              'maggies', 'maggies_unc', 'phot_mask']
    odat = hf.create_group('obs')
    for k, v in list(obs.items):
        if k == 'filters':
            try:
                v = [f.name for f in v]
            except:
                pass
        if v in dnames:
            odat.create_dataset(k, data=v)
        else:
            try:
                odat.attrs[k] = json.dumps(v)
            except(TypeError):
                odat.attrs[k] = 'Unserializable'
                print("Could not serialize {}".format(k))
    
    hf.close()
