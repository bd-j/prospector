import pickle, os, subprocess, time
import numpy as np
from ..models.parameters import functions_to_names, plist_to_pdict
import json

__all__ = ["run_command", "write_pickles"]

def run_command(cmd):
    """Open a child process, and return its exit status and stdout.
    """
    child = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out = [s for s in child.stdout]
    w = child.wait()
    return os.WEXITSTATUS(w), out


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

    # Pull out the git hash for bsfh here.
    nofork = run_params.get('nofork', False)
    if not nofork:
        try:
            bsfh_dir = os.path.dirname(__file__)
            bgh = run_command('cd {0}\n git rev-parse HEAD'.format(bsfh_dir)
                          )[1][0].replace('\n', '')
        except:
            print("Couldn't get Prospector git hash")
            bgh = ''
    else:
        bgh = ''

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


def hdf5_output(run_params, model, obs, sampler, powell_results,
                tsample=None, toptimize=None, sampling_initial_center=None):
    tt = int(time.time())
    filename = '{1}_{0}.mcmc.h5'.format(tt, run_params['outfile'])
    with h5py.File(filename) as out:
        obs = json.dumps(obs)
