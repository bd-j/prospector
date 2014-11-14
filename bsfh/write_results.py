import pickle, os, subprocess, time
import numpy as np
from bsfh import modeldef

def run_command(cmd):
    """
    Open a child process, and return its exit status and stdout
    """
    child = subprocess.Popen(cmd, shell =True, stderr = subprocess.PIPE, 
                             stdin=subprocess.PIPE, stdout = subprocess.PIPE)
    out = [s for s in child.stdout]
    w = child.wait()
    return os.WEXITSTATUS(w), out


def write_pickles(parset, model, sampler, powell_results,
                  tsample=None, toptimize=None,
                  sampling_initial_center=None):

    """Write results to two different pickle files.  One (``*_mcmc``)
    contains only lists, dictionaries, and numpy arrays and is
    therefore robust to changes in object definitions.  The other
    (``*_model``) contains the actual model object (and minimization
    result objects) and is therefore more fragile.
    """

    #pull out the git hash for bsfh here.
    bsfh_dir = os.path.dirname(modeldef.__file__)
    bgh = run_command('cd {0}\n git rev-parse HEAD'.format(bsfh_dir)
                      )[1][0].replace('\n','')

    results, model_store = {}, {}
    
    results['run_params'] = parset.run_params
    results['obs'] = model.obs
    results['model_params'] = [modeldef.functions_to_names(p) for p in parset.model_params]
    results['model_params_asdict'] =  modeldef.plist_to_pdict([modeldef.functions_to_names(p)
                                                               for p in parset.model_params])
    results['initial_theta'] = parset.initial_theta
    results['sampling_initial_center'] = sampling_initial_center
    
    results['chain'] = sampler.chain
    results['lnprobability'] = sampler.lnprobability
    results['acceptance'] = sampler.acceptance_fraction
    results['sampling_duration'] = tsample
    results['optimizer_duration'] = toptimize
    results['bsfh_version'] = bgh

    model_store['powell'] = powell_results
    model_store['model'] = model
    model_store['parset'] = parset
    model_store['bsfh_version'] = bgh
    
    #prospectr_dir = 
    #cgh = run_command('git rev-parse HEAD')[1][0].replace('\n','')
    #results['cetus_version'] = cgh
    
    tt = int(time.time())
    out = open('{1}_{0}_mcmc'.format(tt, parset.run_params['outfile']), 'wb')
    pickle.dump(results, out)
    out.close()

    out = open('{1}_{0}_model'.format(tt, parset.run_params['outfile']), 'wb')
    pickle.dump(model_store, out)
    out.close()
