import sys, os, getopt, json
from copy import deepcopy
import numpy as np
from bsfh import modeldef, sedmodel
from bsfh import datautils as dutils

"""This module has methods to take a .json or .py file containing run paramters,
model parameters and other info and return a parset, a model, and an
initial_params vector
"""

def parse_args_old(argv, rp={'param_file':'', 'sps':''}):
    """ Parse command line arguments given to prospectr.py
    """
    shortopt = ''
    try:
        opts, args = getopt.getopt(argv[1:],shortopt,[k+'=' for k in rp.keys()])
    except getopt.GetoptError:
        print 'bsfh.py --param_file <filename> --sps <sps_type>'
        sys.exit(2)
    for o, a in opts:
        try:
            rp[o[2:]] = float(a)
        except:
            rp[o[2:]] = a
    return rp

def parse_args(argv, argdict={'param_file':None, 'sps':'sps_basis',
                              'custom_filter_keys':None,
                              'compute_vega_mags':False}):
    """ Parse command line arguments, allowing for optional arguments.
    Simple/Fragile.
    """
    args = [sub for arg in argv[1:] for sub in arg.split('=')]
    for i, a in enumerate(args):
        if (a[:2] == '--'):
            abare = a[2:]
            if abare == 'help':
                show_syntax(argv, argdict)
                sys.exit()
        else:
            continue
        if abare in argdict.keys():
            apo = deepcopy(args[i+1])
            func = type(argdict[abare])
            try:
                argdict[abare] = func(apo)
            except TypeError:
                argdict[abare] = apo
    return argdict

def show_syntax(args, ad):
    print('Usage:\n {0} '.format(args[0]) +
          ' '.join(['--{0}=<value>'.format(k) for k in ad.keys()]))


def setup_model(filename, sps=None):
    """Use a .json file or a .py script to intialize a model and obs
    dictionary.

    :param filename:
        (Absolute) path to the .json or .py file

    :param sps: (optional)
        SPS object, required if data is being mocked.

    :returns model:
        A fully initialized model object.

    :returns parset:
        A ProspectrParams object

    
    """
    ext = filename.split('.')[-1]
    if ext == 'py':
        print('reading py script {}'.format(filename))
        setup_module = load_module_from_file(filename)
        rp = deepcopy(setup_module.run_params)
        mp = deepcopy(setup_module.model_params)
        obs = deepcopy(getattr(setup_module, 'obs', None))
        model_type = getattr(setup_module, 'model_type', sedmodel.SedModel)
        #print(np.median(obs['spectrum'][obs['mask']]))
    elif ext == 'json':
        print('reading json {}'.format(filename))
        rp, mp = modeldef.read_plist(filename)
        obs = None
        model_type = getattr(sedmodel,
                             rp.get('model_type','SedModel'))

    
    parset = modeldef.ProspectrParams(rp, mp)
    model = parset.initialize_model(model_type)

    if (obs is None) or (parset.run_params.get('mock', False)):
        obs = load_obs(model, sps=sps, **parset.run_params)
    parset.add_obs_to_model(model, obs)
    return parset, model

def load_obs(model, sps=None,
             mock=False, mock_info=None,
             data_loading_function_name=None,
             **kwargs):
    """Load or mock observations, and return an obs dictionary
    """
    if mock:
        print('loading mock')
        obs = dutils.generate_mock(model, sps, mock_info)
        initial_center = model.theta_from_params()
        initial_center *= np.random.beta(2,2,size=model.ndim)*2.0
        obs['mock_info'] = mock_info
    elif data_loading_function_name is not None:
        obsfunction = getattr(dutils, data_loading_function_name)
        obs = obsfunction(**kwargs)
    if 'maggies' in obs:
        obs['phot_mask'] = (obs.get('phot_mask',
                                    np.ones(len(obs['maggies']), dtype= bool)) *
                            np.isfinite(obs['maggies']))
    return obs

def load_module_from_file(path_to_file):
    """This has to break everything ever, right?
    """
    from importlib import import_module
    path, filename = os.path.split(path_to_file)
    modname = filename.replace('.py','')
    sys.path.insert(0,path)
    user_module = import_module(modname)
    sys.path.remove(path)
    return user_module

class Bunch(object):
    """ Simple storage.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def custom_filter_dict(filename):
    filter_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            ind, name = line.split()
            filter_dict[name.lower()] = Bunch(index = int(ind)-1)
            
    return filter_dict
