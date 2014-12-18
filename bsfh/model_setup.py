import sys, os, getopt, json
from copy import deepcopy
import numpy as np
from bsfh import parameters, sedmodel
from bsfh import datautils as dutils

"""This module has methods to take a .json or .py file containing run paramters,
model parameters and other info and return a parset, a model, and an
initial_params vector
"""

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
    """Show command line syntax corresponding to the provided arg
    dictionary `ad`.
    """
    print('Usage:\n {0} '.format(args[0]) +
          ' '.join(['--{0}=<value>'.format(k) for k in ad.keys()]))


def setup_gp(use_george=False):
    if use_george:
        import george
        kernel = (george.kernels.WhiteKernel(0.0) +
                  0.0 * george.kernels.ExpSquaredKernel(0.0))
        gp = george.GP(kernel, solver=george.HODLRSolver)
    else:
        from bsfh.gp import GaussianProcess
        gp = GaussianProcess(kernel=np.array([0.0, 0.0, 0.0]))
    return gp
    
def setup_model(filename, sps=None):
    """Use a .json file or a .py script to intialize a model and obs
    dictionary.

    :param filename:
        (Absolute) path to the .json or .py file

    :param sps: (optional)
        SPS object, required if data is being mocked.

    :returns model:
        A fully initialized model object.
    """
    ext = filename.split('.')[-1]
    # Read from files
    if ext == 'py':
        print('reading py script {}'.format(filename))
        setup_module = load_module_from_file(filename)
        rp = deepcopy(setup_module.run_params)
        mp = deepcopy(setup_module.model_params)
        obs = deepcopy(getattr(setup_module, 'obs', None))
        mock_info = deepcopy(getattr(setup_module, 'mock_info', None))
        model_type = getattr(setup_module, 'model_type', sedmodel.SedModel)
        
    elif ext == 'json':
        print('reading json {}'.format(filename))
        rp, mp = parameters.read_plist(filename)
        obs = None
        mock_info = rp.get('mock_info', None)
        model_type = getattr(sedmodel, rp.get('model_type','SedModel'))
        
    # Instantiate a model and add observational info, depending on whether
    # obs needs to be mocked or read using a named function, or simply added
    model = model_type(rp, mp)
    if model.run_params.get('mock', False):
        print('loading mock')
        obs = dutils.generate_mock(model, sps, mock_info)
        model.add_obs(obs)
        model.mock_info = mock_info
        model.initial_theta *= np.random.beta(2,2,size=model.ndim)*2.0
    elif obs is None:
        funcname = model.run_params['data_loading_function_name']
        obsfunction = getattr(dutils, funcname)
        obs = obsfunction(**model.run_params)
        model.add_obs(obs)
    else:
        model.add_obs(obs)
    
    return model

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
