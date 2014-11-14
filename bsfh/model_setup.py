import sys, os, json
from copy import deepcopy
import numpy as np
from bsfh import modeldef, sedmodel
from bsfh import datautils as dutils

"""This module has methods to take a .json or .py file containing run paramters,
model parameters and other info and return a parset, a model, and an
initial_params vector
"""


def parse_args(argv, rp={'param_file':''}):
    """ Parse command line arguments given to prospectr.py
    """
    shortopt = ''
    try:
        opts, args = getopt.getopt(argv[1:],shortopt,[k+'=' for k in rp.keys()])
    except getopt.GetoptError:
        print 'bsfh.py --param_file <filename>'
        sys.exit(2)
    for o, a in opts:
        try:
            rp[o[2:]] = float(a)
        except:
            rp[o[2:]] = a
    return rp

def setup_model(filename, sps=None):
    """Use a .json file or a .py script to intialize a model and obs
    dictionary.

    :param filename:
        (Absolute) path tot he .json or .py file

    :param sps: (optional)
        SPS object, required if data is being mocked.

    :returns model:
        A fully initialized model object.

    :returns parset:
        A ProspectrParams object

    
    """
    ext = filename.split('.')[-1]
    if ext == 'py':
        print('reading py script')
        setup_module = load_module_from_file(filename)
        rp = setup_module.run_params
        mp = setup_module.model_params
        obs = getattr(setup_module, 'obs', None)
        model_type = getattr(setup_module, 'model_type', sedmodel.SedModel)
        
    elif ext == 'json':
        print('reading json')
        rp, mp = modeldef.read_plist(filename)
        obs = None
        model_type = getattr(sedmodel,
                             rp.get('model_type','SedModel'))

    
    parset = modeldef.ProspectrParams(rp, mp)
    model = parset.initialize_model(model_type)

    if obs is None:
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
        obs = dutils.generate_mock(model, sps, mock_info)
        initial_center = model.theta_from_params()
        initial_center *= np.random.beta(2,2,size=model.ndim)*2.0

    if loading_function_name is not None:
        obsfunction = getattr(dutils, loading_function_name)
        obs = obsfunction(**kwargs)
    return obs

def load_module_from_file(path_to_file):
    """This has to break everything ever, right?
    """
    from importlib import import_module
    path, filename = os.path.split(path_to_file)
    modname = filename.replace('.py','')
    print(modname, path)
    sys.path.append(path)
    print(sys.path)
    user_module = import_module(modname)
    sys.path.remove(path)
    return user_module
