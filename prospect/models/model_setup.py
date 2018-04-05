import sys, os, getopt, json
from copy import deepcopy
import numpy as np
from . import parameters, sedmodel
from ..utils.obsutils import fix_obs

"""This module has methods to take a .json or .py file containing run
parameters, model parameters and other info and return a run_params dictionary,
an obs dictionary, and a model.  It also has methods to parse command line
options and return an sps object and a gp object.

Most of the load_<x> methods are just really shallow wrappers on
```import_module_from_file(param_file).load_<x>(**kwargs)``` and could probably
be done away with at this point, as they add a mostly useless layer of
abstraction.  Kept here for future flexibility.
"""

__all__ = ["parse_args", "import_module_from_file", "get_run_params",
           "load_model", "load_obs", "load_sps", "load_gp", "show_syntax"]


def parse_args(argv, argdict={}):
    """Parse command line arguments, allowing for optional arguments.
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
                if func is bool:
                    argdict[abare] = apo in ['True', 'true', 'T', 't', 'yes']
            except TypeError:
                argdict[abare] = apo
    return argdict


def get_run_params(param_file=None, argv=None, **kwargs):
    """Get a run_params dictionary from the param_file (if passed) otherwise
    return the kwargs dictionary.

    The order of precedence of parameter specification locations is:
        * 1. param_file (lowest)
        * 2. kwargs passsed to this function
        * 3. command line arguments
    """
    if param_file is None:
        rp = {}
    ext = param_file.split('.')[-1]
    if ext == 'py':
        setup_module = import_module_from_file(param_file)
        rp = deepcopy(setup_module.run_params)
    elif ext == 'json':
        rp, mp = parameters.read_plist(param_file)
    if kwargs is not None:
        kwargs.update(rp)
        rp = kwargs
    if argv is not None:
        rp = parse_args(argv, argdict=rp)
    rp['param_file'] = param_file

    return rp


def load_sps(param_file=None, **kwargs):
    """Return an ``sps`` object which is used to hold spectral libraries,
    perform interpolations, convolutions, etc.
    """
    ext = param_file.split('.')[-1]
    if ext == 'py':
        setup_module = import_module_from_file(param_file)
        sps = setup_module.load_sps(**kwargs)
    else:
        sps = None
    return sps


def load_gp(param_file=None, **kwargs):
    """Return two Gaussian Processes objects, either using BSFH's internal GP
    objects or George.

    :returns gp_spec:
        The gaussian process object to use for the spectroscopy.

    :returns gp_phot:
        The gaussian process object to use for the photometry.
    """
    ext = param_file.split('.')[-1]
    if ext == 'py':
        setup_module = import_module_from_file(param_file)
        gp_spec, gp_phot = setup_module.load_gp(**kwargs)
        try:
            gp_spec, gp_phot = setup_module.load_gp(**kwargs)
        except:
            gp_spec, gp_phot = None, None
    else:
        gp_spec, gp_phot = None, None
    return gp_spec, gp_phot


def load_model(param_file=None, **extras):
    """Load the model object from a model config list given in the config file.

    :returns model:
        An instance of the parameters.ProspectorParams object which has
        been configured
    """
    ext = param_file.split('.')[-1]
    if ext == 'py':
        setup_module = import_module_from_file(param_file)
        #mp = deepcopy(setup_module.model_params)
        model = setup_module.load_model(**extras)
    elif ext == 'json':
        rp, mp = parameters.read_plist(param_file)
        model_type = getattr(sedmodel, rp.get('model_type', 'SedModel'))
        model = model_type(mp)
    return model


def load_obs(param_file=None, data_loading_function_name=None, **kwargs):
    """Load the obs dictionary using information in ``param_file``.  kwargs are
    passed to ``fix_obs()`` and, if using a json configuration file, to the
    data_loading_function.

    :returns obs:
        A dictionary of observational data.
    """
    ext = param_file.split('.')[-1]
    obs = None
    if ext == 'py':
        print('reading py script {}'.format(param_file))
        setup_module = import_module_from_file(param_file)
        #obs = setup_module.load_obs(**kwargs)
        try:
            obs = setup_module.load_obs(**kwargs)
        except(AttributeError):
            obs = deepcopy(getattr(setup_module, 'obs', None))
    if obs is None:
        funcname = data_loading_function_name
        obsfunction = getattr(setup_module, funcname)
        obs = obsfunction(**kwargs)

    obs = fix_obs(obs, **kwargs)
    return obs


def import_module_from_file(path_to_file):
    """This has to break everything ever, right?
    """
    from importlib import import_module
    path, filename = os.path.split(path_to_file)
    modname = filename.replace('.py', '')
    sys.path.insert(0, path)
    user_module = import_module(modname)
    sys.path.remove(path)
    return user_module


def import_module_from_string(source, name, add_to_sys_modules=True):
    """Well this seems dangerous.
    """
    import imp
    user_module = imp.new_module(name)
    exec(source, user_module.__dict__)
    if add_to_sys_modules:
        sys.modules[name] = user_module

    return user_module


def show_syntax(args, ad):
    """Show command line syntax corresponding to the provided arg dictionary
    `ad`.
    """
    print('Usage:\n {0} '.format(args[0]) +
          ' '.join(['--{0}=<value>'.format(k) for k in ad.keys()]))


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
            filter_dict[name.lower()] = Bunch(index=int(ind)-1)

    return filter_dict
