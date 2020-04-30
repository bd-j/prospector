#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, getopt, json, warnings
from copy import deepcopy
import numpy as np
from . import parameters
from ..utils.obsutils import fix_obs

"""This module has methods to take a .py file containing run parameters, model
parameters and other info and return a run_params dictionary, an obs
dictionary, and a model.  It also has methods to parse command line options and
return an sps object and noise objects.

Most of the load_<x> methods are just really shallow wrappers on
```import_module_from_file(param_file).load_<x>(**kwargs)``` and could probably
be done away with at this point, as they add a mostly useless layer of
abstraction.  Kept here for future flexibility.
"""

__all__ = ["parse_args", "import_module_from_file", "get_run_params",
           "load_model", "load_obs", "load_sps", "load_gp", "show_syntax"]


deprecation_msg = ("Use argparse based operation; usage via prospector_*.py "
                   "scripts will be disabled in the future.")


def parse_args(argv, argdict={}):
    """Parse command line arguments, allowing for optional arguments.
    Simple/Fragile.
    """
    warnings.warn(deprecation_msg, FutureWarning)
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
    warnings.warn(deprecation_msg, FutureWarning)
    rp = {}
    if param_file is None:
        ext = ""
    else:
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
    warnings.warn(deprecation_msg, FutureWarning)
    ext = param_file.split('.')[-1]
    assert ext == 'py'
    setup_module = import_module_from_file(param_file)

    if hasattr(setup_module, 'load_sps'):
        builder = setup_module.load_sps
    elif hasattr(setup_module, 'build_sps'):
        builder = setup_module.build_sps
    else:
        warnings.warn("Could not find load_sps or build_sps methods in param_file")
        return None

    sps = builder(**kwargs)

    return sps


def load_gp(param_file=None, **kwargs):
    """Return two Gaussian Processes objects, either using BSFH's internal GP
    objects or George.

    :returns gp_spec:
        The gaussian process object to use for the spectroscopy.

    :returns gp_phot:
        The gaussian process object to use for the photometry.
    """
    warnings.warn(deprecation_msg, FutureWarning)
    ext = param_file.split('.')[-1]
    assert ext == "py"
    setup_module = import_module_from_file(param_file)

    if hasattr(setup_module, 'load_gp'):
        builder = setup_module.load_gp
    elif hasattr(setup_module, 'build_noise'):
        builder = setup_module.build_noise
    else:
        warnings.warn("Could not find load_gp or build_noise methods in param_file")
        return None, None

    spec_noise, phot_noise = builder(**kwargs)

    return spec_noise, phot_noise


def load_model(param_file=None, **kwargs):
    """Load the model object from a model config list given in the config file.

    :returns model:
        An instance of the parameters.ProspectorParams object which has
        been configured
    """
    warnings.warn(deprecation_msg, FutureWarning)
    ext = param_file.split('.')[-1]
    assert ext == 'py'
    setup_module = import_module_from_file(param_file)
        #mp = deepcopy(setup_module.model_params)

    if hasattr(setup_module, 'load_model'):
        builder = setup_module.load_model
    elif hasattr(setup_module, 'build_model'):
        builder = setup_module.build_model
    else:
        warnings.warn("Could not find load_model or build_model methods in param_file")
        return None

    model = builder(**kwargs)

    return model


def load_obs(param_file=None, **kwargs):
    """Load the obs dictionary using the `obs` attribute or methods in
    ``param_file``.  kwargs are passed to these methods and ``fix_obs()``

    :returns obs:
        A dictionary of observational data.
    """
    warnings.warn(deprecation_msg, FutureWarning)
    ext = param_file.split('.')[-1]
    obs = None
    assert ext == 'py'
    print('reading py script {}'.format(param_file))
    setup_module = import_module_from_file(param_file)

    if hasattr(setup_module, 'obs'):
        return fix_obs(deepcopy(setup_module.obs))
    if hasattr(setup_module, 'load_obs'):
        builder = setup_module.load_obs
    elif hasattr(setup_module, 'build_obs'):
        builder = setup_module.build_obs
    else:
        warnings.warn("Could not find load_obs or build_obs methods in param_file")
        return None

    obs = builder(**kwargs)
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
