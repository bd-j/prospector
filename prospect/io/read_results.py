import sys, os
import pickle, json
import numpy as np
try:
    import h5py
except:
    pass
try:
    from sedpy.observate import load_filters
except:
    pass

from ..models.parameters import names_to_functions

"""Convenience functions for reading and reconstructing results from a fitting
run, including reconstruction of the model for making posterior samples
"""

__all__ = ["results_from",
           "get_sps", "get_model",
           "traceplot", "subcorner"]


def unpick(pickled):
    """create a serialized object that can go into hdf5 in py2 and py3, and can be read by both
    """
    try:
        obj = pickle.loads(pickled, encoding='bytes')
    except(TypeError):
        obj = pickle.loads(pickled)

    return obj

    
def results_from(filename, model_file=None, dangerous=True, **kwargs):
    """Read a results file with stored model and MCMC chains.

    :param filename:
        Name and path to the file holding the results.  If ``filename`` ends in
        "h5" then it is assumed that this is an HDF5 file, otherwise it is
        assumed to be a pickle.

    :param dangerous: (default, False)
        If True, use the stored paramfile text to import the parameter file and
        reconstitute the model object.  This executes code in the stored paramfile
        text during import, and is therefore dangerous.

    :returns sample_results:
        A dictionary of various results including the sampling chain.

    :returns obs:
        The obs dictionary

    :returns model:
        The models.sedmodel() object.
    """
    # Read the basic chain, parameter, and run_params info
    if filename.split('.')[-1] == 'h5':
        res = read_hdf5(filename, **kwargs)
        mf_default = filename.replace('_mcmc.h5', '_model')
    else:
        with open(filename, 'rb') as rf:
            res = pickle.load(rf)
        mf_default = filename.replace('_mcmc', '_model')

    # Now try to read the model object itself from a pickle
    if model_file is None:
        mname = mf_default
    else:
        mname = model_file
    param_file = (res['run_params'].get('param_file', ''),
                  res.get("paramfile_text", ''))
    model, opt_results = read_model(mname, param_file=param_file, dangerous=dangerous,
                                    **kwargs)
    if dangerous:
        try:
            model = get_model(res)
        except:
            model = None
    res['model'] = model
    res["optimization_results"] = opt_results 

    return res, res["obs"], model


def read_model(model_file, param_file=('', ''), dangerous=False, **extras):
    """Read the model pickle.  This can be difficult if there are user defined
    functions that have to be loaded dynamically.  In that case, import the
    string version of the paramfile and *then* try to unpickle the model
    object.

    :param model_file:
        String, name and path to the model pickle.

    :param dangerous: (default: False)
        If True, try to import the given paramfile.

    :param param_file:
        2-element tuple.  The first element is the name of the paramfile, which
        will be used to set the name of the imported module.  The second
        element is the param_file contents as a string.  The code in this
        string will be imported.
    """
    model = powell_results = None
    if os.path.exists(model_file):
        try:
            with open(model_file, 'rb') as mf:
                mod = pickle.load(mf)
        except(AttributeError):
            # Here one can deal with module and class names that changed
            with open(model_file, 'rb') as mf:
                mod = load(mf)
        except(ImportError):
            # here we load the parameter file as a module using the stored
            # source string.  Obviously this is dangerous as it will execute
            # whatever is in the stored source string.  But it can be used to
            # recover functions (especially dependcy functions) that are user
            # defined
            path, filename = os.path.split(param_file[0])
            modname = filename.replace('.py', '')
            if dangerous:
                from ..models.model_setup import import_module_from_string
                user_module = import_module_from_string(param_file[1], modname)
            with open(model_file, 'rb') as mf:
                mod = pickle.load(mf)

        model = mod['model']

        for k, v in list(model.theta_index.items()):
            if type(v) is tuple:
                model.theta_index[k] = slice(*v)
        powell_results = mod['powell']

    return model, powell_results


def read_hdf5(filename, **extras):
    """Read an HDF5 file (with a specific format) into a dictionary of results.

    This HDF5 file is assumed to have the groups ``sampling`` and ``obs`` which
    respectively contain the sampling chain and the observational data used in
    the inference.

    All attributes of these groups as well as top-level attributes are loaded
    into the top-level of the dictionary using ``json.loads``, and therefore
    must have been written with ``json.dumps``.  This should probably use
    JSONDecoders, but who has time to learn that.

    :param filename:
        Name of the HDF5 file.
    """
    groups = {'sampling': {}, 'obs': {}}
    res = {}
    with h5py.File(filename, "r") as hf:
        # loop over the groups
        for group, d in groups.items():
            # read the arrays in that group into the dictionary for that group
            for k, v in hf[group].items():
                d[k] = np.array(v)
            # unserialize the attributes and put them in the dictionary
            for k, v in hf[group].attrs.items():
                try:
                    d[k] = json.loads(v)
                except:
                    try:
                        d[k] = unpick(v)
                    except:
                        d[k] = v
        # do top-level attributes.
        for k, v in hf.attrs.items():
            try:
                res[k] = json.loads(v)
            except:
                try:
                    res[k] = unpick(v)
                except:
                    res[k] = v
        res.update(groups['sampling'])
        res['obs'] = groups['obs']
        try:
            res['obs']['filters'] = load_filters([str(f) for f in res['obs']['filters']])
        except:
            pass
        try:
            res['rstate'] = unpick(res['rstate'])
        except:
            pass
        try:
            mp = [names_to_functions(p.copy()) for p in res['model_params']]
            res['model_params'] = mp
        except:
            pass

    return res


def read_pickles(filename, **kwargs):
    """Alias for backwards compatability. Calls results_from().
    """
    return results_from(filename, **kwargs)


def get_sps(res):
    """This gets exactly the SPS object used in the fiting (modulo any
    changes to FSPS itself).

    It (scarily) imports the paramfile (stored as text in the results
    dictionary) as a module and then uses the `load_sps` method defined in the
    paramfile module.

    :param res:
        A results dictionary (the output of `results_from()`)

    :returns sps:
        An sps object (i.e. from prospect.sources)
    """
    import os
    from ..models.model_setup import import_module_from_string
    param_file = (res['run_params'].get('param_file', ''),
                  res.get("paramfile_text", ''))
    path, filename = os.path.split(param_file[0])
    modname = filename.replace('.py', '')
    user_module = import_module_from_string(param_file[1], modname)
    sps = user_module.load_sps(**res['run_params'])

    # Now check that the SSP libraries are consistent
    flib = res['run_params'].get('sps_libraries', None)
    try:
        rlib = sps.ssp.libraries
    except(AttributeError):
        rlib = None
    if (flib is None) or (rlib is None):
        print("Could not check SSP library versions.")
    else:
        liberr = ("The FSPS libraries used in fitting({}) are not the "
                  "same as the FSPS libraries that you are using now ({})".format(flib, rlib))
        assert (flib[0] == rlib[0]) and (flib[1] == rlib[1]), liberr

    return sps


def get_model(res):
    """This gets exactly the model object used in the fiting.

    It (scarily) imports the paramfile (stored as text in the results
    dictionary) as a module and then uses the `load_model` method defined in the
    paramfile module, with `run_params` dictionary passed to it.

    :param res:
        A results dictionary (the output of `results_from()`)

    :returns model:
        A prospect.models.SedModel object
    """
    import os
    from ..models.model_setup import import_module_from_string
    param_file = (res['run_params'].get('param_file', ''),
                  res.get("paramfile_text", ''))
    path, filename = os.path.split(param_file[0])
    modname = filename.replace('.py', '')
    user_module = import_module_from_string(param_file[1], modname)
    model = user_module.load_model(**res['run_params'])
    return model


def traceplot(results, showpars=None, start=0, chains=slice(None),
              figsize=None, truths=None, **plot_kwargs):
    """Plot the evolution of each parameter value with iteration #, for each
    walker in the chain.

    :param results:
        A Prospector results dictionary, usually the output of
        ``results_from('resultfile')``.

    :param showpars: (optional)
        A list of strings of the parameters to show.  Defaults to all
        parameters in the ``"theta_labels"`` key of the ``sample_results``
        dictionary.

    :param chains:
        If results are from an ensemble sampler, setting `chain` to an integer
        array of walker indices will cause only those walkers to be used in
        generating the plot.  Useful for to keep the plot from getting too cluttered.

    :param start: (optional, default: 0)
        Integer giving the iteration number from which to start plotting.

    :param **plot_kwargs:
        Extra keywords are passed to the
        ``matplotlib.axes._subplots.AxesSubplot.plot()`` method.

    :returns tracefig:
        A multipaneled Figure object that shows the evolution of walker
        positions in the parameters given by ``showpars``, as well as
        ln(posterior probability)
    """
    import matplotlib.pyplot as pl


    # Get parameter names
    try:
        parnames = np.array(results['theta_labels'])
    except(KeyError):
        parnames = np.array(sample_results['model'].theta_labels())
    # Restrict to desired parameters
    if showpars is not None:
        ind_show = np.array([p in showpars for p in parnames], dtype=bool)
        parnames = parnames[ind_show]
    else:
        ind_show = slice(None)

    # Get the arrays we need (trace, lnp, wghts)
    trace = results['chain'][..., ind_show]
    if trace.ndim ==2:
        trace = trace[None, :]
    trace = trace[chains, start:, :]
    lnp = np.atleast_2d(results['lnprobability'])[chains, start:]
    wghts = results.get('weights', None)
    if wghts is not None:
        wghts = wghts[start:]
    nwalk = trace.shape[0]

    # Set up plot windows
    ndim = len(parnames) + 1
    nx = int(np.floor(np.sqrt(ndim)))
    ny = int(np.ceil(ndim * 1.0 / nx))
    sz = np.array([nx, ny])
    factor = 3.0           # size of one side of one panel
    lbdim = 0.2 * factor   # size of left/bottom margin
    trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.05 * factor         # w/hspace size
    plotdim = factor * sz + factor * (sz - 1) * whspace
    dim = lbdim + plotdim + trdim

    if figsize is None:
        fig, axes = pl.subplots(nx, ny, figsize=(dim[1], dim[0]), sharex=True)
    else:
        fig, axes = pl.subplots(nx, ny, figsize=figsize, sharex=True)
    axes = np.atleast_2d(axes)
    #lb = lbdim / dim
    #tr = (lbdim + plotdim) / dim
    #fig.subplots_adjust(left=lb[1], bottom=lb[0], right=tr[1], top=tr[0],
    #                    wspace=whspace, hspace=whspace)

    # Sequentially plot the chains in each parameter
    for i in range(ndim - 1):
        ax = axes.flat[i]
        for j in range(nwalk):
            ax.plot(trace[j, :, i], **plot_kwargs)
        ax.set_title(parnames[i], y=1.02)
    # Plot lnprob
    ax = axes.flat[-1]
    for j in range(nwalk):
        ax.plot(lnp[j, :], **plot_kwargs)
    ax.set_title('lnP', y=1.02)


    [ax.set_xlabel("iteration") for ax in axes[-1,:]]
    #[ax.set_xticklabels('') for ax in axes[:-1, :].flat]

    if truths is not None:
        for i, t in enumerate(truths[ind_show]):
            axes.flat[i].axhline(t, color='k', linestyle=':')

    pl.tight_layout()
    
    return fig


def param_evol(results, **kwargs):
    """Backwards compatability
    """
    return traceplot(results, **kwargs)


def subcorner(results, showpars=None, truths=None,
              start=0, thin=1, chains=slice(None),
              logify=["mass", "tau"], **kwargs):
    """Make a triangle plot of the (thinned, latter) samples of the posterior
    parameter space.  Optionally make the plot only for a supplied subset of
    the parameters.

    :param showpars: (optional)
        List of string names of parameters to include in the corner plot.

    :param truths: (optional)
        List of truth values for the chosen parameters.

    :param start: (optional, default: 0)
        The iteration number to start with when drawing samples to plot.

    :param thin: (optional, default: 1)
        The thinning of each chain to perform when drawing samples to plot.

    :param chains: (optional)
        If results are from an ensemble sampler, setting `chain` to an integer
        array of walker indices will cause only those walkers to be used in
        generating the plot.  Useful for emoving stuck walkers.

    :param kwargs:
        Remaining keywords are passed to the ``corner`` plotting package.

    :param logify:
        A list of parameter names to plot in `log10(parameter)` instead of
        `parameter`
    """
    try:
        import corner as triangle
    except(ImportError):
        import triangle

    # pull out the parameter names and flatten the thinned chains
    # Get parameter names
    try:
        parnames = np.array(results['theta_labels'], dtype='S20')
    except(KeyError):
        parnames = np.array(sample_results['model'].theta_labels())
    # Restrict to desired parameters
    if showpars is not None:
        ind_show = np.array([parnames.tolist().index(p) for p in showpars])
        parnames = parnames[ind_show]
    else:
        ind_show = slice(None)

    # Get the arrays we need (trace, wghts)
    trace = results['chain'][..., ind_show]
    if trace.ndim == 2:
        trace = trace[None, :]
    trace = trace[chains, start::thin, :]
    wghts = results.get('weights', None)
    if wghts is not None:
        wghts = wghts[start::thin]
    samples = trace.reshape(trace.shape[0] * trace.shape[1], trace.shape[2])

    # logify some parameters
    xx = samples.copy()
    if truths is not None:
        xx_truth = np.array(truths).copy()
    else:
        xx_truth = None
    for p in logify:
        if p in parnames:
            idx = parnames.tolist().index(p)
            xx[:, idx] = np.log10(xx[:,idx])
            parnames[idx] = "log({})".format(parnames[idx])
            if truths is not None:
                xx_truth[idx] = np.log10(xx_truth[idx])

    # mess with corner defaults
    corner_kwargs = {"plot_datapoints": False, "plot_density": False,
                     "fill_contours": True, "show_titles": True}
    corner_kwargs.update(kwargs)
    
    fig = triangle.corner(xx, labels=parnames, truths=xx_truth,
                          quantiles=[0.16, 0.5, 0.84], weights=wghts, **corner_kwargs)

    return fig


def subtriangle(results, **kwargs):
    """Backwards compatability
    """
    return subcorner(results, **kwargs)

# --- Deprecated code
# All this because scipy changed the name of one class, which
# shouldn't even be a class.

renametable = {
    'Result': 'OptimizeResult',
    }


def mapname(name):
    if name in renametable:
        return renametable[name]
    return name


def mapped_load_global(self):
    module = mapname(self.readline()[:-1])
    name = mapname(self.readline()[:-1])
    klass = self.find_class(module, name)
    self.append(klass)


def load(file):
    unpickler = pickle.Unpickler(file)
    unpickler.dispatch[pickle.GLOBAL] = mapped_load_global
    return unpickler.load()
