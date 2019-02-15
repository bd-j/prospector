from copy import deepcopy
import numpy as np
import json, pickle
from . import priors
from .templates import describe

__all__ = ["ProspectorParams"] #, "plist_to_pdict", "pdict_to_plist"]


# A template for what parameter configuration list element should look like
param_template = {'name': '',
                  'N': 1,
                  'isfree': True,
                  'init': 0.5, 'units': '',
                  'prior': priors.TopHat(mini=0, maxi=1.0),
                  'depends_on': None}


class ProspectorParams(object):
    """
    This is the base model class that holds model parameters and information
    about them (e.g. priors, bounds, transforms, free vs fixed state).  In
    addition to the documented methods, it contains several important
    attributes:

    * :py:attr:`params`: model parameter state dictionary.
    * :py:attr:`theta_index`: A dictionary that maps parameter names to indices (or rather
      slices) of the parameter vector ``theta``.
    * :py:attr:`config_dict`: Information about each parameter as a dictionary keyed by
      parameter name for easy access.
    * :py:attr:`config_list`: Information about each parameter stored as a list.

    Intitialization is via, e.g.,

    .. code-block:: python

       model_dict = {"mass": {"N": 1, "isfree": False, "init": 1e10}}
       model = ProspectorParams(model_dict, param_order=None)

    :param configuration:
        A list or dictionary of model parameters specifications.
    """

    def __init__(self, configuration, verbose=True, param_order=None, **kwargs):
        """
        :param configuration:
            A list or dictionary of parameter specification dictionaries.

        :param param_order: (optional, default: None)
            If given and `configuration` is a dictionary, this will specify the
            order in which the parameters appear in the theta vector.  Iterable
            of strings.
        """
        self.init_config = deepcopy(configuration)
        self.parameter_order = param_order
        if type(configuration) == list:
            self.config_list = configuration
            self.config_dict = plist_to_pdict(self.config_list)
        elif type(configuration) == dict:
            self.config_dict = configuration
            self.config_list = pdict_to_plist(self.config_dict, order=param_order)
        else:
            raise(TypeError, ("Configuration variable not of valid type: "
                              "{}".format(type(configuration))))
        self.configure(**kwargs)
        self.verbose = verbose

    def __repr__(self):
        return ":::::::\n{}\n\n{}".format(self.__class__, self.description)
        
    def configure(self, reset=False, **kwargs):
        """Use the :py:attr:`config_dict` to generate a :py:attr:`theta_index`
        mapping, and propogate the initial parameters into the
        :py:attr:`params` state dictionary, and store the intital theta vector
        thus implied.

        :param kwargs:
            Keyword parameters can be used to override or add to the initial
            parameter values specified in :py:attr:`config_dict`

        :param reset: (default: False)
            If true, empty the params dictionary before re-reading the
            :py:attr:`config_dict`
        """
        self._has_parameter_dependencies = False
        if (not hasattr(self, 'params')) or reset:
            self.params = {}

        self.map_theta()
        # Propogate initial parameter values from the configure dictionary
        # Populate the 'prior' key of the configure dictionary
        # Check for 'depends_on'
        for par, info in list(self.config_dict.items()):
            self.params[par] = np.atleast_1d(info['init']).copy()
            try:
                # this is for backwards compatibility
                self.config_dict[par]['prior'] = info['prior_function']
            except(KeyError):
                pass
            if info.get('depends_on', None) is not None:
                assert callable(info["depends_on"])
                self._has_parameter_dependencies = True
        # propogate user supplied values to the params state, overriding the
        # configure `init` values
        for k, v in list(kwargs.items()):
            self.params[k] = np.atleast_1d(v)
        # store these initial values
        self.initial_theta = self.theta.copy()

    def map_theta(self):
        """Construct the mapping from parameter name to the index in the theta
        vector corresponding to the first element of that parameter.  Called
        during configuration.
        """
        self.theta_index = {}
        count = 0
        for par in self.free_params:
            self.theta_index[par] = slice(count, count+self.config_dict[par]['N'])
            count += self.config_dict[par]['N']
        self.ndim = count

    def set_parameters(self, theta):
        """Propagate theta into the model parameters :py:attr:`params` dictionary.

        :param theta:
            A theta parameter vector containing the desired parameters. ndarray
            of shape ``(ndim,)``
        """
        assert len(theta) == self.ndim
        for k, inds in list(self.theta_index.items()):
            self.params[k] = np.atleast_1d(theta[inds]).copy()
        self.propagate_parameter_dependencies()

    def prior_product(self, theta, nested=False, **extras):
        """Public version of _prior_product to be overridden by subclasses.

        :param theta:
            The parameter vector for which you want to calculate the
            prior. ndarray of shape ``(..., ndim)``

        :param nested:
            If using nested sampling, this will only return 0 (or -inf).  This
            behavior can be overridden if you want to include complicated
            priors that are not included in the unit prior cube based proposals
            (e.g. something that is difficult to transform from the unit cube.)

        :returns lnp_prior:
            The natural log of the prior probability at ``theta``
        """
        lpp = self._prior_product(theta)
        if nested & np.any(np.isfinite(lpp)):
            return 0.0
        return lpp

    def _prior_product(self, theta, **extras):
        """Return a scalar which is the ln of the product of the prior
        probabilities for each element of theta.  Requires that the prior
        functions are defined in the theta descriptor.

        :param theta:
            Iterable containing the free model parameter values. ndarray of
            shape ``(ndim,)``

        :returns lnp_prior:
            The natural log of the product of the prior probabilities for these
            parameter values.
        """
        lnp_prior = 0
        for k, inds in list(self.theta_index.items()):

            func = self.config_dict[k]['prior']
            kwargs = self.config_dict[k].get('prior_args', {})
            this_prior = np.sum(func(theta[..., inds], **kwargs), axis=-1)
            lnp_prior += this_prior

        return lnp_prior

    def prior_transform(self, unit_coords):
        """Go from unit cube to parameter space, for nested sampling.

        :param unit_coords:
            Coordinates in the unit hyper-cube. ndarray of shape ``(ndim,)``.
            
        :returns theta:
            The parameter vector corresponding to the location in prior CDF
            corresponding to ``unit_coords``. ndarray of shape ``(ndim,)``
        """
        theta = np.zeros(len(unit_coords))
        for k, inds in list(self.theta_index.items()):
            func = self.config_dict[k]['prior'].unit_transform
            kwargs = self.config_dict[k].get('prior_args', {})
            theta[inds] = func(unit_coords[inds], **kwargs)
        return theta

    def propagate_parameter_dependencies(self):
        """Propogate any parameter dependecies. That is, for parameters whose
        value depends on another parameter, calculate those values and store
        them in the :py:attr:`self.params` dictionary.
        """
        if self._has_parameter_dependencies is False:
            return
        for p, info in list(self.config_dict.items()):
            if 'depends_on' in info:
                value = info['depends_on'](**self.params)
                self.params[p] = np.atleast_1d(value)

    def rectify_theta(self, theta, epsilon=1e-10):
        """Replace zeros in a given theta vector with a small number epsilon.
        """
        zero = (theta == 0)
        theta[zero] = epsilon
        return theta

    @property
    def theta(self):
        """The current value of the theta vector, pulled from the ``params``
        state dictionary.
        """
        theta = np.zeros(self.ndim)
        for k, inds in list(self.theta_index.items()):
            theta[inds] = self.params[k]
        return theta

    @property
    def free_params(self):
        """A list of the names of the free model parameters.
        """
        return [k['name'] for k in pdict_to_plist(self.config_list)
                if k['isfree']]

    @property
    def fixed_params(self):
        """A list of the names fixed model parameters that are specified in the
        ``config_dict``.
        """
        return [k['name'] for k in pdict_to_plist(self.config_list)
                if (k['isfree'] is False)]

    @property
    def description(self):
        return describe(self.config_dict)
        
    def theta_labels(self, name_map={}):
        """Using the theta_index parameter map, return a list of the model
        parameter names that has the same order as the sampling chain array.

        :param name_map:
            A dictionary mapping model parameter names to output label
            names.

        :returns labels:
            A list of labels of the same length and order as the theta
            vector.
        """
        label, index = [], []
        for p, inds in list(self.theta_index.items()):
            nt = inds.stop - inds.start
            try:
                name = name_map[p]
            except(KeyError):
                name = p
            if nt is 1:
                label.append(name)
                index.append(inds.start)
            else:
                for i in range(nt):
                    label.append(name+'_{0}'.format(i+1))
                    index.append(inds.start+i)
        return [l for (i, l) in sorted(zip(index, label))]

    def theta_bounds(self):
        """Get the bounds on each parameter from the prior.

        :returns bounds:
            A list of length ``ndim`` of tuples ``(lo, hi)`` giving the
            parameter bounds.
        """
        bounds = np.zeros([self.ndim, 2])
        for p, inds in list(self.theta_index.items()):
            kwargs = self.config_dict[p].get('prior_args', {})
            try:
                pb = self.config_dict[p]['prior'].bounds(**kwargs)
            except(AttributeError):
                # old style, including for backwards compatibility
                pb = priors.plotting_range(self.config_dict[p]['prior_args'])
            bounds[inds, :] = np.array(pb).T
        # Force types ?
        bounds = [(np.atleast_1d(a)[0], np.atleast_1d(b)[0])
                  for a, b in bounds]
        return bounds

    def theta_disps(self, default_disp=0.1, fractional_disp=False):
        """Get a vector of absolute dispersions for each parameter to use in
        generating sampler balls for emcee's Ensemble sampler.  This can be
        overridden by subclasses if fractional dispersions are desired.

        :param initial_disp: (default: 0.1)
            The default dispersion to use in case the ``"init_disp"`` key is
            not provided in the parameter configuration.

        :param fractional_disp: (default: False)
            Treat the dispersion values as fractional dispersions.

        :returns disp:
            The dispersion in the parameters to use for generating clouds of
            walkers (or minimizers.) ndarray of shape ``(ndim,)``
        """
        disp = np.zeros(self.ndim) + default_disp
        for par, inds in list(self.theta_index.items()):
            d = self.config_dict[par].get('init_disp', default_disp)
            disp[inds] = d
        if fractional_disp:
            disp = self.theta * disp
        return disp

    def theta_disp_floor(self, thetas=None):
        """Get a vector of dispersions for each parameter to use as a floor for
        the emcee walker-calculated dispersions. This can be overridden by
        subclasses.

        :returns disp_floor:
            The minimum dispersion in the parameters to use for generating
            clouds of walkers (or minimizers.) ndarray of shape ``(ndim,)``
        """
        dfloor = np.zeros(self.ndim)
        for par, inds in list(self.theta_index.items()):
            d = self.config_dict[par].get('disp_floor', 0.0)
            dfloor[inds] = d
        return dfloor

    def clip_to_bounds(self, thetas):
        """Clip a set of parameters theta to within the priors.

        :param thetas:
            The parameter vector, ndarray of shape ``(ndim,)``.

        :returns thetas:
            The input vector, clipped to the bounds of the priors.
        """
        bounds = self.theta_bounds()
        for i in range(len(bounds)):
            lower, upper = bounds[i]
            thetas[i] = np.clip(thetas[i], lower, upper)

        return thetas

    @property
    def _config_dict(self):
        """Backwards compatibility
        """
        return self.config_dict


def plist_to_pdict(inplist):
    """Convert from a parameter list to a parameter dictionary, where the keys
    of the cdictionary are the parameter names.
    """
    plist = deepcopy(inplist)
    if type(plist) is dict:
        return plist.copy()
    pdict = {}
    for p in plist:
        name = p.pop('name')
        pdict[name] = p
    return pdict


def pdict_to_plist(pdict, order=None):
    """Convert from a dictionary of parameter dictionaries to a list of
    parameter dictionaries, adding each key to each value dictionary as the
    `name' keyword.  Optionally, do this in an order specified by `order`. This
    method is not used often, so it can be a bit inefficient

    :param pdict:
        A dictionary of parameter specification dictionaries, keyed by
        parameter name.  If a list is given instead of a dictionary, this same
        list is returned.

    :param order:
        An iterable of parameter names specifying the order in which they
        should be added to the parameter list

    :returns plist:
        A list of parameter specification dictinaries (with the `"name"` key
        added.)  The listed dictionaries are *not* copied from the input
        dictionary.
    """
    if type(pdict) is list:
        return pdict[:]
    plist = []
    if order is not None:
        assert len(order) == len(pdict)
    else:
        order = pdict.keys()
    for k in order:
        v = pdict[k]
        v['name'] = k
        plist += [v]
    return plist


def names_to_functions(p):
    """Replace names of functions (or pickles of objects) in a parameter
    description with the actual functions (or pickles).
    """
    from importlib import import_module
    for k, v in list(p.items()):
        try:
            m = import_module(v[1])
            f = m.__dict__[v[0]]
        except:
            try:
                f = pickle.loads(v)
            except:
                f = v

        p[k] = f

    return p


def functions_to_names(p):
    """Replace prior and dust functions (or objects) with the names of those
    functions (or pickles).
    """
    for k, v in list(p.items()):
        if callable(v):
            try:
                p[k] = [v.__name__, v.__module__]
            except(AttributeError):
                p[k] = pickle.dumps(v, protocol=2)
    return p
