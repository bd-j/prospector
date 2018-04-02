from copy import deepcopy
import numpy as np
import json, pickle
from . import priors
from ..utils.obsutils import logify_data, norm_spectrum

__all__ = ["ProspectorParams", "ProspectorParamsHMC"]

param_template = {'name': '', 'N': 1, 'isfree': False,
                  'init': 0.0, 'units': '',
                  'prior_function': None, 'prior_args': None}


class ProspectorParams(object):
    """
    :param configuration:
        A list or dictionary of ``model parameters``.
    """
    # -- Information about each parameter stored as a list:
    # config_list = []
    # -- Information about each parameter as a dictionary keyed by parameter
    # -- name for easy access
    # config_dict = {}
    # -- Model parameter state, keyed by parameter name
    # params = {}
    # -- Mapping from parameter name to index in the theta vector
    # theta_index = {}
    # -- Initial theta vector
    # theta_init = np.array([])

    def __init__(self, configuration, verbose=True):
        self.init_config = deepcopy(configuration)
        if type(configuration) == list:
            self.config_list = config_list
            self.config_dict = plist_to_pdict(self.config_list)
        elif type(configuration) == dict:
            self.config_dict = configuration
            self.config_list = pdict_to_plist(self.config_dict)
        else:
            raise(TypeError, "Configuration variable not of valid type: {}".format(type(configuration)))
        self.configure()
        self.verbose = verbose

    def configure(self, reset=False, **kwargs):
        """Use the parameter config_dict to generate a theta_index mapping, and
        propogate the initial parameters into the params state dictionary, and
        store the intital theta vector implied by the config dictionary.

        :param kwargs:
            Keyword parameters can be used to override or add to the initial
            parameter values specified in config_list

        :param reset: (default: False)
            If true, empty the params dictionary before rereading the
            config_list.
        """
        self._has_parameter_dependencies = False
        if (not hasattr(self, 'params')) or reset:
            self.params = {}

        self.map_theta()
        # Propogate initial parameter values from the configure dictionary
        # Populate the 'prior' key of the configure dictionary
        # Check for 'depends_on'
        for par, info in list(self.config_dict.items()):
            self.params[par] = np.atleast_1d(info['init'])
            try:
                # this is for backwards compatibility
                self.config_dict[par]['prior'] = info['prior_function']
            except(KeyError):
                pass
            if info.get('depends_on', None) is not None:
                self._has_parameter_dependencies = True
        # propogate user supplied values, overriding the configure
        for k, v in list(kwargs.items()):
            self.params[k] = np.atleast_1d(v)
        # store these initial values
        self.initial_theta = self.theta.copy()  # self.rectify_theta((self.theta.copy()))

    def map_theta(self):
        """Construct the mapping from parameter name to the index in the theta
        vector corresponding to the first element of that parameter.
        """
        self.theta_index = {}
        count = 0
        for par in self.free_params:
            self.theta_index[par] = slice(count, count+self.config_dict[par]['N'])
            count += self.config_dict[par]['N']
        self.ndim = count

    def set_parameters(self, theta):
        """Propagate theta into the model parameters dictionary.

        :param theta:
            A theta parameter vector containing the desired parameters.
            ndarray of shape (ndim,)
        """
        assert len(theta) == self.ndim
        for k, inds in list(self.theta_index.items()):
            self.params[k] = np.atleast_1d(theta[inds])
        self.propagate_parameter_dependencies()

    def prior_product(self, theta, nested=False, **extras):
        """Public version of _prior_product to be overridden by subclasses.

        :param nested:
            If using nested sampling, this will only return 0 (or -inf).  This
            behavior can be overridden if you want to include complicated
            priors that are not included in the unit prior cube based proposals
            (e.g. something that is difficult to transform from the unit cube.)
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
            Iterable containing the free model parameter values.  Of shape
            (..., ndim)

        :returns lnp_prior:
            The log of the product of the prior probabilities for these
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
        them in the ``params`` dictionary.
        """
        if self._has_parameter_dependencies is False:
            return
        for p, info in list(self.config_dict.items()):
            if 'depends_on' in info:
                value = info['depends_on'](**self.params)
                self.params[p] = np.atleast_1d(value)

    def rectify_theta(self, theta):
        tiny_number = 1e-10
        zero = (theta == 0)
        theta[zero] = tiny_number
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
        """A list of the free model parameters.
        """
        return [k['name'] for k in pdict_to_plist(self.config_list)
                if k['isfree']]

    @property
    def fixed_params(self):
        """A list of the fixed model parameters that are specified in the
        ``model_params``.
        """
        return [k['name'] for k in pdict_to_plist(self.config_list)
                if (k['isfree'] is False)]

    def theta_labels(self, name_map={'amplitudes': 'A',
                                     'emission_luminosity': 'eline'}):
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

    #def write_json(self, filename):
    #    pass

    def theta_bounds(self):
        """Get the bounds on each parameter from the prior.

        :returns bounds:
            A list of length self.ndim of tuples (lo, hi) giving the parameter
            bounds.
        """
        bounds = np.zeros([self.ndim, 2])
        for p, inds in list(self.theta_index.items()):
            kwargs = self.config_dict[p].get('prior_args', {})
            try:
                pb = self.config_dict[p]['prior'].bounds(**kwargs)
            except(AttributeError):
                # old style
                pb = priors.plotting_range(self.config_dict[p]['prior_args'])
            bounds[inds, :] = np.array(pb).T
        # Force types ?
        bounds = [(np.atleast_1d(a)[0], np.atleast_1d(b)[0]) for a, b in bounds]
        return bounds

    def theta_disps(self, default_disp=0.1, fractional_disp=False):
        """Get a vector of absolute dispersions for each parameter to use in
        generating sampler balls for emcee's Ensemble sampler.  This can be
        overridden by subclasses if fractional dispersions are desired.

        :param initial_disp: (default: 0.1)
            The default dispersion to use in case the `init_disp` key is not
            provided in the parameter configuration.

        :param fractional_disp: (default: False)
            Treat the dispersion values as fractional dispersions
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
        the walker-calculated dispersions. This can be overridden by subclasses
        """
        dfloor = np.zeros(self.ndim)
        for par, inds in list(self.theta_index.items()):
            d = self.config_dict[par].get('disp_floor', 0.0)
            dfloor[inds] = d
        return dfloor

    def clip_to_bounds(self, thetas):
        """Clip a set of parameters theta to within the priors.

        :returns thetas:
            Clipped to theta priors.
        """
        bounds = self.theta_bounds()
        for i in range(len(bounds)):
            lower, upper = bounds[i]
            thetas[i] = np.clip(thetas[i], lower, upper)

        return thetas


class ProspectorParamsHMC(ProspectorParams):
    """Object describing a model parameter set, and conversions between a
    parameter dictionary and a theta vector (for use in MCMC sampling).  Also
    contains a method for computing the prior probability of a given theta
    vector.
    """

    def lnp_prior_grad(self, theta):
        """Return a vector of gradients in the prior probability.  Requires
        that functions giving the gradients are given in the theta descriptor.

        :param theta:
            A theta parameter vector containing the desired
            parameters.  ndarray of shape (ndim,)
        """
        lnp_prior_grad = np.zeros_like(theta)
        for k, inds in list(self.theta_index.items()):
            grad = self.config_dict[k]['prior'].gradient
            kwargs = self.config_dict[k].get('prior_args', {})
            lnp_prior_grad[inds] = grad(theta[inds], **kwargs)
        return lnp_prior_grad

    def check_constrained(self, theta):
        """For HMC, check if the trajectory has hit a wall in any parameter.
        If so, reflect the momentum and update the parameter position in the
        opposite direction until the parameter is within the bounds. Bounds
        are specified via the 'upper' and 'lower' keys of the theta descriptor.

        :param theta:
            A theta parameter vector containing the desired parameters.
            ndarray of shape (ndim,)
        """
        oob = True
        sign = np.ones_like(theta)
        if self.verbose:
            print('theta in={0}'.format(theta))
        while oob:
            oob = False
            for k, inds in list(self.theta_index.items()):
                par = self.config_dict[k]
                if 'upper' in par.keys():
                    above = theta[inds] > par['upper']
                    oob = oob or np.any(above)
                    theta[inds][above] = 2 * par['upper'] - theta[inds][above]
                    sign[inds][above] *= -1
                if 'lower' in par.keys():
                    below = theta[inds] < par['lower']
                    oob = oob or np.any(below)
                    theta[inds][below] = 2 * par['lower'] - theta[inds][below]
                    sign[inds][below] *= -1
        if self.verbose:
            print('theta out={0}'.format(theta))
        return theta, sign, oob


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


def pdict_to_plist(pdict):
    """Convert from a parameter dictionary to a parameter list of dictionaries,
    adding each key to each value dictionary as the `name' keyword.
    """
    if type(pdict) is list:
        return pdict[:]
    plist = []
    for k, v in list(pdict.items()):
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
