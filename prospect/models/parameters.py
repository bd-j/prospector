from copy import deepcopy
import numpy as np
import json
from . import priors
from ..utils.obsutils import logify_data, norm_spectrum

__all__ = ["ProspectorParams", "ProspectorParamsHMC"]

param_template = {'name': '', 'N': 1, 'isfree': False,
                  'init': 0.0, 'units': '',
                  'prior_function_name': None, 'prior_args': None}


class ProspectorParams(object):
    """
    :param config_list:
        A list of ``model parameters``.
    """
    # information about each parameter stored as a list
    # config_list = []
    # information about each parameter as a dictionary keyed by
    # parameter name for easy access
    # _config_dict = {}
    # Model parameter state, keyed by parameter name
    # params = {}
    # Mapping from parameter name to index in the theta vector
    # theta_index = {}
    # the initial theta vector
    # theta_init = np.array([])

    # Information about the fiting parameters, filenames, fitting and
    # configure options, runtime variables, etc.
    # run_params = {}
    # obs = {}

    def __init__(self, config_list, verbose=True):
        self.init_config_list = config_list
        self.config_list = config_list
        self.configure()
        self.verbose = verbose

    def configure(self, reset=False, **kwargs):
        """Use the parameter config_list to generate a theta_index mapping, and
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
        self._config_dict = plist_to_pdict(self.config_list)
        self.map_theta()
        # propogate initial parameter values from the configure dictionary
        for par, info in list(self._config_dict.items()):
            self.params[par] = np.atleast_1d(info['init'])
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
            self.theta_index[par] = slice(count, count+self._config_dict[par]['N'])
            count += self._config_dict[par]['N']
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

    def prior_product(self, theta):
        """Public version of _prior_product to be overridden by subclasses
        """
        return self._prior_product(theta)

    def _prior_product(self, theta):
        """Return a scalar which is the ln of the product of the prior
        probabilities for each element of theta.  Requires that the prior
        functions are defined in the theta descriptor.

        :param theta:
            Iterable containing the free model parameter values.

        :returns lnp_prior:
            The log of the product of the prior probabilities for these
            parameter values.
        """
        lnp_prior = 0
        for k, inds in list(self.theta_index.items()):
            this_prior = np.sum(self._config_dict[k]['prior_function']
                                (theta[inds], **self._config_dict[k]['prior_args']))

            if (not np.isfinite(this_prior)):
                print('WARNING: ' + k + ' is out of bounds')
            lnp_prior += this_prior
        return lnp_prior

    def propagate_parameter_dependencies(self):
        """Propogate any parameter dependecies. That is, for parameters whose
        value depends on another parameter, calculate those values and store
        them in the ``params`` dictionary.
        """
        if self._has_parameter_dependencies is False:
            return
        for p, info in list(self._config_dict.items()):
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
                for i in xrange(nt):
                    label.append(name+'_{0}'.format(i+1))
                    index.append(inds.start+i)
        return [l for (i, l) in sorted(zip(index, label))]

    def info(self, par):
        pass

    def reconfigure(self, par, config):
        pass

    def write_json(self, filename):
        pass

    def theta_bounds(self):
        """Get the bounds on each parameter from the prior.

        :returns bounds:
            A list of length self.ndim of tuples (lo, hi) giving the parameter
            bounds.
        """
        bounds = self.ndim * [(0., 0.)]
        for p, inds in list(self.theta_index.items()):
            sz = inds.stop - inds.start
            pb = priors.plotting_range(self._config_dict[p]['prior_args'])
            if sz == 1:
                bounds[inds] = pb
            else:
                for k in range(sz):
                    try:
                        bounds[inds.start + k] = (pb[0][k], pb[1][k])
                    except(TypeError, IndexError):
                        bounds[inds.start + k] = (pb[0], pb[1])
        # Force types
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
            d = self._config_dict[par].get('init_disp', default_disp)
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
            d = self._config_dict[par].get('disp_floor', 0.0)
            dfloor[inds] = d
        return dfloor

    def clip_to_bounds(self, thetas):
        """Clip a set of parameters theta to within the priors.

        :returns thetas:
            Clipped to theta priors.
        """
        bounds = self.theta_bounds()
        for i in xrange(len(bounds)):
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
            lnp_prior_grad[inds] = (self._config_dict[k]['prior_gradient_function']
                                          (theta[inds], **self._config_dict[k]['prior_args']))
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
                par = self._config_dict[k]
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
    """Replace names of functions in a parameter description with the actual
    functions.
    """
    from importlib import import_module
    for k, v in list(p.items()):
        try:
            m = import_module(v[1])
            f = m.__dict__[v[0]]
        except:
            pass
        else:
            p[k] = f
    return p


def functions_to_names(p):
    """Replace prior and dust functions with the names of those functions.
    """
    for k, v in list(p.items()):
        if callable(v):
            p[k] = [v.__name__, v.__module__]
    return p


def write_plist(plist, runpars, filename=None):
    """Write the list of parameter dictionaries to a JSON file, taking care to
    replace functions with their names.
    """
    for p in plist:
        p = functions_to_names(p)

    if filename is not None:
        runpars['param_file'] = filename
        f = open(filename + '.bpars.json', 'w')
        json.dump([runpars, plist], f)
        f.close()
    else:
        return json.dumps([runpars, plist])


def read_plist(filename, raw_json=False):
    """Read a JSON file into a run_param dictionary and a list of model
    parameter dictionaries, taking care to add actual functions when given
    their names.
    """
    with open(filename, 'r') as f:
        runpars, modelpars = json.load(f)
    runpars['param_file'] = filename
    if raw_json:
        return runpars, modelpars
    for p in modelpars:
        p = names_to_functions(p)

    return runpars, modelpars
