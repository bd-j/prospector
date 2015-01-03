from copy import deepcopy
import numpy as np
import json
from bsfh import priors
from bsfh.datautils import logify_data, norm_spectrum

param_template = {'name':'', 'N':1, 'isfree': False,
                  'init':0.0, 'units':'',
                  'prior_function_name': None, 'prior_args': None}

class ProspectrParams(object):
    """
    :param mp:
        A list of ``model parameters``.
    
    """
    # information about each parameter stored as a list
    #config_list = [] 
    # information about each parameter as a dictionary keyed by
    # parameter name for easy access
    #_config_dict = {}
    # Model parameter state, keyed by parameter name
    #params = {}
    # Mapping from parameter name to index in the theta vector
    #theta_index = {}
    # the initial theta vector
    #theta_init = np.array([])

    # Information about the fiting parameters, filenames, fitting and
    # configure options, runtime variables, etc.
    #run_params = {}
    #obs = {}
    
    def __init__(self, config_list, verbose=True):
        self.config_list = config_list
        self.configure()
        self.verbose = verbose

    def configure(self, reset=False, **kwargs):
        """
        Use the parameter config_list to generate a theta_index
        mapping, and propogate the initial parameters into the params
        state dictionary, and store the intital theta vector implied
        by the config dictionary.

        :param kwargs:
            Keyword parameters can be used to override or add to the
            initial parameter values specified in  config_list
        """
        if (not hasattr(self, 'params')) or reset:
            self.params = {}
        self._config_dict = plist_to_pdict(self.config_list)
        self.map_theta()
        # propogate initial parameter values from the configure dictionary
        for par, info in self._config_dict.iteritems():
            self.params[par] = np.atleast_1d(info['init'])
        # propogate user supplied values, overriding the configure
        for k,v in kwargs.iteritems():
            self.params[k] = np.atleast_1d(v)
        # store these initial values
        self.initial_theta = self.rectify_theta((self.theta.copy()))
        #print('pivot_wave' in self.params)
        
    def map_theta(self):
        """
        Construct the mapping from parameter name to the index in the
        theta vector corresponding to the first element of that
        parameter.
        """
        self.theta_index = {}
        count = 0
        for par in self.free_params:
            self.theta_index[par] = (count,
                                     count + self._config_dict[par]['N'])
            count += self._config_dict[par]['N']
        self.ndim = count

    def set_parameters(self, theta):
        """
        Propagate theta into the model parameters.

        :param theta:
            A theta parameter vector containing the desired
            parameters.  ndarray of shape (ndim,)
        """
        assert len(theta) == self.ndim
        for k, v in self.theta_index.iteritems():
            start, end = v
            self.params[k] = np.atleast_1d(theta[start:end])

    def prior_product(self, theta):
        return self.simple_prior_product(theta)
            
    def simple_prior_product(self, theta):
        """
        Return a scalar which is the ln of the product of the prior
        probabilities for each element of theta.  Requires that the
        prior functions are defined in the theta descriptor.

        :param theta:
            Iterable containing the free model parameter values.

        :returns lnp_prior:
            The log of the product of the prior probabilities for
            these parameter values.
        """
        lnp_prior = 0
        for k, v in self.theta_index.iteritems():
            start, end = v
            #print(k)
            lnp_prior += np.sum(self._config_dict[k]['prior_function']
                                (theta[start:end], **self._config_dict[k]['prior_args']))
        return lnp_prior
            
    def rescale_parameter(self, par, func):
        ind = [p['name'] for p in self.config_list].index(par)
        self.config_list[ind]['init'] = func(self.config_list[ind]['init'])
        for k,v in self.config_list[ind]['prior_args'].iteritems():
            self.config_list[ind]['prior_args'][k] = func(v)
            
    def rectify_theta(self, theta):
        tiny_number = 1e-10
        zero = (theta == 0)
        theta[zero] = tiny_number
        return theta
    
    @property
    def theta(self):
        """The current value of the theta vector, pulled from the
        params state dictionary.
        """
        theta = np.zeros(self.ndim)
        for k, v in self.theta_index.iteritems():
            start, end = v
            theta[start:end] = self.params[k]
        return theta

    @property
    def free_params(self):
        """A list of the free model parameters.
        """
        return [k['name'] for k in pdict_to_plist(self.config_list)
                if k['isfree']]

    @property
    def fixed_params(self):
        """A list of the fixed model parameters that are specified in
        the `model_params`.
        """
        return [k['name'] for k in pdict_to_plist(self.config_list)
                if (k['isfree'] is False)]

    def theta_labels(self, name_map={'amplitudes':'A',
                                     'emission_luminosity':'eline'}):
        """
        Using the theta_index parameter map, return a list of
        the model parameter names that has the same order as the
        sampling chain array.

        :param name_map:
            A dictionary mapping model parameter names to output label
            names.
            
        :returns labels:
            A list of labels of the same length and order as the theta
            vector.
        """
        label, index = [], []
        for p,v in self.theta_index.iteritems():
            nt = v[1]-v[0]
            try:
                name = name_map[p]
            except(KeyError):
                name = p
            if nt is 1:
                label.append(name)
                index.append(v[0])
            else:
                for i in xrange(nt):
                    label.append(name+'{0}'.format(i+1))
                    index.append(v[0]+i)
        return [l for (i,l) in sorted(zip(index,label))]
    
    def info(self, par):
        pass

    def reconfigure(self, par, config):
        pass

    def write_json(self, filename):
        pass

    def theta_bounds(self):
        bounds = self.ndim * [(0.,0.)]
        for p, v in self.theta_index.iteritems():
            sz = v[1] - v[0]
            pb = priors.plotting_range(self._config_dict[p]['prior_args'])
            if sz == 1:
                bounds[v[0]] = pb
            else:
                for k in range(sz):
                    bounds[v[0]+k] = (pb[0][k], pb[1][k])

        #force types
        bounds = [(np.atleast_1d(a)[0], np.atleast_1d(b)[0]) for a,b in bounds]        
        return bounds

    
def plist_to_pdict(inplist):
    """Convert from a parameter list to a parameter dictionary, where
    the keys of the cdictionary are the parameter names.
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
    """Convert from a parameter dictionary to a parameter list of
    dictionaries, adding each key to each value dictionary as the
    `name' keyword.
    """
    if type(pdict) is list:
        return pdict[:]
    plist = []
    for k, v in pdict.iteritems():
        v['name'] = k
        plist += [v]
    return plist

def write_plist(plist, runpars, filename=None):
    """Write the list of parameter dictionaries to a JSON file, taking
    care to replace functions with their names.
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
    """Read a JSON file into a run_param dictionary and a list of
    model parameter dictionaries, taking care to add actual functions
    when given their names.
    """
    
    with open(filename, 'r') as f:
        runpars, modelpars = json.load(f)
    runpars['param_file'] = filename
    if raw_json:
        return runpars, modelpars
    
    for p in modelpars:
        p = names_to_functions(p)
        
    return runpars, modelpars

def names_to_functions(p):
    """Replace names of functions in a parameter description with the
    actual functions.
    """
    #put the dust curve function in
    if 'dust_curve_name' in p:
        from sedpy import attenuation
        p['init'] = attenuation.__dict__[p['dust_curve_name']]
    #put the prior function in
    if 'prior_function_name' in p:
        p['prior_function'] = priors.__dict__[p['prior_function_name']]
        #print(p['prior_function_name'], p['prior_function'])
    else:
        p['prior_function'] = p.get('prior_function', None)
    return p

def functions_to_names(p):
    """Replace prior and dust functions with the names of those
    functions.
    """
    pf = p.get('prior_function', None)
    cond = ((pf in priors.__dict__.values()) and
            (pf is not None))
    if cond:
        p['prior_function_name'] = pf.func_name
    else:
        p.pop('prior_function_name', None)
    _ = p.pop('prior_function', None)
        
    #replace dust curve functions with name of function
    if p['name'] == 'dust_curve':
        from sedpy import attenuation
        df = p.get('init', None)
        cond = ((df is not None) and
                (df in attenuation.__dict__.values()))
        if cond:
            p['dust_curve_name'] = df.func_name
            _ = p.pop('init', None)
    return p
