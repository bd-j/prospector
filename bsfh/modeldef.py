from copy import deepcopy
import numpy as np
import json
from sedpy import observate, attenuation
from bsfh import priors, sedmodel, elines, gp
from bsfh.datautils import logify

param_template = {'name':'', 'N':1, 'isfree': False,
                  'init':0.0, 'units':'',
                  'prior_function_name': None, 'prior_args': None}

from bsfh.default_params import default_parlist, rp

class ProspectrParams(object):
    """
    Keep the parameters stored in an object.
    """
    def __init__(self, filename=None):
        if filename is not None:
            self.read_from_json(filename=filename)
        else:
            self.run_params = rp.copy()
            self.model_params = deepcopy(default_parlist)
            self.run_params['param_file'] = None
    
    def write_to_json(self, filename=None):
        if filename is not None:
            self.run_params['param_file'] = filename
        write_plist(pdict_to_plist(self.model_params),
                    self.run_params, self.run_params['param_file'])

    def read_from_json(self, filename=None, **kwargs):
        if filename is not None:
            self.filename = filename
        self.run_params, self.model_params = read_plist(self.filename,
                                                        **kwargs)
        self.run_params['param_file'] = self.filename

    def get_theta_desc(self):
        plist = deepcopy(pdict_to_plist(self.model_params))
        return get_theta_desc(plist)
    
    @property
    def free_params(self):
        return [k['name'] for k in pdict_to_plist(self.model_params)
                if k['isfree']]
            
    @property
    def fixed_params(self):
        return [k['name'] for k in pdict_to_plist(self.model_params)
                if (k['isfree']==False)]

    def parindex(self, parname):
        return [p['name'] for p in
                pdict_to_plist(self.model_params)].index(parname)
        
    def parinfo(self, parname):
        try:
            return self.run_params[parname]
        except(KeyError):
            return self.model_params[parname] 
        finally:
            return self.model_params[self.parindex(parname)]

    def initialize_model(self, modelclass):
        
        tdesc, init, fixed = self.get_theta_desc()
        model = modelclass(theta_desc=tdesc, **fixed)
        model.verbose = self.run_params['verbose']
        
        return model, init

def add_obs_to_model(model, obs, initial_center,
                     spec=True, phot=True,
                     logify_spectrum=True, normalize_spectrum=True,
                     add_gaussian_process=True):

    """ Needs to be rewritten to more gracefully determine whether
    spec or phot data being fit.
    """
    
    model.add_obs(obs)
    model.ndof = 0
    
    if (normalize_spectrum and spec):
        model, initial_center = norm_spectrum(model,
                                              initial_center)
        
    if (add_gaussian_process and spec):
        mask = model.obs['mask']
        model.gp = gp.GaussianProcess(model.obs['wavelength'][mask],
                                      model.obs['unc'][mask])
    if (logify_spectrum and spec):
        s, u, m = logify(model.obs['spectrum'], model.obs['unc'],
                         model.obs['mask'])
        model.obs['spectrum'] = s
        model.obs['unc'] = u
        model.obs['mask'] = m
        if normalize_spectrum:
            fudge = np.log(model.theta_desc['spec_norm']['prior_args']['maxi'])
            model.theta_desc['spec_norm']['prior_args'] = {'mini':-fudge,
                                                           'maxi':fudge }
            initial_center[model.theta_desc['spec_norm']['i0']] = 1e-2
    if not spec:
        model.obs['spectrum'] = None
        model.obs['unc'] = None
    if not phot:
        model.obs['mags'] = None
        model.obs['mags_unc'] = None

    if spec:
        model.ndof += len(model.obs['spectrum'])
    if phot:
        model.ndof += len(model.obs['mags'])
        
def plist_to_pdict(plist):
    """Convert from a parameter list to a parameter dictionary, where
    the keys of the cdictionary are the parameter names.
    """
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
    """
    Write the list of parameter dictionaries to a JSON file,
    taking care to replace functions with their names.
    """
    
    for p in plist:
        p = functions_to_names(p)

    if filename is not None:
        runpars['param_file'] = filename
        f = open(filename + '.bpars.json', 'w')
        json.dump([rp, plist], f)
        f.close()    
    else:
        return json.dumps([rp, plist])
    
def read_plist(filename, raw_json=False):
    """
    Read a JSON file into a run_param dictionary and a list
    of model parameter dictionaries, taking care to add actual
    functions when given their names.
    """
    
    with open(filename, 'r') as f:
        runpars, modelpars = json.load(f)
    rp['param_file'] = filename
    if raw_json:
        return runpars, modelpars
    
    for p in modelpars:
        p = names_to_functions(p)
        
    return runpars, modelpars

def names_to_functions(p):
    #print(p['name'], p.get('prior_function_name','nope'))
    #put the dust curve function in
    if 'dust_curve_name' in p:
        p['init'] = attenuation.__dict__[p['dust_curve_name']]
    #put the prior function in
    if 'prior_function_name' in p:
        p['prior_function'] = priors.__dict__[p['prior_function_name']]
        #print(p['prior_function_name'], p['prior_function'])
    else:
        p['prior_function'] = None
    return p

def functions_to_names(p):
    #replace prior functions with names of those function
    pf = p.get('prior_function', None)
    #print(p['name'], pf)
    cond = ((pf in priors.__dict__.values()) and
            (pf is not None))
    if cond:
        p['prior_function_name'] = pf.func_name
    else:
        p.pop('prior_function_name', None)
    _ = p.pop('prior_function', None)
        
    #replace dust curve functions with name of function
    if p['name'] == 'dust_curve':
        df = p.get('init', None)
        cond = ((df is not None) and
                (df in attenuation.__dict__.values()))
        if cond:
            p['dust_curve_name'] = df.func_name
            _ = p.pop('init', None)
    return p

def get_theta_desc(model_params):
        
    theta_desc, fixed_params  = {}, {}
    theta_init = []
    count = 0
    for i, par in enumerate(model_params):
        if par['isfree']:
            par['i0'] = count
            par = names_to_functions(par)
            theta_desc[par['name']] = par
            count += par['N']
            #need to deal with iterables here
            init = np.array(par['init']).flatten().tolist()
            n = len(init)
            if par['N'] != n:
                raise TypeError("Parameter value vector of "\
                    "{0} not same as declared size".format(par['name']))
            # Finally, append this to the initial
            #  parameter vector
            theta_init += init
        else:
            fixed_params[par['name']] = par['init']

    return theta_desc, theta_init, fixed_params
   
def initialize_model(rp, plist, obs):
    """
    Take a run parameter dictionary and a model parameter list and
    return a SedModel object, as well as a vector of initial values.
    """

    tdesc, init, fixed = get_theta_desc(deepcopy(plist))

    # SED Model
    model = sedmodel.SedModel(theta_desc = tdesc, **fixed)
    model.add_obs(obs)
    model, initial_theta = norm_spectrum(model, init)
    #model.params['pivot_wave'] = 4750.
    model.ndof = len(model.obs['wavelength']) + len(model.obs['mags'])
    model.verbose = rp['verbose']

    #Add Gaussian Process
    mask = model.obs['mask']
    model.gp = gp.GaussianProcess(model.obs['wavelength'][mask],
                                  model.obs['unc'][mask])

    return model, initial_theta


def norm_spectrum(model, initial_center, band_name='f475w'):
    """
    Initial guess of spectral normalization using photometry.

    This multiplies the observed spectrum by the factor required
    to reproduce the photometry.  Default is to produce a spectrum
    that is approximately in erg/s/cm^2/AA (Lsun/AA/Mpc**2).

    The inverse of the multiplication factor is saved as a fixed
    parameter to be used in producing the mean model.
    """
    
    norm_band = [i for i,f in enumerate(model.obs['filters'])
                 if band_name in f.name][0]
    
    synphot = observate.getSED(model.obs['wavelength'],
                               model.obs['spectrum'],
                               model.obs['filters'])

    # Factor by which the observed spectra should be multiplied to give you
    #  the photometry (or the cgs apparent spectrum), using the F475W filter as truth
    norm = 10**(-0.4*(synphot[norm_band] - model.obs['mags'][norm_band]))
    model.params['normalization_guess'] = norm
       
    # Assume you've got this right to within some factor after
    #  marginalized over everything that changes spectral shape within
    #  the band (polynomial terms, dust, age, etc)
    #fudge = (1 + 10 * model.obs['mags_unc'][norm_band]/1.086)
    fudge = 3.0
    model.theta_desc['spec_norm']['prior_args'] = {'mini':1 / fudge,
                                                   'maxi':fudge }

    # Pivot the polynomial near the filter used for approximate
    # normalization
    model.params['pivot_wave'] =  model.obs['filters'][norm_band].wave_effective 
    #model.params['pivot_wave'] = 4750.
 
    initial_center[model.theta_desc['spec_norm']['i0']] = 1.0
 
    return model, initial_center


