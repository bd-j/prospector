from copy import deepcopy
import numpy as np
import json
from sedpy import observate, attenuation
from bsfh import priors, sedmodel, elines, gp

tophat = priors.tophat

rp = {'verbose':True,
      'filename':'data/mmt/nocal/020.B192-G242.s.fits',
      'objname': 'B192-G242',
      'outfile':'results/test',
      'wlo':3750., 'whi': 7200.,
      'ftol':0.5e-5, 'maxfev':500, 'nsamplers':1,
      'walker_factor':3, 'nthreads':1, 'nburn':3 * [10], 'niter': 10, 'initial_disp':0.1
      }

default_parlist = []
default_parlist.append({'name': 'lumdist', 'N': 1,
                        'isfree': False,
                        'init': 0.783,
                        'units': 'Mpc',
                        'prior_function': None,
                        'prior_args': None})

###### SFH ################

default_parlist.append({'name': 'mass', 'N': 1,
                        'isfree': True,
                        'init': 10e3,
                        'units': r'M$_\odot$',
                        'prior_function': tophat,
                        'prior_args': {'mini':1e2, 'maxi': 1e6}})

default_parlist.append({'name': 'tage', 'N': 1,
                        'isfree': True,
                        'init': 0.250,
                        'units': 'Gyr',
                        'prior_function':tophat,
                        'prior_args':{'mini':0.001, 'maxi':2.5}})

default_parlist.append({'name': 'zmet', 'N': 1,
                        'isfree': True,
                        'init': -0.2,
                        'units': r'$\log (Z/Z_\odot)$',
                        'prior_function': tophat,
                        'prior_args': {'mini':-1, 'maxi':0.19}})

default_parlist.append({'name': 'sfh', 'N':1,
                        'isfree': False,
                        'init': 0,
                        'units': None})

###### DUST ##################

default_parlist.append({'name': 'dust_curve', 'N': 1,
                        'isfree': False,
                        'init': attenuation.cardelli,
                        'units': None})

default_parlist.append({'name': 'dust2', 'N': 1,
                        'isfree': True,
                        'init': 0.5,
                        'units': r'$\tau_V$',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':2.5}})

default_parlist.append({'name': 'dust1', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': r'$\tau_V$',
                        'prior_function': None,
                        'prior_args': None})

default_parlist.append({'name': 'dust_tesc', 'N': 1,
                        'isfree': False,
                        'init': 0.01,
                        'units': 'Gyr',
                        'prior_function': None,
                        'prior_args': None})

default_parlist.append({'name': 'dust_type', 'N': 1,
                        'isfree': False,
                        'init': 1,
                        'units': 'NOT USED'})

###### IMF ###################

default_parlist.append({'name': 'imf_type', 'N': 1,
                        'isfree': False,
                        'init': 2, #2 = kroupa
                        'units': None})

default_parlist.append({'name': 'imf3', 'N':1,
                        'isfree': False,
                        'init': 2.3,
                        'units': None,
                        'prior_function':tophat,
                        'prior_args':{'mini':1.3, 'maxi':3.5}})

###### WAVELENGTH SCALE ######

default_parlist.append({'name': 'zred', 'N':1,
                        'isfree': True,
                        'init': 0.00001,
                        'units': None,
                        'prior_function': tophat,
                        'prior_args': {'mini':-0.001, 'maxi':0.001}})

default_parlist.append({'name': 'sigma_smooth', 'N': 1,
                        'isfree': True,
                        'init': 2.2,
                        'units': r'$\AA$',
                        'prior_function': tophat,
                        'prior_args': {'mini':1.0, 'maxi':6.0}})

default_parlist.append({'name': 'smooth_velocity', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'units': None})

default_parlist.append({'name': 'min_wave_smooth', 'N': 1,
                        'isfree': False,
                        'init': 3700.0,
                        'units': r'$\AA$'})

default_parlist.append({'name': 'max_wave_smooth', 'N': 1,
                        'isfree': False,
                        'init': 7300.0,
                        'units': r'$\AA$'})

###### CALIBRATION ###########

polyorder = 2
polymin = [-5, -50]
polymax = [5, 50]
polyinit = [0.1, 0.1]

default_parlist.append({'name': 'poly_coeffs', 'N': polyorder,
                        'isfree': True,
                        'init': polyinit,
                        'units': None,
                        'prior_function': tophat,
                        'prior_args': {'mini':polymin, 'maxi':polymax}})
    
default_parlist.append({'name': 'spec_norm', 'N':1,
                        'isfree': True,
                        'init':1,
                        'units': None,
                        'prior_function': tophat,
                        'prior_args': {'mini':0.1, 'maxi':10}})

default_parlist.append({'name': 'gp_jitter', 'N':1,
                        'isfree': True,
                        'init': 0.001,
                        'units': 'spec units',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':0.1}})

default_parlist.append({'name': 'gp_amplitude', 'N':1,
                        'isfree': True,
                        'init': 0.04,
                        'units': 'spec units',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':0.2}})

default_parlist.append({'name': 'gp_length', 'N':1,
                        'isfree': True,
                        'init': 100.0,
                        'units': r'$\AA$',
                        'prior_function': tophat,
                        'prior_args': {'mini':20.0, 'maxi':500}})

default_parlist.append({'name': 'phot_jitter', 'N':1,
                        'isfree': True,
                        'init': 0.01,
                        'units': 'mags',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':0.1}})

###### EMISSION ##############

linelist = ['CaII_K', 'NaI_5891', 'NaI_5897',
            'Ha', 'NII_6585','SII_6718','SII_6732',
            'HeI_3821','HeI_4010','HeI_4027','HeI_4145', 'HeI_4389','HeI_4473','HeI_4923','HeI_5017'
            ]
linemin = 3 * [-100] + 4 * [0.]  + 8 * [-50.0 ]
linemax = 3 * [0.] + 4 * [100.0 ] + 8 * [50.0 ]
lineinit = 3 * [-0.1 ] + 4 * [1.0 ] + 8 * [0.1 ]

nlines = len(linelist)
ewave = [elines.wavelength[l] for l in linelist]

default_parlist.append({'name': 'emission_rest_wavelengths', 'N': nlines,
                        'isfree': False,
                        'init': ewave,
                        'line_names': linelist,
                        'units': r'$\AA$'})

default_parlist.append({'name': 'emission_luminosity', 'N': nlines,
                        'isfree': True,
                        'init': lineinit,
                        'units': r'$L_\odot$',
                        'prior_function':tophat,
                        'line_names': linelist,
                        'prior_args':{'mini': linemin, 'maxi': linemax}})

default_parlist.append({'name': 'emission_disp', 'N': 1, 'isfree': True,
                        'init': 2.2,
                        'units': r'$\AA$',
                        'prior_function':tophat,
                        'prior_args':{'mini':1.0, 'maxi':6.0}})


class ProspectrParams(object):
    """
    Keep the parameters stored in an object.  experimental/under dev
    """
    def __init__(self, filename=None):
        if filename is not None:
            self.read_from_json(filename=filename)
        else:
            self.run_params = rp.copy()
            self.model_params = deepcopy(default_parlist)
            #self.model_params = plist_to_pdict(self.model_params)

    #def __repr__(self):
    #    pass
    
    def write_to_json(self, filename=None):
        if filename is not None:
            self.filename = filename
        write_plist(pdict_to_plist(self.model_params),
                    self.run_params, self.filename)

    def read_from_json(self, filename=None):
        if filename is not None:
            self.filename = filename
        self.run_params, self.model_params = read_plist(self.filename)
        #self.model_params = plist_to_pdict(self.model_params)


    def get_theta_desc(self):
        plist = deepcopy(pdict_to_plist(self.model_params))
        return get_theta_desc(plist)
    
    @property
    def free_params(self):
        return [k for k, v in self.model_params.iteritems()
                if v['isfree']]
            
    @property
    def fixed_params(self):
        return [k for k, v in self.model_params.iteritems()
                if (v['isfree']==False)]

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

def write_plist(plist, runpars, filename):
    """
    Write the list of parameter dictionaries to a JSON file,
    taking care to replace functions with their names.
    """
    runpars['param_file'] = filename
    for p in plist:
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
        
    f = open(filename + '.bpars.json', 'w')
    json.dump([rp, plist], f)
    f.close()    

def read_plist(filename, raw_json  = False):
    """
    Read a JSON file into a run_param dictionary and a list of model
    parameter dictionaries, taking care to add actual functions when
    given their names.
    """
    
    f = open(filename, 'r')
    runpars, modelpars = json.load(f)
    f.close()
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
                    "{0} not same as declared size".format(name))
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
    #use f475w for normalization
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
    model.params['pivot_wave'] = 4750.
 
    initial_center[model.theta_desc['spec_norm']['i0']] = 1.0
 
    return model, initial_center


