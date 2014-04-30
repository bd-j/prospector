#!/usr/local/bin/python

import time, sys, os, getopt
import numpy as np
import pickle
from scipy.optimize import minimize

try:
    import astropy.io.fits as pyfits
except(ImportError):
    import pyfits

import observate
from fsps import StellarPopulation, find_filter
from clustermodel import ClusterModel
from priors import tophat
import gp
import fitterutils

########
# SETUP
##########
rp = {'verbose':False, 'powell_results':None, #'results/dao69.imf3_2.3_powell_1394030533',
      'file':'data/mmt/DAO69.fits', 'dist':0.783, 'vel':0.,
      'ftol':3., 'maxfev':200, 'guess_factor':2, 'nsamplers':1,
      'walker_factor':10, 'nthreads':1, 'nburn':1 * [10], 'niter': 10, 'initial_disp':0.1,
      'imf3':2.7}
    
rp['outfile'] = 'results/' + os.path.basename(rp['file']).split('.')[0].lower()
rp = fitterutils.parse_args(sys.argv,rp)

lsun, pc = 3.846e33, 3.085677581467192e18 #in cgs
to_cgs = lsun/10**( np.log10(4.0*np.pi)+2*np.log10(pc*10) )

#print(rp['imf3'])

############
# REAL OBS EXAMPLE
##############
obs = fitterutils.load_obs(rp)
#ignore the IR magnitudes
obs['mags'] = obs['mags'][0:4]
obs['mags_unc'] = obs['mags_unc'][0:4]
obs['filters'] = obs['filters'][0:4]

###############
#MODEL SET UP
##############
if rp['verbose']:
    print('Setting up model')
masstheta = {'i0':0, 'N': 1, 'dtype':'<f8', 
             'prior_function':tophat,
             'prior_args':{'mini':1e2, 'maxi': 1e6}}

agestheta = {'i0':1, 'N': 1,  'dtype':'<f8',
             'prior_function':tophat,
             'prior_args':{'mini':0.0001, 'maxi':2.5}}

#zmetstheta = {'i0':2, 'N': 1,'dtype':'<f8',
#             'prior_function':tophat,
#             'prior_args':{'mini':2.51, 'maxi':4.5}}
        
dusttheta = {'i0':2, 'N': 1, 'dtype':'<f8', 
             'prior_function':tophat,
             'prior_args':{'mini':0.0, 'maxi':2.5}}

veltheta = {'i0':3, 'N': 1, 'dtype':'<f8', 
             'prior_function':tophat,
             'prior_args':{'mini':100.0, 'maxi':200.0}}

redtheta = {'i0':4, 'N':1, 'dtype':'<f8',
            'prior_function':tophat,
            'prior_args':{'mini':-0.001, 'maxi':0.001}}
order = 2 #up to x**3
polytheta = {'i0':5, 'N':order, 'dtype':'<f8',
            'prior_function':tophat,
            'prior_args':{'mini':np.array([-3,-5]), 'maxi':np.array([3,5])}}

normtheta = {'i0':5+order, 'N':1, 'dtype':'<f8',
            'prior_function':tophat,
            'prior_args':{'mini':0.1, 'maxi':10}}

imftheta = {'i0':6+order, 'N':1, 'dtype':'<f8',
            'prior_function':tophat,
            'prior_args':{'mini':1.3, 'maxi':3.5}}
    
theta_desc = {'mass':masstheta, 'tage': agestheta, 'dust2':dusttheta,
              'zred':redtheta,'vel_broad':veltheta,
              #'zmet':zmetstheta,
              'imf3':imftheta,
              'poly_coeffs':polytheta,'spec_norm':normtheta}

sps_fixed_params = {'sfh':0, 'imf_type': 2, 'dust_type': 1,# 'imf3': 2.3,
                    'zmet':4}
gp_pars = {'s':0, 'a': 1.0, 'l':100}

    
model = ClusterModel(theta_desc)
model.add_obs(obs)
model.ndof = len(model.obs['wavelength']) + len(model.obs['mags'])
model.verbose = rp['verbose']

    
sps = StellarPopulation(compute_vega_mags = False)
for k, v in sps_fixed_params.iteritems():
    sps.params[k] = v
    if k in model.ssp_pars_from_theta:
        model.ssp_pars_from_theta.remove(k)


model.sps_fixed_params = sps_fixed_params
model.verbose = False
ndim = 0
for p, d in model.theta_desc.iteritems():
    ndim += d['N']
rp['ndim'] = ndim

mask = model.obs['mask']
gprocess = gp.GaussianProcess(model.obs['wavelength'][mask], model.obs['unc'][mask])
gprocess.factor(gp_pars['s'], gp_pars['a'], gp_pars['l'])

def lnprobfn(theta, mod):
    """wrapper on the model instance method, defined here
    globally to enable multiprocessing"""
    #print(theta)
    lnp_prior = mod.prior_product(theta)
    if np.isfinite(lnp_prior):
        t = time.time()
        spec, phot, x = mod.model(theta, sps = sps)
        d1 = time.time() - t
        r = (mod.obs['spectrum'] - spec)[mod.obs['mask']]
        t = time.time()
        lnp_spec = gprocess.lnlike(r)
        d2 = time.time() - t
    #    print('model calc = {0}s, lnlike calc = {1}'.format(d1,d2))
        maggies = 10**(-0.4 * mod.obs['mags'])
        phot_var = (maggies * mod.obs['mags_unc']/1.086)**2 
        lnp_phot =  -0.5*( (phot - maggies)**2 / phot_var ).sum()
    #    print('lnp = {0}, lnp_spec = {1}, lnp_phot = {2}'.format(lnp_spec + lnp_phot + lnp_prior, lnp_spec, lnp_phot))
        return lnp_prior + lnp_phot + lnp_spec
    else:
        return -np.infty
    
#################
#INITIAL GUESS OF SPECTRAL NORMALIZATION USING PHOTOMETRY
#################
#use f475w for normalization
norm_band = [i for i,f in enumerate(model.obs['filters']) if 'f475w' in f.name][0]
synphot = observate.getSED(model.obs['wavelength'], model.obs['spectrum'] * to_cgs, model.obs['filters'])
#factor by which model spectra should be multiplied to give you the observed spectra, using the F475W filter as truth
norm = 10**(-0.4 * (synphot[norm_band]  - model.obs['mags'][norm_band]))
#assume you've got this right to within 5% (and 3 sigma) after marginalized over everything
#  that changes spectral shape within the band (polynomial terms, dust, age, etc)
fudge = (1 + 5 * model.obs['mags_unc'][norm_band]/1.068) * 1.05
model.theta_desc['spec_norm']['prior_args'] = {'mini':norm/fudge, 'maxi':norm * fudge}
#pivot the polynomial near the filter used for approximate normalization
model.cal_pars['pivot_wave'] =  model.obs['filters'][norm_band].wave_effective 
model.cal_pars['pivot_wave'] = 4750.
if rp['verbose']:
    print('spectral normalization guess = {0}'.format(norm))

#################
#INITIAL GUESS USING POWELL MINIMIZATION
#################

def chi2(theta):
    """A sort of chi2 function that allows for maximization of lnP using minimization routines"""
    return -lnprobfn(theta, model)
powell_opt = {'ftol':rp['ftol']/model.ndof * 2., 'maxfev':rp['maxfev']}

initial_center = np.array([8e3, 2e-2, 0.5, 160., -0.0002, 0.1, 0.1, norm, 2.3])
powell_guess = minimize(chi2, initial_center, method = 'powell',options = powell_opt)

#expand the allowed range for IMF and normalization
#model.theta_desc['spec_norm']['prior_args'] = {'mini':norm/1.2, 'maxi':norm * 1.2}

###################
#SAMPLE
####################
if rp['verbose']:
    print('emcee...')

nsamplers = int(rp['nsamplers'])

tstart = time.time()
initial_center = powell_guess.x #np.array([8e3, 2e-2, 0.5, 0.1, 0.1, norm])
esampler = fitterutils.run_a_sampler(model, sps, lnprobfn, initial_center, rp)
edur = time.time() - tstart


results = {}
results['run_params'] = rp
results['obs'] = model.obs
results['theta'] = model.theta_desc
results['initial_center'] = initial_center
results['chain'] = esampler.chain
results['lnprobability'] = esampler.lnprobability
results['acceptance'] = esampler.acceptance_fraction
results['duration'] = edur
results['model'] = model
results['gp'] = gprocess

out = open('{1}.imf3_{3}_sampler{2:02d}_mcmc_{0}'.format(int(time.time()), rp['outfile'], 1, sps.params['imf3']), 'wb')
pickle.dump(results, out)
out.close()

