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
import fitterutils

########
# SETUP
##########
rp = {'verbose':True, 'powell_results':None, #'results/dao69.imf3_2.3_powell_1394030533',
      'file':'data/mmt/DAO69.fits', 'dist':0.783, 'vel':0.,
      'ftol':3., 'maxfev':200, 'guess_factor':3, 'nsamplers':2,
      'walker_factor':10, 'nthreads':4, 'nburn':10 * [100], 'niter': 5000, 'initial_disp':0.1,
      'imf3':2.7}
    
rp['outfile'] = 'results/' + os.path.basename(rp['file']).split('.')[0].lower()
rp = fitterutils.parse_args(sys.argv,rp)

lsun, pc = 3.846e33, 3.085677581467192e18 #in cgs
to_cgs = lsun/10**( np.log10(4.0*np.pi)+2*np.log10(pc*10) )

print(rp['imf3'])

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

zmetstheta = {'i0':2, 'N': 1,'dtype':'<f8',
             'prior_function':tophat,
             'prior_args':{'mini':2.51, 'maxi':4.5}}
        
dusttheta = {'i0':3, 'N': 1, 'dtype':'<f8', 
             'prior_function':tophat,
             'prior_args':{'mini':0.0, 'maxi':2.5}}

veltheta = {'i0':4, 'N': 1, 'dtype':'<f8', 
             'prior_function':tophat,
             'prior_args':{'mini':100.0, 'maxi':200.0}}

redtheta = {'i0':5, 'N':1, 'dtype':'<f8',
            'prior_function':tophat,
            'prior_args':{'mini':-0.001, 'maxi':0.001}}
order = 4 #up to x**3
polytheta = {'i0':6, 'N':order, 'dtype':'<f8',
            'prior_function':tophat,
            'prior_args':{'mini':np.array([-3,-5,-10, -20]), 'maxi':np.array([3,5,10, 20])}}

normtheta = {'i0':6+order, 'N':1, 'dtype':'<f8',
            'prior_function':tophat,
            'prior_args':{'mini':0.1, 'maxi':10}}

theta_desc = {'mass':masstheta, 'tage': agestheta, 'zmet':zmetstheta,
              'dust2':dusttheta, 'vel_broad':veltheta,
              'zred':redtheta, 'poly_coeffs':polytheta,'spec_norm':normtheta}

model = ClusterModel(theta_desc)
model.add_obs(obs)
model.ndof = len(model.obs['wavelength']) + len(model.obs['mags'])
model.verbose = rp['verbose']

sps_fixed_params = {'sfh':0, 'imf_type': 2, 'dust_type': 1, 'imf3': rp['imf3']}
sps = StellarPopulation(compute_vega_mags = False)
for k, v in sps_fixed_params.iteritems():
    sps.params[k] = v
model.ssp_pars_from_theta.remove('imf3')
model.sps_fixed_params = sps_fixed_params

ndim = 0
for p, d in model.theta_desc.iteritems():
    ndim += d['N']
rp['ndim'] = ndim

def lnprobfn(theta, mod):
    """wrapper on the model instance method, defined here
    globally to enable multiprocessing"""
    return mod.lnprob(theta, sps = sps)

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
fudge = (1 + 3 * model.obs['mags_unc'][norm_band]/1.068) * 1.05
model.theta_desc['spec_norm']['prior_args'] = {'mini':norm/fudge, 'maxi':norm * fudge}
#pivot the polynomial near the filter used for approximate normalization
model.cal_pars['pivot_wave'] =  model.obs['filters'][norm_band].wave_effective 
model.cal_pars['pivot_wave'] = 4750.
if rp['verbose']:
    print('spectral normalization guess = {0}'.format(norm))

#################
#INITIAL GUESS USING POWELL MINIMIZATION
#################
if rp['verbose']:
    print('Initial minimization...')

def chi2(theta):
    """A sort of chi2 function that allows for maximization of lnP using minimization routines"""
    return -model.lnprob(theta, sps = sps)

def pminimize(pminargs):
    """For multiprocessing of the minimization"""
    pos, powell_opt = pminargs[0], pminargs[1]
    return minimize(chi2, pos, method = 'powell',options = powell_opt)

if rp['powell_results'] is None:
    tstart = time.time()
    powell_opt = {'ftol':rp['ftol']/model.ndof * 2., 'maxfev':rp['maxfev']}

    powell_guesses, pinitial, strictopt = fitterutils.initialize_params(chi2, pminimize, model, sps, rp, powell_opt)
    fsort = np.argsort([pg.fun for pg in powell_guesses])
    idur = time.time() - tstart

    presults = {}
    presults['run_params'] = rp
    presults['powell_initial'] = pinitial
    presults['powell_guesses'] = powell_guesses
    presults['initialization_duration'] = idur
    presults['final_powell_opts'] = strictopt
    presults['model'] = model

    rp['powell_results'] = '{1}.imf3_{2}_powell_{0}'.format(int(time.time()), rp['outfile'], sps.params['imf3'])
    out = open(rp['powell_results'], 'wb')
    pickle.dump(presults, out)
    out.close()
    
else:
    f = open(rp['powell_results'], 'rb')
    presults = pickle.load(f)
    f.close()
    fsort = np.argsort([pg.fun for pg in presults['powell_guesses']])

#expand the allowed range for IMF and normalization
#model.theta_desc['spec_norm']['prior_args'] = {'mini':norm/1.2, 'maxi':norm * 1.2}

###################
#SAMPLE
####################
if rp['verbose']:
    print('emcee...')

nsamplers = int(rp['nsamplers'])

for isampler in range(nsamplers):
    tstart = time.time()
    initial_center = presults['powell_guesses'][fsort[isampler]].x
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
    
    out = open('{1}.imf3_{3}_sampler{2:02d}_mcmc_{0}'.format(int(time.time()), rp['outfile'], isampler+1, sps.params['imf3']), 'wb')
    pickle.dump(results, out)
    out.close()

