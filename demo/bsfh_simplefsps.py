#script for playing with SFH inference at fixed population parameters


import time
import sys
import numpy as np
from scipy.optimize import minimize

import fsps
import cspmodel

import hmc
import emcee
import acor
from convergence import gelman_rubin_R as grR
import priors

rp = {}
snr = 100.
nage = 15
logt0 = 7.0
dlogt = 0.2
filters = None
if filters: filters = observate.load_filters(filters)

###############
# SET UP
###############

ages = 10**(np.arange(nage)*dlogt + logt0)/1e9
#ages = [1,10]
metals = [1,5]
ncomp = len(metals) * len(ages)

masstheta = {'i0':0, 'N': ncomp, 'lower':0., 'dtype':'<f8', 
             'prior_function':priors.zeros,
             'prior_gradient_function':priors.zeros,
             'prior_args':{}}

dusttheta = {'i0':ncomp, 'N': 1, 'lower':0., 'upper': 2, 'dtype':'<f8', 
             'prior_function':priors.zeros,
             'prior_gradient_function':priors.zeros,
             'prior_args':{}}

theta_desc = {'mass':masstheta}#, 'dust2':dusttheta}
model = cspmodel.CompositeModel(theta_desc, ages, metals)
ndim = 0
for p, d in model.theta_desc.iteritems():
    ndim += d['N']
rp['ndim'] = ndim

sps_fixed_params = {'sfh':0, 'imf_type': 2, 'dust_type': 1, 'imf3': rp['imf3']}
sps = StellarPopulation(compute_vega_mags = False)
for k, v in sps_fixed_params.iteritems():
    sps.params[k] = v
model.sps_fixed_params = sps_fixed_params

def lnprobfn(theta, mod):
    """wrapper on the model instance method, defined here
    globally to enable multiprocessing"""
    return mod.lnprob(theta, sps = sps)

###############
# MOCK
################
obs = {}
tweight = np.log(model.ssp['tage']/model.ssp['tage'].min()) + 0.1
mock_theta = np.array( tweight.tolist() )
mock_spec, mock_phot, mock_other = model.model(mock_theta)
obs['spectrum'] = mock_spec * (1 + np.random.normal(0,1./snr, len(model.sps.wavelengths)))
obs['unc'] = mock_spec/snr
obs['mask'] = ((model.sps.wavelengths > 3e3) & (model.sps.wavelengths < 10e3))
if model.filters is not None:
    obs['filters'] = filters
    obs['maggies'] = mock_phot
    obs['mags_unc'] = np.zeros_like(obs['maggies'])+0.05
else:
    obs['filters'] = None

model.add_obs(obs)
model.ndof = len(model.obs['wavelength']) + len(model.obs['mags'])
model.verbose = False

###############
# INITIALIZE
################
def chi2(theta):
    """A sort of chi2 function that allows for maximization of lnP"""
    return -model.lnprob(theta, sps =sps)

def chi2_grad(theta):
    """A sort of chi2 gradient function that allows for maximization of lnP"""
    return -model.lnprob_grad(theta, sps=sps)

bounds = len(mock_theta) * [(0, None)]
bfgs_opt = {'ftol':1e-10, 'gtol':1e-8, 'maxfev':1e4}
x0 = mock_theta * np.clip(np.random.normal(5.0,5.0, len(mock_theta)),0,np.infty)
bfgs_guess = minimize(chi2, x0, method='L-BFGS-B', jac = chi2_grad,
                    options=bfgs_opt, bounds = bounds)

sys.exit()
###############
#SAMPLE
###############

ndim = len(mock_theta)
nsampler = 4
nsegmax = 20
iterations = 50
length = 50
tt = np.zeros([nsegmax,ndim, nsampler])
Rhat = np.zeros([nsegmax,ndim])
alleps = np.zeros([nsegmax,nsampler])

maxshrink = 2.0
target = 0.8
badness = np.zeros([nsegmax,ndim])
allchain = np.zeros([ndim, nsampler, iterations])


#hmc
tstart = time.time()
#hinitial = np.random.uniform(0,2 * tweight.mean(),ndim)
#hinitial[-1] = np.random.uniform(0,2)
samplers = [hmc.BasicHMC() for k in range(nsampler)]
eps = np.zeros(nsampler)
#initial conditions and calulate initial epsilons
for i,s in enumerate(samplers):
    hinitial = (np.random.normal(1,0.1, ndim) ) * bfgs_guess.x
    hinitial.shape
    pos, prob, thiseps = s.sample(hinitial, model, iterations = 10, epsilon = None,
                                  length = length, store_trajectories = False, nadapt = 0)
    eps[i] = thiseps
    print(i)

print('burn-in')
for k in range(nsegmax):
    allchain *= 0.
    #advance each sampler after adjusting step size
    for i,s in enumerate(samplers):
        afrac = s.accepted.sum()*1.0/s.chain.shape[0]
        da = afrac - target
        if da > 0:
            shrink = 1 + da/(1-target)*(maxshrink-1)
        elif da <= 0:
            shrink = 1 + (maxshrink-1) *  (-da)/target
            shrink = 1/shrink

        if afrac > 0.9:
            shrink = 2.0
        elif afrac < 0.6:
            shrink = 1/2.0
        else:
            shrink = 1.0
        
        eps[i] *= shrink
        print('shrink = {0}'.format(shrink))
        pos, prob, thiseps = s.sample(s.chain[-1,:], model, iterations = iterations,
                                  epsilon = eps[i], length = length, store_trajectories = False, nadapt = 0)
        eps[i] = thiseps #this should not have actually changed during the sampling
        alleps[k,i] = thiseps
        #calculate the correlation lengths for each parameter
        for j in range(ndim):
            try:
                tt[k,j, i], mean, sigma = acor.acor(s.chain[:,j])
            except (RuntimeError):
            #acor sometimes bails
                tt[k,j, i] = -1

        allchain[:,i,:] =  s.chain.T
    #print('tau(lnp) = {0}, tau(theta) = {1}'.format(tau, tt[k,:]))
    #calculate the gelman rubin statistic
    Rhat[k,:] = grR(allchain, nsplit = 2)

sys.exit()

        
# Production run
afrac = hsampler.accepted.sum()*1.0/hsampler.chain.shape[0]
if afrac < 0.5:
    eps = eps/shrink

hpos, hprob, eps = hsampler.sample(hpos, model, iterations = 500,
                                   epsilon = eps, length = 50,
                                   store_trajectories = False, nadapt = 0)

try:
    tau, mean, sigma = acor.acor(hsampler.lnprob)
except:
    pass



