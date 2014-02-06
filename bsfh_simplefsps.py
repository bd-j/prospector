import time
import numpy as np

import fsps
import simplemodel

import hmc
import emcee

import priors

snr = 100.
nage = 7
logt0 = 6.75
dlogt = 0.5
filters = None
if filters: filters = observate.load_filters(filters)


#SET UP
ages = 10**(np.arange(nage)*dlogt + logt0)/1e9
#ages = [1,10]
metals = [2]
ncomp = len(metals) * len(ages)

masstheta = {'i0':0, 'N': ncomp, 'lower':0., 'dtype':'<f8', 
             'prior_function':priors.zeros, 'prior_gradient_function':priors.zeros}

dusttheta = {'i0':ncomp, 'N': 1, 'lower':0., 'upper': 2, 'dtype':'<f8', 
             'prior_function':priors.zeros, 'prior_gradient_function':priors.zeros}


theta_desc = {'mass':masstheta, 'dust2':dusttheta}
sps = fsps.StellarPopulation()
model = simplemodel.SimpleFSPSModel(theta_desc, sps, ages, metals, filters = filters)
model.verbose = True

#MOCK
obs = {}
tweight = np.log(model.ssp['tage']/model.ssp['tage'].min()) + 0.1
mock_theta = np.array( tweight.tolist()+ [0.5])
mock_spec, mock_phot, mock_other = model.model(mock_theta)
obs['spectrum'] = mock_spec * (1 + np.random.normal(0,1./snr, len(model.sps.wavelengths)))
obs['unc'] = mock_spec/snr
obs['mask'] = ((model.sps.wavelengths > 2e3) & (model.sps.wavelengths < 20e3))
if model.filters is not None:
    obs['filters'] = filters
    obs['maggies'] = mock_phot
    obs['mags_unc'] = np.zeros_like(obs['maggies'])+0.05
else:
    obs['filters'] = None

model.add_obs(obs)
model.verbose = False
#SAMPLE
iterations = 20
ndim = len(mock_theta)

#raise ValueError('wait')

#hmc
tstart = time.time()
#hinitial = np.random.uniform(0,2 * tweight.mean(),ndim)
#hinitial[-1] = np.random.uniform(0,2)

hinitial = (np.random.normal(1,0.3, ndim) ) * mock_theta
hsampler = hmc.BasicHMC()
hpos, hprob, eps = hsampler.sample(hinitial, model, iterations = 100, epsilon = None, length = 100, store_trajectories = False, nadapt = 0)
#initial = hpos
#hpos, hprob, eps = hsampler.sample(initial, model, iterations = 500, epsilon = None, length = 50, store_trajectories = False)

x =none

hinitial2 = (np.random.normal(2,0.3, ndim) ) * mock_theta
hsampler2 = hmc.BasicHMC()
hpos, hprob, eps = hsampler2.sample(hinitial2, model, iterations = 100, epsilon = None, length = 1000, store_trajectories = False)


hdur = time.time() - tstart
print('emcee:')
#emcee
model.theta_desc['mass']['prior_function'] = priors.positive
model.theta_desc['dust2']['prior_function'] = priors.positive
nwalkers = ndim * 10
initial = np.random.uniform(0,2 * tweight.mean(),[nwalkers, len(mock_theta)])
initial[:,-1] = np.random.uniform(0,2, nwalkers)
tstart = time.time()
esampler = emcee.EnsembleSampler(nwalkers, ndim, model.lnprob, threads = 1)
epos, eprob, state = esampler.run_mcmc(initial, 100)
edur = time.time() - tstart
