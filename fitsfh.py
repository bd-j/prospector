import numpy as np
import emcee
import time

class SFHfitter(object):
    isoc_ages = 10**np.arange(6.6,10.1,0.05)

    def __init__(self, rp, sps):
        self.rp = rp
        self.wavelengths = sps.wavelengths
        self.sps = sps

    def sample(self, walker_factor = 50, nburn = 10, nsteps = 10, nthreads = 1, **extras):
        ndim = self.rp['nbin']
        nwalkers = ndim * walker_factor

        #make a guess based on the average sfr.  in practice this
        #can be worked out pretty well a priori from real data even
        initial = np.random.uniform(size = (nwalkers,ndim)) * self.mock_sfh.mean() * 5
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, threads=nthreads, args=[self.data,self.err,self.mask])
        tstart = time.time()
        pos,prob,state = sampler.run_mcmc(initial, nburn)
        sampler.reset()
        sampler.run_mcmc(np.array(pos), nsteps, rstate0=state, thin = 5)

        self.rp['tsample'] = time.time()-tstart
        self.rp['maf'] = np.mean(sampler.acceptance_fraction)
        self.rp['nsample'] = sampler.flatchain.shape[0]
        
        print('Sampling done in {0}s'.format( self.rp['tsample']) )
        print("Mean acceptance fraction: {0:.3f}".format(self.rp['maf']))
        return sampler.flatchain


class FixedZ(SFHfitter):

        def lnprob(self, theta, data, err, mask):
        #prior bounds test.  all components must be non-negative
        #prior probabilities could also be introduced here via a priors blob in the call?
        ptest = theta >= 0

        if (False in ptest):
            #set lnp to -infty if some component is negative
            return -np.infty
        else:  
            sfrs = theta #[:nbin] #not necessary
            model_flux = (sfrs * self.spec_array.T).sum(axis = 1)
            delta = (model_flux - data)/err*mask
            return -0.5*np.sum(delta**2)


    def load_spectra(self, age_bins):
        """Build the basis spectra"""
        self.age_bins = age_bins
        nbin = len(age_bins)
        self.rp['nbin'] = nbin
        self.spec_array = np.zeros( [nbin,self.wavelengths.shape[0]] )
        self.mstar_array = np.zeros( nbin )
        
        for j,abin in enumerate(age_bins):
            ## spectra are normalized to an SFR of one for each component
            ## This should be fixed to actually do the convolution
            ## Also, just use the default SSP time points
            for i,iage in enumerate(self.isoc_ages):
                if (iage > abin['start']) & (iage < abin['end']):
                    if i == 0:
                        dt = (iage - abin['start'])+ np.min([self.isoc_ages[i+1]-iage, abin['end']-iage])
                    elif i == (len(self.isoc_ages)-1):
                        dt  = np.min([iage-self.isoc_ages[i-1], iage-abin['start']]) + (abin['end']-iage)
                    else:
                        dt = ( np.min([iage-self.isoc_ages[i-1], iage-abin['start']]) +
                               np.min([self.isoc_ages[i+1]-iage, abin['end']-iage]) )
                        
                    wave, spec = self.sps.get_spectrum(zmet = self.rp['met_index'], tage = iage*1e-9, peraa=True)
                    mstar = 10**self.sps.log_mass
                    mformed = 10 #for some reason fsps ssps start with log_mass = 1
                    norm = dt/mformed #for an sfr of 1
                    self.mstar_array[j] += (norm * mstar)
                    self.spec_array[j,:] += (norm * spec)


    def fit_sfh(self):
        self.sfr_samples = self.sample(**self.rp)
        

    def mock_spectrum(self, rp = None):
        if rp is None:
            rp = self.rp
        mock_spec = (self.mock_sfh * self.spec_array.T).sum(axis = 1)
        self.mask = np.where((self.wavelengths > rp['wlo']) & (self.wavelengths < rp['whi']), 1, 0) 
        noised_mock = mock_spec * (1+np.random.normal(0,1,self.wavelengths.shape[0])/rp['snr'])
        self.data = noised_mock
        self.err = mock_spec/rp['snr']


    def recovery_stats(self):
        delta =  np.ma.masked_invalid( np.log10(self.sfr_samples / self.mock_sfh) )
        return np.append(delta.std(axis = 0), delta.std()), np.append(delta.mean(axis = 0), delta.mean())
