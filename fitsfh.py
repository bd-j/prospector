import numpy as np
import emcee
import time

#Objects to do the fitting of the full spectrum.  The superclass
#includes a wrapper on emcee to handle the actual sampling, while subclassses
#for different situations should include a lnprob function which actually
#constructs the model for a given set of parameters.

# each object is initialized with a parameter dictionary and an 'sps' object
# that containes the stellar population synthesis model (i.e. python-FSPS)

class SFHfitter(object):
    isoc_ages = 10**np.arange(6.6, 10.1, 0.05)

    def __init__(self, rp, sps):
        self.rp = rp
        self.wavelengths = sps.wavelengths
        self.sps = sps

    def sample(self, walker_factor = 50, nburn = 10, nsteps = 10, nthreads = 1, **extras):
        """Wrapper around emcee that handles burn-in, etc"""
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

class MixedZ(SFHfitter):

    def spec_model(theta):
        """given parameter vector, return a model of the spectrum"""
        


class FixedZ(SFHfitter):
    """Keep the model stellar metallicity fixed"""
    
    def lnprob(self, theta, data, err, mask):
        """Given SFR in each bin (theta), the data, and the error estimate, return lnP"""
        #prior bounds test.  all components must be non-negative
        #prior probabilities could also be introduced here via a priors blob in the call?
        ptest = theta >= 0
        # theta should actaully be sfr**2 to avoid this discontinuity....?
        if (False in ptest):
            #set lnp to -infty if some component is negative
            return -np.infty
        else:  
            sfrs = theta #[:nbin] #not necessary
            model_flux = (sfrs * self.spec_array.T).sum(axis = 1)
            delta = (model_flux - data)/err*mask
            return -0.5*np.sum(delta**2)


    def load_spectra(self, age_bins):
        """Build the basis spectra using python-FSPS.  This will compute and
        store the stellar mass and spectrum for each 'time bin' (or rather
        collection of SSPs).  Should be rewritten to do this more precisely. """
        self.age_bins = age_bins
        nbin = len(age_bins)
        self.rp['nbin'] = nbin
        self.spec_array = np.zeros( [nbin,self.wavelengths.shape[0]] )
        self.mstar_array = np.zeros( nbin )
        
        for j,abin in enumerate(age_bins):
            ## spectra are normalized to an SFR of one for each component
            # This should be fixed to actually do the convolution of SSPs for a given SFH time bin?
            # no, should just us SSPs as is.... then conversion to an actual SFH is handled elsewhere
            ## Also, just use the default SSP time points

            #clunky bit to sum SSPs into a given time bin.
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
                    mformed = 10 #for some reason FSPS ssps start with log_mass = 1 instead of 0
                    norm = dt/mformed #for an sfr of 1
                    self.mstar_array[j] += (norm * mstar)
                    self.spec_array[j,:] += (norm * spec)

    def fit_sfh(self):
        self.sfr_samples = self.sample(**self.rp)
        
    def mock_spectrum(self, rp = None):
        """Mock a spectrum for a given SFH and SNR"""
        if rp is None:
            rp = self.rp
        mock_spec = (self.mock_sfh * self.spec_array.T).sum(axis = 1)
        self.mask = np.where((self.wavelengths > rp['wlo']) & (self.wavelengths < rp['whi']), 1, 0) 
        noised_mock = mock_spec * (1+np.random.normal(0,1,self.wavelengths.shape[0])/rp['snr'])
        self.data = noised_mock
        self.err = mock_spec/rp['snr']

    def recovery_stats(self):
        """return standard deviation and mean of SFR_recovered/SFR_input"""
        delta =  np.ma.masked_invalid( np.log10(self.sfr_samples / self.mock_sfh) )
        return np.append(delta.std(axis = 0), delta.std()), np.append(delta.mean(axis = 0), delta.mean())
