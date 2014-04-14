import numpy as np

from observate import getSED
import attenuation
from scipy.interpolate import griddata
from simplemodel import Model


class CompositeModel(Model):

    def __init__(self, theta_desc, ages, metals, filters = None):
        self.verbose = False
        self.filters = filters
        self.theta_desc = theta_desc
        #self.sps = sps
        #self.sps.params['sfh'] = 0
        #self.sps.params['imf_type'] = 2
        self.jitter = 0
        self.ssp_gen(ages, metals)
        self.dust_law = attenuation.calzetti(self.sps.wavelengths)
        self.compsp_pars_from_theta = ['mass']#, 'dust2']
        self.ssp_pars_from_theta = [] #['imf1','f_bhb'] #etc...
        self.cal_pars_from_theta = []#['poly_coeffs', 'jitter'] #etc....

    def set_parameters(self, theta):
        """Propogate theta into the model parameters"""

        #Parameters for compsp - these are propogated into the parameters for each component
        for p in self.compsp_pars_from_theta:
            start, end = self.theta_desc[p]['i0'], self.theta_desc[p]['i0'] + self.theta_desc[p]['N']
            self.ssp[p] = np.array(theta[start:end])
        #Parameters for the calibration model.  these are stored in cal_pars
        self.cal_pars = {}
        for p in self.cal_pars_from_theta:
            start, end = self.theta_desc[p]['i0'], self.theta_desc[p]['i0'] + self.theta_desc[p]['N']
            self.cal_pars[p] = np.array(theta[start:end])
        #parameters for the SSP generation.  these are propogated through to the sps object and should be FSPS params
        for p in self.ssp_pars_from_theta:
            start, end = self.theta_desc[p]['i0'], self.theta_desc[p]['i0'] + self.theta_desc[p]['N']
            self.sps.params[p] = theta[start:end] #this should increase dirtiness and force a regeneration of the SSPs

    def model(self, theta, sps = None, filters = None):
        """Given a theta vector, generate a spectrum, photometry, and any extras."""
        self.set_parameters(theta, sps = sps)
        if sps.params.dirtiness > 0:
            self.ssp_gen(self.metals, self.ages, sps = sps)
        spec, phot, mass = self.compsp()
        z1 = 1.0 #need to figure out how do deal with such things

        #decalibrated spectrum
        try:
            outspec = griddata(self.ssp_wave * z1, spec/z1, self.obs['wavelength']) * self.calibration() 
        except (KeyError):
            outspec = spec
        #photometry
        if self.filters is not None:
            filters = self.filters #Bad pattern
        if filters is not None:
            phot = 10.**(-0.4 * getSED(self.ssp_wave * z1, spec / z1 * to_cgs, filters))
        else:
            phot = None
        others = mass
        return outspec, phot, others


    def ssp_gen(self, metals, ages, sps = None):
        """use py-fsps to build the SSP dictionary/structured array"""
        self.build_array(ages, metals, ssp.wavelengths)
        # Loop over age and metallicity, getting the spectrum, current mass,
        #  and photometry of each SSP, and store in the record array
        i = 0
        for iz, met in enumerate(metals):
            for it, age  in enumerate(ages):
                self.ssp[i]['tage'] = age
                self.ssp[i]['zmet'] = np.abs(met - np.arange(1,5.01,1)).argmin()
                wave, spec = sps.get_spectrum(peraa = True, tage =age, zmet =met)
                self.ssp[i]['spectrum'][:] = spec
                if self.filters is not None:
                    self.ssp[i]['phot'] = 10**(-0.4 * getSED(wave, spec, self.filters))
                self.ssp[i]['cur_mass'] = sps.stellar_mass
                i += 1

    
    def compsp(self, masses = None, dust = None, dust_law = None):
        """Combine SSPs into a total spectrum including dust attenuation (and eventually
        velocity dispersion.  Returns the spectrum, photometry and extras from the combined SSPs"""

        if masses is None:
            masses = self.ssp['mass']
        if dust is  None:
            dust = self.ssp['dust2']
        if dust_law is None:
            dust_law = self.dust_law
            
        object_spec = ( masses[:,None] * self.ssp['spectrum'] *
                       np.exp(-dust_law[None,:] * dust[:,None]) ).sum(axis = 0)
        # Below DOES NOT INCLUDE DUST!!! DO NOT USE!!!!
        object_phot = ( masses[:,None] * self.ssp['phot'] ).sum(axis = 0)
        object_mass = (masses * self.ssp['cur_mass']).sum()

        return object_spec, object_phot, object_mass

    def build_array(self, ages, metals, wave):
        """Build a numpy record array containing the spectrum and info of the components"""
        nssp = len(ages)*len(metals)
        names = ['mass', 'tage', 'zmet', 'spectrum', 'phot', 'cur_mass', 'dust2']
        types = len(names) * ['<f8']
        shapes = len(names) * [1]
        if self.filters is not None:
            shapes[4] = len(self.filters)
        shapes[3] = len(wave)
        dt = np.dtype(zip(names,types, shapes))
        self.ssp = np.zeros(nssp, dtype = dt)
        self.ssp_wave = wave

    def lnprob_grad(self, theta, sps = sps):
        """Given theta, return a vector of gradients in lnP along the theta directions."""
        #shortcuts for observational uncertainties
        total_var_spec =  (self.obs['unc'] + self.jitter * self.obs['spectrum'])**2
        mask = self.obs['mask']
        wave = sps.wavelengths

        spec, phot, other = self.model(theta)
        comp_spec = self.ssp['spectrum'] * np.exp(-self.dust_law[None,:] * self.ssp['dust2'][:, None])

        #Spectroscopy terms
        gradp_spec = {} 
        spec_part = -(spec - self.obs['spectrum'])/total_var_spec 
        gradp_spec['mass'] = (spec_part[None,:] * comp_spec )[:,mask].sum(axis = 1)
        gradp_spec['dust2'] = -(spec_part[None,:] * comp_spec * 
                                self.ssp['mass'][:,None] * self.dust_law[None,:] )[:,mask].sum(axis = 1)
        #Hack for a single dust value
        gradp_spec['dust2'] = gradp_spec['dust2'].sum()
        #print(gradp_spec['dust2'].shape,gradp_spec['mass'].shape)
        #Jitter terms
        gradp_jitter = {}
        if self.jitter is not 0:
            raise ValueError('gradients in jitter term not written')
            
        #photometry terms
        gradp_phot = {}
        if self.filters is not None:
            phot_var = (self.obs['maggies']*self.obs['mags_unc']/1.086)**2 
            #partial for the squared term
            phot_part = -np.atleast_1d((phot - self.obs['maggies']) / phot_var)
            #photometry of each attenuated component
            
            comp_phot = np.atleast_2d(getSED(wave, comp_spec, self.obs['filters']))
            dust_phot = np.atleast_2d(getSED(wave, 
                                             self.ssp['mass'][:,None] *comp_spec * self.dust_law[None,:], 
                                             obs['filters']))

            #print(phot_part.shape, comp_phot.shape)
            gradp_phot['mass'] = (phot_part[None,:] * comp_phot).sum(axis = 1)
            gradp_phot['dust2'] =  -(phot_part * dust_phot).sum(axis = 1)
            #Hack for a single dust value
            gradp_phot['dust2'] = gradp_phot['dust2'].sum()

        # Sum the gradients
        all_grads = [gradp_spec, gradp_phot, gradp_jitter]
        #start with the gradients in the priors.  defaults to 0 if no gradients defined
        gradp = self.lnp_prior_grad(theta)
        for p in self.theta_desc.keys():
            start, stop = self.theta_desc[p]['i0'], self.theta_desc[p]['i0'] + self.theta_desc[p]['N']
            for g in all_grads:
                gradp[start:stop] += g.get(p, 0)

        return gradp


    def update_sps(self, pardict):
        #set the ssp parameters
        for p, v in pardict.iteritems():
            try:
                self.sps.params[p] = v
            except:
                pass

