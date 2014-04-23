import numpy as np

from observate import getSED
import attenuation
from simplemodel import Model


class CompositeModel(Model):

    def __init__(self, theta_desc, tage = [1.0], zmet = [-0.5], sps = None, dust_curve = attenuation.calzetti):
        self.verbose = False
        self.theta_desc = theta_desc
        self.jitter = 0
        self.dust_curve = dust_curve
        self.params = {'tage':tage,'zmet':zmet,'dust_curve':dust_curve}

    def set_parameters(self, theta):
        """Propagate theta into the model parameters"""
        
        self.params['imf_type'] = 2
        #self.params[] = 
        for p in self.theta_desc:
            start, end = self.theta_desc[p]['i0'], self.theta_desc[p]['i0'] + self.theta_desc[p]['N']
            self.params[p] = np.array(theta[start:end])

    def model(self, theta, sps = None):
        """
        Given a theta vector, generate a spectrum, photometry, and any extras (e.g. stellar mass).
        """
        if sps is None:
            sps = self.sps
        self.set_parameters(theta)
        spec, phot, extras = sps.get_spectrum(self.params, self.obs['wavelength'], self.obs['filters'])
        spec *= self.calibration()
        return spec, phot, extras
            
 
    def lnprob_grad(self, theta, sps = None):
        """
        Given theta, return a vector of gradients in lnP along the theta directions.
        Theta can *only* include amplitudes in this formulation, though potentially dust
        and calibration parameters might be added.
        """
        if sps is None:
            sps = self.sps

        status = ((len(theta) == self.theta_desc['mass']['N']) and (self.theta_desc.keys() == ['mass']))
        if status is False:
            raise ValueError('You are attempting to use gradients for parameters where they are not calculated!!!')
        
        self.set_parameters(theta)
        comp_spec, comp_phot, comp_extra = sps.get_components(self.params, self.obs['wavelength'], self.obs['filters'])
        spec = (comp_spec  * self.params['mass'][:,None]).sum(axis = 0) * self.calibration()
        phot = (comp_phot  * self.params['mass'][:,None]).sum(axis = 0)
        
        # Spectroscopy terms
        if self.obs['spectrum'] is not None:
            #shortcuts for observational uncertainties
            total_var_spec =  (self.obs['unc'] + self.params.get('jitter',0) * self.obs['spectrum'])**2
            mask = self.obs['mask']
            wave = self.obs['wavelength']
            delta_spec = -(spec - self.obs['spectrum'])/total_var_spec 
            gradp_spec = {} 
            
            gradp_spec['mass'] = (delta_spec[None,:] * self.calibration() * comp_spec )[:,mask].sum(axis = 1)

            #jitter term
            gradp_jitter = {}
            if self.params.get('jitter',0.) != 0:
                raise ValueError('gradients in jitter term not written')
            
        # Photometry terms
        gradp_phot = {}
        if self.obs['filters'] is not None:
            phot_var = (10**(-0.4 * self.obs['mags'])*self.obs['mags_unc']/1.086)**2 
            #partial for the squared term
            phot_part = -np.atleast_1d((phot - 10**(-0.4 * self.obs['mags'])) / phot_var)
            gradp_phot['mass'] = (phot_part[None,:] * comp_phot).sum(axis = 1)

        # Sum the gradients
        all_grads = [gradp_spec, gradp_phot, gradp_jitter]
        #start with the gradients in the priors.  defaults to 0 if no gradients defined
        gradp = self.lnp_prior_grad(theta)
        for p in self.theta_desc.keys():
            start, stop = self.theta_desc[p]['i0'], self.theta_desc[p]['i0'] + self.theta_desc[p]['N']
            for g in all_grads:
                gradp[start:stop] += g.get(p, 0)

        return gradp

    def calibration(self):
        x = self.obs['wavelength']/self.params['pivot_wave'] - 1.0
        poly = np.zeros_like(x)
        powers = np.arange( len(self.params['poly_coeffs']) ) + 1
        poly = (x[None,:] ** powers[:,None] * self.params['poly_coeffs'][:,None]).sum(axis = 0)
        
        return (1.0 + poly) * self.params['spec_norm']
