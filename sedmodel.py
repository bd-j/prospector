import numpy as np

from observate import getSED
import attenuation

class ThetaParameters(object):

    def __init__(self, theta_desc = None, theta_init = None, **kwargs):
        self.theta_desc = theta_desc

        self.params = {}
        if theta_init:
            self.set_parameters(theta_init)
        for k,v in kwargs.iteritems():
            self.params[k] = np.array(v)

    @property
    def ndim(self):
        #should probably cache this
        ndim = 0
        for p, v in self.theta_desc.iteritems():
            ndim += v['N']
        return ndim
            
    def set_parameters(self, theta):
        """Propagate theta into the model parameters"""
        assert len(theta) == self.ndim
        for p, v in self.theta_desc.iteritems():
            start, end = v['i0'], v['i0'] + v['N']
            self.params[p] = np.array(theta[start:end])

        
    def prior_product(self, theta):
        """
        Return a scalar which is the ln of the prioduct of the prior
        probabilities for each element of theta.  Requires that the prior 
        functions are defined in the theta descriptor.
        """
        lnp_prior = 0
        for p, v in self.theta_desc.iteritems():
            start, stop = v['i0'], v['i0'] + v['N']
            lnp_prior += np.sum(v['prior_function'](theta[start:stop], **v['prior_args']))
        return lnp_prior

    def lnp_prior_grad(self, theta):
        """
        Return a vector of gradients in the prior probability.  Requires 
        that functions giving the gradients are given in the theta descriptor.
        """
        lnp_prior_grad = np.zeros_like(theta)
        for p, v in self.theta_desc.iteritems():
            start, stop = v['i0'], v['i0'] + v['N']
            lnp_prior_grad[start:stop] = v['prior_gradient_function'](theta[start:stop], **v['prior_args'])
        return lnp_prior_grad

    def check_constrained(self, theta):
        """
        For HMC, check if the trajectory has hit a wall in any parameter.  
        If so, reflect the momentum and update the parameter position in the 
        opposite direction until the parameter is within the bounds. Bounds 
        are specified via the 'upper' and 'lower' keys of the theta descriptor
        """
        oob = True
        sign = np.ones_like(theta)
        if self.verbose: print('theta in={0}'.format(theta))
        while oob:
            oob = False
            for p,v in self.theta_desc.iteritems():
                start, end = v['i0'], v['i0'] + v['N']
                if 'upper' in v.keys():
                    above = theta[start:end] > v['upper']
                    oob = oob or np.any(above)
                    theta[start:end][above] = 2 * v['upper'] - theta[start:end][above]
                    sign[start:end][above] *= -1
                if 'lower' in v.keys():
                    below = theta[start:end] < v['lower']
                    oob = oob or np.any(below)
                    theta[start:end][below] = 2 * v['lower'] - theta[start:end][below]
                    sign[start:end][below] *= -1
        if self.verbose: print('theta out={0}'.format(theta))            
        return theta, sign, oob


class SedModel(ThetaParameters):

    def add_obs(self, obs):
        self.filters = obs['filters']
        self.obs = obs

    def model(self, theta, sps = None, **kwargs):
        """
        Given a theta vector, generate a spectrum, photometry, and any
        extras (e.g. stellar mass).
        """
        
        if sps is None:
            sps = self.sps
        self.set_parameters(theta)
        spec, phot, extras = sps.get_spectrum(self.params, self.obs['wavelength'], self.obs['filters'])
        spec *= self.calibration()
        return spec, phot, extras

    def calibration(self):
        """
        Implements a polynomial calibration model.

        :returns cal:
           a polynomial given by 'spec_norm' * (1 + \Sum_{m=1}^M 'poly_coeffs'[m-1] x**m)
        """
            
        x = self.obs['wavelength']/self.params['pivot_wave'] - 1.0
        poly = np.zeros_like(x)
        powers = np.arange( len(self.params['poly_coeffs']) ) + 1
        poly = (x[None,:] ** powers[:,None] * self.params['poly_coeffs'][:,None]).sum(axis = 0)
        
        return (1.0 + poly) * self.params['spec_norm']

    def lnprob(self, theta, **extras):
        """
        Given a theta vector, return the ln of the posterior probability.
        
        """

        # Determine prior probability for this theta
        lnp_prior = self.prior_product(theta)
        if self.verbose:
            print('theta = {0}'.format(theta))
            print('lnP_prior = {0}'.format(lnp_prior))

        if np.isfinite(lnp_prior):  # Get likelihood if prior is finite   
            spec, phot, other = self.model(theta, **extras)
            
            if self.obs['spectrum'] is not None:  # Spectroscopic term
                jitter = self.params.get('jitter',0)
                residual = (self.obs['spectrum'] - spec)
                total_var_spec =  (self.obs['unc']**2 + (jitter * self.obs['spectrum'])**2)
                mask = self.obs['mask']                
                lnp_spec = -0.5* ( residual**2 / total_var_spec)[mask].sum()
                if jitter != 0:  # Spectroscopic jitter term
                    lnp_spec += log(2*pi*total_var_spec[mask]).sum()
            else:
                lnp_spec = 0
                
            if self.obs['filters'] is not None: # Photometry term
                jitter = self.params.get('phot_jitter',0)
                maggies = 10**(-0.4 * self.obs['mags'])
                phot_var = maggies**2 * ((self.obs['mags_unc']/1.086)**2 + jitter**2)
                lnp_phot =  -0.5*( (phot - maggies)**2 / phot_var ).sum()
                if jitter != 0: # Photometric jitter term
                    lnp_phot += log(2*pi*phot_var).sum()
            else:
                lnp_phot = 0

            if self.verbose:
                print('lnP = {0}'.format(lnp_spec + lnp_phot + lnp_prior))
                
            return lnp_spec + lnp_phot + lnp_prior
        else:
            return -np.infty
  
    def lnprob_grad(self, theta, sps = None):
        """
        Given theta, return a vector of gradients in lnP along the theta directions.
        Theta can *only* include amplitudes in this formulation, though potentially dust
        and calibration parameters might be added.
        """
        if sps is None:
            sps = self.sps

        status = ((len(theta) == self.theta_desc['mass']['N']) and
                  (self.theta_desc.keys() == ['mass']))
        if status is False:
            raise ValueError('You are attempting to use gradients for parameters where they are not calculated!!!')
        
        self.set_parameters(theta)
        comp_spec, comp_phot, comp_extra = sps.get_components(self.params, self.obs['wavelength'], self.obs['filters'])
        cal = self.calibration()
        spec = (comp_spec  * self.params['mass'][:,None]).sum(axis = 0) * cal
        phot = (comp_phot  * self.params['mass'][:,None]).sum(axis = 0)

        gradp_spec = {} # Spectroscopy terms
        if self.obs['spectrum'] is not None: 
            jitter = self.params.get('jitter',0)
            total_var_spec =  (self.obs['unc']**2 + (jitter * self.obs['spectrum'])**2)
            mask = self.obs['mask']
            delta = -(spec - self.obs['spectrum'])/total_var_spec 
            
            gradp_spec['mass'] = (delta[None,:] * cal * comp_spec )[:,mask].sum(axis = 1)
            
        gradp_jitter = {} #jitter terms
        if self.params.get('jitter',0.) != 0: 
            raise ValueError('gradients in jitter term not written')
            
        gradp_phot = {} # Photometry terms
        if self.obs['filters'] is not None: 
            jitter = self.params.get('phot_jitter',0)
            maggies = 10**(-0.4 * self.obs['mags'])
            phot_var = maggies**2 * ((self.obs['mags_unc']/1.086)**2 + jitter**2)
            delta = -np.atleast_1d((phot - maggies) / phot_var)
            
            gradp_phot['mass'] = (delta[None,:] * comp_phot).sum(axis = 1)

        # Sum the gradients
        all_grads = [gradp_spec, gradp_phot, gradp_jitter]
        #start with the gradients in the priors.  defaults to 0 if no gradients defined
        gradp = self.lnp_prior_grad(theta)
        for p in self.theta_desc.keys():
            start, stop = self.theta_desc[p]['i0'], self.theta_desc[p]['i0'] + self.theta_desc[p]['N']
            for g in all_grads:
                gradp[start:stop] += g.get(p, 0)

        return gradp

