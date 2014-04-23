import numpy as np

from observate import getSED
import attenuation
from scipy.interpolate import griddata

class Model(object):

    verbose = False
    
    def add_obs(self, obs):
        self.filters = obs['filters']
        self.obs = obs

    def lnprob(self, theta, **extras):
        """
        Given a theta vector, return the ln of the posterior probability.
        """

        # Determine prior probability for this theta
        lnp_prior = self.prior_product(theta)
        if self.verbose:
            print('theta = {0}'.format(theta))
            print('lnp_prior = {0}'.format(lnp_prior))

        # Get likelihood if prior is finite   
        if np.isfinite(lnp_prior):
            # Get the spectrum for this theta
            spec, phot, other = self.model(theta, **extras)

            # Spectroscopic term
            if self.obs['spectrum'] is not None:
                # Shortcuts for observational uncertainties
                #total_var_spec =  (self.obs['unc'] + self.params.get('jitter',0) * self.obs['spectrum'])**2
                total_var_spec =  (self.obs['unc']**2 + (self.params.get('jitter',0) * self.obs['spectrum'])**2)
                mask = self.obs['mask']
                lnp_spec = -0.5* ((spec - self.obs['spectrum'])**2 / total_var_spec)[mask].sum()
                #lnp_spec = -0.5 * (((np.log(spec) - np.log(self.obs['spectrum']))**2) / (total_var_spec/spec**2) )[mask].sum()

                r = (self.obs['spectrum'] - spec)
                sigma = total_var_spec * self.params.get('jitter',0)
                
                # Jitter term
                if self.params.get('jitter',0) != 0:
                    lnp_spec += log(2*pi*total_var_spec[mask]).sum()
            else:
                lnp_spec = 0
                
            # Photometry term
            if self.obs['filters'] is not None:
                maggies = 10**(-0.4 * self.obs['mags'])
                phot_var = (maggies * self.obs['mags_unc']/1.086)**2 
                lnp_phot =  -0.5*( (phot - maggies)**2 / phot_var ).sum()
            else:
                lnp_phot = 0

            #print out
            if self.verbose:
                print('lnp = {0}'.format(lnp_spec + lnp_phot + lnp_prior))
            return lnp_spec + lnp_phot + lnp_prior
        else:
            return -np.infty
        
    def prior_product(self, theta):
        """
        Return a scalar which is the ln of the prioduct of the prior
        probabilities for each element of theta.  Requires that the prior 
        functions are defined in the theta descriptor.
        """
        lnp_prior = 0
        for p in self.theta_desc.keys():
            start, stop = self.theta_desc[p]['i0'], self.theta_desc[p]['i0'] + self.theta_desc[p]['N']
            lnp_prior += np.sum(self.theta_desc[p]['prior_function'](theta[start:stop],
                                                                     **self.theta_desc[p]['prior_args']))
        return lnp_prior

    def lnp_prior_grad(self, theta):
        """
        Return a vector of gradients in the prior probability.  Requires 
        that functions giving the gradients are given in the theta descriptor.
        """
        lnp_prior_grad = np.zeros_like(theta)
        for p in self.theta_desc.keys():
            start, stop = self.theta_desc[p]['i0'], self.theta_desc[p]['i0'] + self.theta_desc[p]['N']
            lnp_prior_grad[start:stop] = self.theta_desc[p]['prior_gradient_function'](theta[start:stop], **self.theta_desc[p]['prior_args'])
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
            for p,pdict in self.theta_desc.iteritems():
                start, end = pdict['i0'], pdict['i0'] + pdict['N']
                if 'upper' in pdict.keys():
                    above = theta[start:end] > pdict['upper']
                    oob = oob or np.any(above)
                    theta[start:end][above] = 2 * pdict['upper'] - theta[start:end][above]
                    sign[start:end][above] *= -1
                if 'lower' in pdict.keys():
                    below = theta[start:end] < pdict['lower']
                    oob = oob or np.any(below)
                    theta[start:end][below] = 2 * pdict['lower'] - theta[start:end][below]
                    sign[start:end][below] *= -1
        if self.verbose: print('theta out={0}'.format(theta))            
        return theta, sign, oob


