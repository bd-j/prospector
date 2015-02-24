import time, sys, os
import numpy as np

class LikelihoodFunction(object):
    
    def __init__(self, obs=None, model=None, lnspec=True):
        self.obs = obs
        self.model = model
        self.lnspec = lnspec
        
    def lnlike_spec(self, spec_mu, obs=None, gp=None):
        """Calculate the likelihood of the spectroscopic data given
        the spectroscopic model.  Allows for the use of a gaussian
        process covariance matrix for multiplicative residuals.

        :param spec_mu:
            The mean model spectrum, in linear or logarithmic units, including
            e.g. calibration and sky emission.
            
        :param obs: (optional)
            A dictionary of the observational data, including the keys
              *``spectrum`` a numpy array of the observed spectrum, in
               linear or logarithmic units (same as ``spec_mu``).
              *``unc`` the uncertainty of same length as ``spectrum``
              *``mask`` optional boolean array of same length as
               ``spectrum``
              *``wavelength`` if using a GP, the metric that is used
               in the kernel generation, of same length as
               ``spectrum`` and typically giving the wavelength array.

            If not supplied then the obs dictionary given at
            initialization will be used.

        :param gp: (optional)
            A Gaussian process object with the methods `compute` and
            `lnlikelihood`.  If gp is supplied, the `wavelength` entry
            in the obs dictionary must exist
            
        :returns lnlikelhood:
            The natural logarithm of the likelihood of the data given
            the mean model spectrum.
        """
        
        if obs is None:
            obs = self.obs
        if obs['spectrum'] is None:
            return 0.0
    
        mask = obs.get('mask', np.ones( len(obs['spectrum']), dtype= bool))
        delta = (obs['spectrum'] - spec_mu)[mask]
        if gp is not None:
            gp.compute(obs['wavelength'][mask], obs['unc'][mask])
            return gp.lnlikelihood(delta)
        
        var = obs['unc'][mask]**2
        lnp = -0.5*( (delta**2/var).sum() +
                     np.log(var).sum() )
        return lnp
        


    def lnlike_phot(self, phot_mu, obs=None, gp=None):
        """Calculate the likelihood of the photometric data given the
        spectroscopic model.  Allows for the use of a gaussian process
        covariance matrix, though for *additive* residuals.  In
        general, one would only want to mess with the diagonal in
        linear space

        :param phot_mu:
            The mean model sed, in linear flux units (i.e. maggies),
            including e.g. calibration and sky emission and nebular
            emission.
            
        :param obs: (optional)
            A dictionary of the observational data, including the keys
              *``maggies`` a numpy array of the observed SED, in
               linear flux units
              *``maggies_unc`` the uncertainty of same length as
               ``maggies``
              *``phot_mask`` optional boolean array of same length as
               ``maggies``
              *``filter_metric`` if using a GP, the metric that is
               used in the kernel generation, of same length as
               ``maggies`` and typically giving the wavelengths or
               some ordering.

           If not supplied then the obs dictionary given at
           initialization will be used.  This should really all be in
           kwargs.

        :param gp: (optional)
            A Gaussian process object with the methods ``compute()`` and
            ``lnlikelihood()``.
            
        :returns lnlikelhood:
            The natural logarithm of the likelihood of the data given
            the mean model spectrum.
        """

        if obs is None:
            obs = self.obs
        if obs['maggies'] is None:
            return 0.0
    
        mask = obs.get('phot_mask', np.ones( len(obs['maggies']), dtype= bool))
        delta = (obs['maggies'] - phot_mu)[mask]
        if gp is not None:
            gp.flux = obs['maggies'][mask]
            gp.compute(obs['filter_metric'][mask], obs['maggies_unc'][mask])
            return gp.lnlikelihood(delta)
        
        var = (obs['maggies_unc'][mask])**2
        lnp = -0.5*( (delta**2/var).sum() +
                     np.log(var).sum() )
        return lnp
    
    def ln_prior_prob(self, theta, model=None):
        if model is None:
            model = self.model
        return model.prior_product(theta)

    def lnpostfn(self, theta, model=None, obs=None,
               sps=None, gp=None, **extras):
        """A specific implementation of a posterior probability
        function, as an example.
        """
        if model is None:
            model = self.model
        if obs is None:
            obs = self.obs
            
        # Get the prior
        lnp_prior= self.ln_prior_prob(theta)
        if np.isfinite(lnp_prior):
            # Get the mean model and GP Kernel
            spec, phot, x = model.mean_model(theta, sps = sps)
            log_mu = np.log(spec) + mod.calibration(theta)
            s, a, l = (mod.params['gp_jitter'], mod.params['gp_amplitude'],
                       mod.params['gp_length'])
            gp.kernel[:] = np.log(np.array([s[0],a[0]**2,l[0]**2]))
            # Get the likelihoods
            lnp_spec = self.lnlike_spec_log(np.exp(log_mu), obs=obs, gp=gp)
            lnp_phot = self.lnlike_phot(phot, obs=obs, gp=None)
            # Return the sum
            return lnp_prior + lnp_phot + lnp_spec
        else:
            return -np.infty
