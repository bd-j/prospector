"""
testing intermediary product to build flexible NoiseModel objects
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from statsmodels.nonparametric.kernel_density import KDEMultivariate

__all__ = ["Kernel_photsamples", 
           "Uncorrelated_photsamples", "Correlated_photsamples", 
           "KDE_photsamples"] 

class Kernel_photsamples(object):

    def __init__(self, parnames=[], name=''):
        """
        :param parnames:
            A list of names of the kernel params, used to alias the intrinsic
            parameter names.  This way different instances of the same kernel
            can have different parameter names.

        WPBWPB:
            parnames usage: identify the keys in the kwargs dictionary
                in the update method that correspond to the ``kernel_params`` 
                lists in each subclass below
        """
#        if len(parnames) == 0:
#            parnames = self.kernel_params
#        assert len(parnames) == len(self.kernel_params)
#        self.param_alias = dict(zip(self.kernel_params, parnames))
        self.params = {}
        self.name = name
###        self.setup_complete = False

#    def __repr__(self):
#        return '{}({})'.format(self.__class__, self.param_alias.items())

    def update(self, **kwargs):
        """Take a dictionary of parameters, pick out the properly named
        parameters according to the alias, and put them in the param state
        dictionary.
        """
#        for k in self.kernel_params:
#            self.params[k] = kwargs[self.param_alias[k]]
### DOES NOTHING - THERE ARE NO PARAMETERS FOR THE KERNELS
        return

    def __call__(self, metric, weights=None, **extras):
        """Return a function that can be evaluated to return the probability
        density function, given a metric. 
        """
        ### SHOULD I ISSUE ``update`` method here???
        # how to implement weights?
        return self.get_lnlike_function(metric)

class Uncorrelated_photsamples(Kernel_photsamples):
    """
    Need to make a decision: should fpho_samples be a 2d numpy array (niter, nbands)
        of the photometry posterior samples, or should it be the true fpho object?
    for now, let it be a 2d array

    FUTURE: need to handle masking appropriately... and weighting?
    """

    kernel_type = 'uncorrelated_normal'

    def initialize(self, metric):
        """ compute and cache items needed for lnlikelihood function"""
        self.mu = np.mean(metric, axis=0)
        self.cov = np.std(metric, axis=0)**2.
        self.log_det = np.sum(np.log(self.cov))
        self.n = metric.shape[1]

    def eval_lnlike(self, phot_mu):
        residual = phot_mu - self.mu
        first_term = np.dot(residual**2., 1.0 / self.cov)
        lnlike = -0.5 * (first_term + self.log_det + self.n * np.log(2.*np.pi))
        return lnlike

    def get_lnlike_function(self, metric):
        self.initialize(metric)
        return self.eval_lnlike


class Correlated_photsamples(Kernel_photsamples):
    """
    """

    kernel_type = 'correlated_normal'

    def initialize(self, metric, check_finite=False):
        """ compute and cache items needed for lnlikelihood function"""
        self.mu = np.mean(metric, axis=0)
        self.cov = np.cov(metric, rowvar=0) 
        self.factorized_Sigma = cho_factor(self.cov, overwrite_a=True,
                                           check_finite=check_finite)
        self.log_det = 2 * np.sum(np.log(np.diag(self.factorized_Sigma[0])))
        assert np.isfinite(self.log_det)
        self.n = metric.shape[1]

    def eval_lnlike(self, phot_mu, check_finite=False):
        residual = phot_mu - self.mu
        first_term = np.dot(residual, cho_solve(self.factorized_Sigma,
                            residual, check_finite=check_finite))
        lnlike = -0.5 * (first_term + self.log_det + self.n * np.log(2.*np.pi))
        return lnlike

    def get_lnlike_function(self, metric):
        self.initialize(metric)
        return self.eval_lnlike

class KDE_photsamples(Kernel_photsamples):

    kernel_type = 'kde'
    metric_lims = None

    def initialize(self, metric):
        if self.metric_lims is None:
            self.metric_lims = np.percentile(metric, [0, 100], axis=0)
        pdf = {}
        # KDE
        pdf['inbounds'] = KDEMultivariate(data=metric, var_type='c'*metric.shape[1]).pdf
        # Correlated normals (use if trial point is out of bounds)
        pdf['outbounds'] = Correlated_photsamples().get_lnlike_function(metric) 
        self.pdf = pdf

    def eval_lnlike(self, phot_mu):
        # self.pdf must already be set (call beforehand)
        lo_check = np.min( phot_mu - self.metric_lims[0] ) >= 0
        hi_check = np.max( phot_mu - self.metric_lims[1] ) <= 0
        if lo_check * hi_check:
            return np.log(self.pdf['inbounds'](phot_mu))
        else:
            return self.pdf['outbounds'](phot_mu)

    def get_lnlike_function(self, metric):
        self.initialize(metric)
        return self.eval_lnlike
 
