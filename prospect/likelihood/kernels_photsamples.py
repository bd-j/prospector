"""
testing intermediary product to build flexible NoiseModel objects
"""

import numpy as np
from scipy.stats import multivariate_normal
from statsmodels.nonparametric.kernel_density import KDEMultivariate

__all__ = ["Kernel_photsamples", "Uncorrelated_photsamples", 
           "Correlated_photsamples", "KDE_photsamples"]

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
        return self.get_pdf_function(metric)

class Uncorrelated_photsamples(Kernel_photsamples):
    """
    Need to make a decision: should fpho_samples be a 2d numpy array (niter, nbands)
        of the photometry posterior samples, or should it be the true fpho object?
    for now, let it be a 2d array

    FUTURE: need to handle masking appropriately... and weighting?
    """

#    kernel_params = ['phot_samples']

    def get_pdf_function(self, metric):
        mu = np.mean(metric, axis=0)
        std = np.std(metric, axis=0)
        cov = np.diag( std**2. )
        pdf_function = multivariate_normal(mean=mu, cov=cov).pdf
        return pdf_function 

class Correlated_photsamples(Kernel_photsamples):
    """
    """

#    kernel_params = ['phot_samples']

    def get_pdf_function(self, metric):
        mu = np.mean(metric, axis=0)
        cov = np.cov(metric, rowvar=0)
        pdf_function = multivariate_normal(mean=mu, cov=cov).pdf
        return pdf_function

class KDE_photsamples(Kernel_photsamples):

#    kernel_params = ['phot_samples']

    def get_pdf_function(self, metric):
        pdf_function = KDEMultivariate(data=metric, var_type='c'*metric.shape[1]).pdf
        return pdf_function

