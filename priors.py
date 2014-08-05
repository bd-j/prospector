#Module containg various functions to be used as priors

import numpy as np

def zeros(theta, **extras):
    return np.zeros_like(theta)

def positive(theta, **extras):
    """
    Require that a parameter be positive.
    """
    p = 1.0 * np.zeros_like(theta)
    n = theta < 0
    p[n] = -np.infty
    return p

def tophat(theta, mini=0.0, maxi=1.0, **extras):
    """
    A simple tophat function.  Input can be scalar or matched vectors
    """
    lnp = 1.0 * np.zeros_like(theta)
    n = (theta < mini) | (theta > maxi)
    lnp[n] = -np.infty
    return lnp

def normal(theta, mean=0.0, sigma=1.0, **extras):
    """
    A simple gaussian.  should make sure it can be vectorized.
    """
    return np.log((2*np.pi)**(-0.5)/sigma) - (theta - mean)**2/(2*sigma**2)

def lognormal(theta, log_mean=0.0, sigma=1.0, **extras):
    """
    A lognormal  gaussian.  should make sure it can be vectorized.
    """
    
    return np.log((2*np.pi)**(-0.5)/(theta*sigma)) - (np.log(theta) - log_mean)**2/(2*sigma**2)

def plotting_range(prior_args):
    if 'mini' in prior_args:
        return prior_args['mini'], prior_args['maxi']
    if 'log_mean' in prior_args:
        return np.exp(prior_args['log_mean'] + np.array([-3, 3.]) *  prior_args['sigma'])
    if 'mean' in prior_args:
        return prior_args['mean'] + np.array([-3, 3.]) *  prior_args['sigma']
