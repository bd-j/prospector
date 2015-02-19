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
    if np.all(theta > 0):
        return ( np.log((2*np.pi)**(-0.5)/(theta*sigma)) -
                (np.log(theta) - log_mean)**2/(2*sigma**2) )
    else:
        return -np.infty

def plotting_range(prior_args):
    if 'mini' in prior_args:
        return prior_args['mini'], prior_args['maxi']
    if 'log_mean' in prior_args:
        mini = np.atleast_1d(prior_args['log_mean']) - 10 * np.array(prior_args['sigma'])
        maxi = np.atleast_1d(prior_args['log_mean']) + 10 * np.array(prior_args['sigma'])
        return np.exp(mini).tolist(), np.exp(maxi).tolist()
    if 'mean' in prior_args:
        mini = np.array(prior_args['mean']) - 10 * np.array(prior_args['sigma'])
        maxi = np.array(prior_args['mean']) + 10 * np.array(prior_args['sigma'])
        return mini.tolist(), maxi.tolist()


class Prior(object):
    """
    Encapsulate the priors in an object.  On initialization each prior
    should have a function name and optioanl parameters specifiyfy
    scale and location passed (e.g. min/max or mean/sigma).  When
    called, the argument should be a variable and it should return the
    prior probability of that value.  One should be able to sample
    from the prior, and to get the gradient of the prior at any
    variable value.  Methods should also be avilable to give a useful
    plotting range and, if there are bounds, to return them.
    """

    def __init__( kind, *args):
        pass
    def __call__(theta):
        pass
    def sample(nsample):
        pass
    def gradient(theta):
        pass
    def range():
        pass
    @property
    def bounds():
        pass

    def serialize():
        pass
