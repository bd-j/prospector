#Module containg various functions to be used as priors

import numpy as np

def zeros(theta, **extras):
    return np.zeros_like(theta)

def positive(theta, **extras):
    p = 1.0 * np.zeros_like(theta)
    n = theta < 0
    p[n] = -np.infty
    return p

def tophat(theta, mini = 0.0, maxi = 1.0, **extras):
    lnp = 1.0 * np.zeros_like(theta)
    n = (theta < mini) | (theta > maxi)
    lnp[n] = -np.infty
    return lnp

def normal(theta, mean = 0.0, sigma = 1.0, **extras):
    pass
