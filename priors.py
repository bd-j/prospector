#Module containg various functions to be used as priors

import numpy as np

def zeros(theta):
    return np.zeros_like(theta)

def positive(theta):
    p = 1.0 * np.zeros_like(theta)
    n = theta < 0
    p[n] = -np.infty
    return p
