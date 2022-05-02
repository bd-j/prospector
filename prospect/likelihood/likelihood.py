# -*- coding: utf-8 -*-

import time, sys, os
import numpy as np
from scipy.linalg import LinAlgError

from .noise_model import NoiseModel

__all__ = ["compute_lnlike", "compute_chi"]


basic_noise_model = NoiseModel()


def compute_lnlike(pred, obs, vectors={}):
    """Calculate the likelihood of the observational data given the
    prediction.  This is a very thin wrapper on the noise model that should be
    attached to each Observation instance.

    :param pred:
        The predicted data, including calibration.

    :param obs: (optional)
        Instance of observation.Observation() or subclass thereof.

    :param vectors: (optional)
        A dictionary of vectors of same length as ``obs.wavelength`` giving
        possible weghting functions for the kernels
    """
    try:
        return obs.noise.lnlike(pred, obs, vectors=vectors)
    except:
        return basic_noise_model.lnlike(pred, obs, vectors=vectors)


def compute_chi(pred, obs):
    """
    Parameters
    ----------
    pred : ndarray of shape (ndata,)
        The model prediction for this observation

    obs : instance of Observation()
        The observational data
    """
    chi = (pred - obs.flux) / obs.uncertainty
    return chi[obs.mask]

