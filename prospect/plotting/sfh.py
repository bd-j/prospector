#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import gamma, gammainc

from ..models.transforms import logsfr_ratios_to_masses
from ..sources.constants import cosmo
from .utils import step
from .corner import quantile

__all__ = ["params_to_sfh", "sfh_quantiles",
           "parametric_cmf", "parametric_mwa", "parametric_sfr",
           "ratios_to_sfrs", "sfh_to_cmf", "nonpar_mwa", "nonpar_recent_sfr"]


def params_to_sfh(params, time=None, agebins=None):

    parametric = (time is not None)

    if parametric:
        taus, tages, masses = params["tau"], params["tage"], params["mass"]
        sfhs = []
        cmfs = []
        for tau, tage, mass in zip(taus, tages, masses):
            sfhs.append(parametric_sfr(tau, tage, mass=mass, time=time))
            cmfs.append(parametric_cmf(tau, tage, time))
        lookback = time.max() - time
        sfhs = np.array(sfhs)
        cmfs = np.array(cmfs)

    else:
        logmass = params["logmass"]
        logsfr_ratios = params["logsfr_ratios"]
        sfhs = np.array([ratios_to_sfrs(logm, sr, agebins)
                        for logm, sr in zip(logmass, logsfr_ratios)])
        cmfs = sfh_to_cmf(sfhs, agebins)
        lookback = 10**(agebins-9)

    return lookback, sfhs, cmfs


def sfh_quantiles(tvec, bins, sfrs, weights=None, q=[16, 50, 84]):
    tt = bins.reshape(bins.shape[0], -1)
    ss = np.array([sfrs, sfrs]).T.reshape(bins.shape[0], -1)
    sf = np.array([np.interp(tvec, t, s, left=0, right=0) for t, s in zip(tt, ss)])
    if weights is not None:
        qq = quantile(sf.T, q=np.array(q)/100., weights=weights)
    else:
        qq = np.percentile(sf, axis=0, q=q)
    return qq


def parametric_cmf(tau=4, tage=13.7, time=None):
    if (tage > time.max()) or (tage < 0):
        return np.zeros_like(time)
    tstart = time.max() - tage
    tt = (time - tstart) / tau
    tt[tt < 0] = 0.0
    cmf = gammainc(2, tt)
    cmf /= cmf[-1]
    return cmf


def parametric_mwa_numerical(tau=4, tage=13.7, power=1, n=1000):
    """Compute Mass-weighted age

    :param power: (optional, default: 1)
        Use 0 for exponential decline, and 1 for te^{-t} (delayed exponential decline)
    """
    p = power + 1
    t = np.linspace(0, tage, n)
    tavg = np.trapz((t**p)*np.exp(-t/tau), t) / np.trapz(t**(power) * np.exp(-t/tau), t)
    return tage - tavg


def parametric_mwa(tau=4, tage=13.7, power=1):
    """Compute Mass-weighted age. This is done analytically

    :param power: (optional, default: 1)
        Use 0 for exponential decline, and 1 for te^{-t} (delayed exponential decline)
    """
    tt = tage / tau
    mwt = gammainc(power+2, tt) * gamma(power+2) / gammainc(power+1, tt) * tau
    return tage - mwt


def parametric_sfr(tau=4, tage=13.7, power=1, mass=None, logmass=None, times=None, **extras):
    """Return the SFR (Msun/yr) for the given parameters

    :param power: (optional, default: 1)
        Use 0 for exponential decline, and 1 for te^{-t} (delayed exponential decline)

    :param times: (optional, ndarray)
        If given, a set up times where you want to calculate the sfr
    """
    if (mass is None) and (logmass is not None):
        mass = 10**logmass
    if times is None:
        times = tage
    else:
        assert len(np.atleast_1d(tage)) == 1
        assert len(np.atleast_1d(tau)) == 1
    p = power + 1
    psi = mass * (times/tau)**power * np.exp(-times/tau) / (tau * gamma(p) * gammainc(p, tage/tau))
    return psi * 1e-9


def ratios_to_sfrs(logmass, logsfr_ratios, agebins):
    """scalar
    """
    masses = logsfr_ratios_to_masses(np.squeeze(logmass),
                                     np.squeeze(logsfr_ratios),
                                     agebins)
    dt = (10**agebins[:, 1] - 10**agebins[:, 0])
    sfrs = masses / dt
    return sfrs


def nonpar_recent_sfr(logmass, logsfr_ratios, agebins, sfr_period=0.1):
    """vectorized
    """
    masses = [logsfr_ratios_to_masses(np.squeeze(logm), np.squeeze(sr), agebins)
              for logm, sr in zip(logmass, logsfr_ratios)]
    masses = np.array(masses)
    ages = 10**(agebins - 9)
    # fractional coverage of the bin by the sfr period
    ft = np.clip((sfr_period - ages[:, 0]) / (ages[:, 1] - ages[:, 0]), 0., 1)
    mformed = (ft * masses).sum(axis=-1)
    return mformed / (sfr_period * 1e9)


def nonpar_mwa(logmass, logsfr_ratios, agebins):
    """mass-weighted age, vectorized
    """
    sfrs = np.array([ratios_to_sfrs(logm, sr, agebins)
                     for logm, sr in zip(logmass, logsfr_ratios)])
    ages = 10**(agebins)
    dtsq = (ages[:, 1]**2 - ages[:, 0]**2) / 2
    mwa = [(dtsq * sfr).sum() / 10**logm
           for sfr, logm in zip(sfrs, logmass)]
    return np.array(mwa) / 1e9


def sfh_to_cmf(sfrs, agebins):
    sfrs = np.atleast_2d(sfrs)
    dt = (10**agebins[:, 1] - 10**agebins[:, 0])
    masses = (sfrs * dt)[..., ::-1]
    cmfs = masses.cumsum(axis=-1)
    cmfs /= cmfs[..., -1][..., None]
    zshape = list(cmfs.shape[:-1]) + [1]
    zeros = np.zeros(zshape)
    cmfs = np.append(zeros, cmfs, axis=-1)
    ages = 10**(np.array(agebins) - 9)
    ages = np.array(ages[:, 0].tolist() + [ages[-1, 1]])
    return ages, np.squeeze(cmfs[..., ::-1])
