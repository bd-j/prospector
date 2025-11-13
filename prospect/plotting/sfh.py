#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import Namespace
from copy import deepcopy

import numpy as np
from scipy.special import gamma, gammainc

from ..models.transforms import logsfr_ratios_to_masses
from ..sources.constants import cosmo
from .corner import quantile

__all__ = ["params_to_sfh", "parametric_pset",
           "parametric_cmf", "parametric_mwa", "parametric_sfr",
           "compute_mass_formed",
           "ratios_to_sfrs", "sfh_quantiles",
           "sfh_to_cmf", "nonpar_mwa", "nonpar_recent_sfr"]


def params_to_sfh(params, time=None, agebins=None):

    """Convert a set of SFH parameters into SFR(t) and CMF(t)

    Parameters
    ----------
    params : dict-like
        A dictionary of SFH parameters.  If it contains the keys
        'tau', 'tage', and 'mass', then a parametric SFH is assumed.
        Otherwise, it should contain the keys 'logmass' and
        'logsfr_ratios', and a non-parametric SFH is assumed.
    time : ndarray (optional)
        If given, a set of times where you want to calculate the SFR and CMF, in
        Gyr. These should be forward time, with the maximum value being the age
        of the universe at the redshift of the object.
    agebins : ndarray (optional)
        If `time` is not given, the SFR and CMF will be computed at the edges
        of the agebins.  This should be an array of shape (nbin, 2) giving the
        log10 of the age bin edges in years.  If not given, a default set of
        7 age bins will be used.
    Returns
    -------
    lookback : ndarray
        The lookback times (in Gyr) at which the SFR and CMF are computed.
    sfhs : ndarray
        The SFRs in Msun/yr.  Shape is (nsamples, ntime)
    cmfs : ndarray
        The cumulative mass formed, normalized to 1 at the oldest time.
        Shape is (nsamples, ntime)
    """
    raise NotImplementedError("This function is deprecated.")

    parametric = (time is not None)

    if parametric:
        taus, tages, masses = params["tau"], params["tage"], params["mass"]
        sfhs = []
        cmfs = []
        lookback = time.max() - time
        for tau, tage, mass in zip(taus, tages, masses):
            sfpar = dict(tau=tau, tage=tage, mass=mass, sfh=params["sfh"])
            sfhs.append(parametric_sfr(times=lookback, tavg=0, **sfpar))
            cmfs.append(parametric_cmf(times=lookback, **sfpar))
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


def parametric_pset(logmass=None, **sfh):
    """Convert a dicionary of FSPS parametric SFH parameters into a
    namespace, making sure they are at least 1d vectors

    :param sfh: dicionary
        FSPS parameteric SFH parameters

    :returns pset:
        A Namespace instance with attributes giving the SFH parameters
    """
    # TODO: make multiple psets if any of the SFH parameters have np.size() > 1

    vectors = ["mass", "sf_start", "tage", "tau", "const", "fburst", "tburst", "sf_trunc", "sf_slope"]
    pset = Namespace(mass=1.0, sfh=4., sf_start=0, tage=1,
                     tau=1.,
                     const=0.,
                     fburst=0., tburst=1.,
                     sf_trunc=0, sf_slope=0.)

    if logmass:
        sfh["mass"] = 10**logmass
    for k in vars(pset).keys():
        if k in sfh:
            setattr(pset, k, sfh[k])
    # vectorize
    for k in vectors:
        setattr(pset, k, np.atleast_1d(getattr(pset, k)))

    return pset


def sfh_quantiles(tvec, bins, sfrs, weights=None, q=[16, 50, 84]):
    """Compute quantiles of a binned SFH

    Parameters
    ----------
    tvec : shape (ntime,)
        Vector of lookback times onto which the SFH will be interpolated.

    bins : shape (nsamples, nbin, 2)
        The age bins, in linear untis, same units as tvec

    sfrs : shape (nsamples, nbin)
        The SFR in each bin

    q : list (optional, default: [16, 50, 84])
        List of quantiles to compute, in percent (0-100)

    Returns
    -------
    sfh_q : shape(ntime, nq)
        The quantiles of the SFHs at each lookback time in `tvec`
    """
    tt = bins.reshape(bins.shape[0], -1)
    ss = np.array([sfrs, sfrs]).transpose(1, 2, 0).reshape(bins.shape[0], -1)
    sf = np.array([np.interp(tvec, t, s, left=0, right=0) for t, s in zip(tt, ss)])
    if weights is not None:
        qq = quantile(sf.T, q=np.array(q)/100., weights=weights)
    else:
        qq = np.percentile(sf, axis=0, q=q)
    return qq


def parametric_sfr(times=None, tavg=1e-3, tage=1, **sfh):
    """Return the SFR (Msun/yr) for the given parameters of a parametric SFH,
    optionally averaging over some timescale.

    Parameters
    ----------
    times : (optional, ndarray)
        If given, a set of *lookback* times where you want to calculate the sfr,
        same units as `tau` and `tage`

    tavg : (optional, float, default: 1e-3)
        If non-zero, average the SFR over the last `tavg` Gyr. This can help
        capture bursts.  If zero, the instantaneous SFR will be returned.

    sfh : optional keywords
        FSPS parametric SFH parametrs, e.g. sfh, tage, tau, sf_trunc

    Returns
    -------
    sfr : (ndarray, same shape as `times`)
        SFR in M_sun/year either for the lookback times given by `times` or at
        lookback time 0 if no times are given.  The SFR will either be
        instaneous or averaged over the last `tavg` Gyr.
    """
    if times is None:
        times = np.array(tage)

    pset = parametric_pset(tage=tage, **sfh)
    sfr, mass = compute_mass_formed(tage - times, pset)
    if tavg > 0:
        _, meps = compute_mass_formed((tage - times) - tavg, pset)
        sfr = (mass - meps) / (tavg * 1e9)
    return sfr


def parametric_cmf(times=None, tage=1., **sfh):
    """Return the cumulative formed mass for the given parameters of a
    parametric SFH.

    :param times: (optional, ndarray)
        If given, a set of *lookback* times (relative to `tage`) where you want
        to calculate the formed mass, in Gyr.  If not given, the formed mass
        will be computed for loockback time of 0.

    :param sfh: optional keywords
        FSPS parametric SFH parametrs, e.g. sfh, tage, tau, sf_trunc

    :returns mass: (ndarray)
        Mass formed up to the supplied lookback time, in units of M_sun.
        Same shape as `times`
    """
    if times is None:
        times = np.array(sfh["tage"])

    pset = parametric_pset(tage=tage, **sfh)
    _, mass = compute_mass_formed(tage - times, pset)
    return mass


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


def compute_mass_formed(times, pset):
    """Compute the SFR and stellar mass formed in a parametric SFH,
    as a function of (forward) time.

    The linear portion of the Simha SFH (sfh=5) is defined as:
        psi(t) = psi_trunc + psi_trunc * sf_slope * (t - sf_trunc)
    where psi_trunc is the SFR of the delay-tau SFH at time sf_trunc

    :param times: ndarray of shape (nt,)
        Forward time in Gyr. Use times = pset.tage - t_lookback to
        convert from lookback times

    :param pset: Namespace instance
        The FSPS SFH parameters, assumed to be scalar or 1 element 1-d arrays.
        Usually the output of parametric_pset()

    :returns sfr: ndarray of shape (nt,)
        The instaneous SFR in M_sun/yr at each of `times`

    :returns mfromed: ndarray of shape (nt,)
        The total stellar mass formed from t=0 to `times`, in unist of M_sun
    """
    # TODO: use broadcasting to deal with multiple sfhs?

    # subtract sf_start
    tmass = pset.tage - pset.sf_start  # the age at which the mass is *specified*
    tprime = times - pset.sf_start     # the ages at which sfr and formed mass are requested

    if pset.sfh == 3:
        raise NotImplementedError("This method does not support tabular SFH")

    if np.any(tmass < 0):
        raise ValueError("SF never started (tage - sf_start < 0) for at least one input")

    if (pset.const + pset.fburst) > 1:
        raise ValueError("Constant and burst fractions combine to be > 1")

    if (pset.sfh == 0):
        # SSPs
        mfrac = 1.0 * (tprime > tmass)
        sfr = np.zeros_like(tprime)  # actually the sfr is infinity

    elif pset.sfh > 0:
        # Compute tau model component, for SFH=1,4,5
        #
        # Integration limits are from 0 to tmax and 0 to tprime, where
        #   - tmass is the tage, and
        #   - tprime is the given `time`,
        #   - ttrunc is where the delay-tau truncates
        ttrunc, tend = np.max(tprime), tmass
        if (pset.sf_trunc > 0) and (pset.sf_trunc > pset.sf_start):
            # Otherwise we integrate tau model to sf_trunc - sf_start
            ttrunc = pset.sf_trunc - pset.sf_start
            tend = min(tmass, ttrunc)

        # Now integrate to get mass formed by Tprime and by Tmax, dealing with
        # truncation that happens after sf_start but before Tmax and/or Tprime.
        power = 1 + int(pset.sfh > 3)
        total_mass_tau = pset.tau * gammainc(power, tend / pset.tau)

        tt = np.clip(tprime, 0, ttrunc)
        mass_tau = (tprime > 0.) * pset.tau * gammainc(power, tt / pset.tau)
        # The SFR at Tprime (unnormalized)
        sfr_tau = (tprime > 0.) * (tprime <= ttrunc) * (tprime / pset.tau)**(power-1.) * np.exp(-tprime / pset.tau)
        # fraction of tau component mass formed by tprime
        mfrac_tau = mass_tau / total_mass_tau

    # Add the constant and burst portions, for SFH=1,4.
    if ((pset.sfh == 1) or (pset.sfh == 4)):
        # Fraction of the burst mass formed by Tprime
        tburst = (pset.tburst - pset.sf_start)
        mfrac_burst = 1.0 * (tprime > tburst)
        # SFR from constant portion at Tprime (integrates to 1 at tmax)
        sfr_const = (tprime > 0) * 1.0 / tmass
        # fraction of constant mass formed by tprime
        mfrac_const = np.clip(tprime, 0, ttrunc) * sfr_const

        # Add formed mass fractions for each component, weighted by component fractions.
        # Fraction of the constant mass formed by Tprime is just Tprime/Tmax
        # TODO : The FSPS source does not include the tburst < tmass logic....
        mfrac = ((1. - pset.const - pset.fburst * (tburst < tmass)) * mfrac_tau +
                 pset.const * mfrac_const +
                 pset.fburst * mfrac_burst)

        # N.B. for Tprime = tburst, sfr is infinite, but we ignore that case.
        sfr = ((1. - pset.const - pset.fburst) * sfr_tau / total_mass_tau + pset.const * sfr_const)
        # We've truncated
        sfr *= (tprime <= ttrunc)

    # Add the linear portion, for Simha, SFH=5.
    # This is the integral of sfr_trunc*(1 - m * (T - Ttrunc)) from Ttrunc to Tz
    elif (pset.sfh == 5):
        #raise NotImplementedError

        m = -pset.sf_slope
        if (m > 0):
            # find time at which SFR=0, if m>0
            Tz = ttrunc + 1.0 / m
        else:
            # m <= 0 will never reach SFR=0
            Tz = np.max(tprime)

        # Logic for Linear portion
        if (ttrunc < 0):
            # Truncation does not occur during the SFH.
            total_mass_linear = 0.
            mass_linear = 0.
            sfr = sfr_tau / total_mass_tau
        else:
            # Truncation does occur, integrate linear to zero crossing or tage.
            Thi = min(Tz, tmass)
            sfr_trunc = (ttrunc/pset.tau) * np.exp(-ttrunc / pset.tau)
            total_mass_linear = (Thi > ttrunc) * sfr_trunc * linear_mass(Thi, ttrunc, m)
            mass_linear = (tprime > ttrunc) * sfr_trunc * linear_mass(tprime, ttrunc, m)
            mass_linear[tprime > Tz] = sfr_trunc * linear_mass(Tz, ttrunc, m)
            # SFR in linear portion
            sfr = sfr_trunc * (1 - m * (tprime - ttrunc)) / (total_mass_tau + total_mass_linear)
            sfr *= ((tprime > ttrunc) & (tprime <= Tz))
            # add portion for tau
            sfr[tprime <= ttrunc] = sfr_tau[tprime <= ttrunc] / (total_mass_tau + total_mass_linear)

        mfrac = (mass_tau + mass_linear) / (total_mass_tau + total_mass_linear)

    return pset.mass * sfr/1e9, pset.mass * mfrac


def linear_mass(t, ttrunc, m):
    """Integrate (1-m*(a-ttrunc)) da from a=ttrunc to a=t
    """
    tt = t - ttrunc
    return ((tt + ttrunc * m * tt) - m/2. * (t**2 - ttrunc**2))


default_sfh = dict(mass=1.0, sfh=4, tage=1., tau=2.,
                   sf_start=0., sf_trunc=0., fburst=0., const=0., tburst=0.5)


def show_par_sfh(times, label="", axes=[], tavg=0.01, tol=1e-3, **params):
    sfh = deepcopy(default_sfh)
    sfh.update(params)
    pset = parametric_pset(**sfh)
    sfr, mass = compute_mass_formed(times, pset)
    sf_label = r"SFR"
    if tavg > 0:
        sfr = parametric_sfr(times, **sfh)
        sf_label = r"$\langle {{\rm SFR}}\rangle_{{{}}}$".format(tavg)

    ax = axes[0]
    ax.plot(times, sfr)
    ax.set_ylabel(sf_label)

    ax = axes[1]
    ax.plot(times, mass)
    ax.set_ylabel("M(<t)")
    ax.axhline(pset.mass, linestyle=":", color='k', linewidth=1.0, label="mass at tage")

    [ax.axvline(pset.tage, linestyle="--", color='firebrick', label="tage") for ax in axes]
    if pset.sf_trunc > 0:
        [ax.axvline(pset.sf_trunc, linestyle="--", color='seagreen', label="sf_trunc") for ax in axes]
    if pset.fburst > 0:
        [ax.axvline(pset.tburst, linestyle="--", color='darkorange', label="tburst") for ax in axes]
    if pset.sfh > 5:
        [ax.axvline(pset.sf_trunc, linestyle="--", label="tburst") for ax in axes]

    mr = np.trapz(sfr, times) * 1e9 / mass.max()
    print("{}: {:0.6f}".format(label, mr))
    if (pset.sfh > 0) & (pset.fburst == 0):
        assert np.abs(mr - 1) < tol
    return mr


if __name__ == "__main__":

    # time array
    times = np.linspace(0, 5, 1000)

    # --- test cases ---
    ncases = 14
    import matplotlib.pyplot as pl
    pl.ion()
    fig, axes = pl.subplots(ncases + 1, 2, sharex="row",
                            figsize=(8.5, ncases * 1.5))
    i = 0

    # default
    show_par_sfh(times, label="default", axes=axes[i])

    # sf start
    i += 1
    show_par_sfh(times, label="test SF start", axes=axes[i],
                 sf_start=1.0, tage=2.0)

    # with const
    i += 1
    show_par_sfh(times, label="with const", axes=axes[i],
                 const=0.5)

    # const w/ sf_start
    i += 1
    show_par_sfh(times, label="const w/ sf_start", axes=axes[i],
                 const=0.5, sf_start=1.0, tage=2.0)

    # const w/ sf_start & trunc
    i += 1
    show_par_sfh(times, label="const w/ sf_start & trunc", axes=axes[i],
                 const=0.5, sf_trunc=4., sf_start=1.0, tage=2.0)

    # pure const w/ sf_start & trunc
    i += 1
    show_par_sfh(times, label="pure const w/ sf_start & trunc", tol=5e-3, axes=axes[i],
                 const=1., sf_trunc=4., sf_start=1.0, tage=2.0)

    # burst before tage
    i += 1
    show_par_sfh(times, label="burst before tage", axes=axes[i],
                 fburst=0.5, tburst=0.5)

    # burst after tage
    i += 1
    show_par_sfh(times, label="burst after tage", axes=axes[i],
                 fburst=0.5, tburst=1.5)

    # burst at sf_start
    i += 1
    show_par_sfh(times, label="burst at sf_start", axes=axes[i],
                 fburst=0.5, tburst=1.5, sf_start=1.5, tage=3)

    # pure exp
    i += 1
    show_par_sfh(times, label="pure EXP", tol=1e-2, axes=axes[i],
                 sfh=1)

    # SSP
    i += 1
    show_par_sfh(times, label="SSP", axes=axes[i],
                 sfh=0)

    # positive slope sf trunc before tage
    show_par_sfh(times, label="pos quench before tage", axes=axes[-4],
                 sfh=5, tage=3, sf_trunc=2, tau=0.5, sf_slope=1)

    # negative slope sf trunc before tage
    show_par_sfh(times, label="neg quench before tage", axes=axes[-3],
                 sfh=5, tage=2, sf_trunc=1.5, tau=3, sf_slope=-1)

    # positive slope sf trunc after tage
    show_par_sfh(times, label="pos quench after tage", axes=axes[-2],
                 sfh=5, sf_trunc=3, tau=1, sf_slope=1)

    # negative slope sf trunc after tage
    show_par_sfh(times, label="neg quench after tage", axes=axes[-1],
                 sfh=5, sf_trunc=3, tau=3, sf_slope=-1)
