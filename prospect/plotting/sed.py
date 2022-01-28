#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from ..utils.smoothing import smoothspec

__all__ = ["convolve_spec", "to_nufnu"]


def convolve_spec(wave, flux, R, minw=1e3, maxw=5e4,
                  nufnu=False, microns=False, fftsmooth=True):
    """Convolve a spectrum for display purposes

    Parameters
    ----------
    wave : ndarray opf shape (nwave,)
        observed frame wavelength,

    flux : iterable or ndarray of shape (nspec, nwave)
        A list or array of spectra

    R : float
        Resolution in units of lambda/sigma(lambda)
    """
    dlnlam = 1.0 / R / 2
    owave = np.exp(np.arange(np.log(minw), np.log(maxw), dlnlam))
    fout = [smoothspec(wave, f, resolution=R, outwave=owave, smoothtype="R", fftsmooth=fftsmooth)
            for f in np.atleast_2d(flux)]
    fout = np.squeeze(np.array(fout))
    if nufnu:
        return to_nufnu(owave, fout, microns)
    else:
        return owave / 10**(4 * microns), fout


def to_nufnu(ang, maggies, microns=True):
    """Convert from maggies (f_nu) and angstroms to nu*f_nu (cgs) and
    optionally microns
    """
    if microns:
        w = ang / 1e4
    else:
        w = ang
    return w, maggies * 3631e-23 * 3e18 / ang
