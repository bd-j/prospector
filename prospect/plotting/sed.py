#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from ..utils.smoothing import smoothspec

__all__ = ["convolve_spec", "to_nufnu"]


def convolve_spec(wave, flux, R, minw=1e3, maxw=5e4, nufnu=False):
    """Convolve a spectrum for display

    Parameters
    ----------
    wave : ndarray opf shape (nwave,)
        observed frame wavelength,

    flux : iterable or ndarray of shape (nspec, nwave)
        A list or array of spectra
    """
    dlnlam = 1.0 / R / 2
    owave = np.exp(np.arange(np.log(minw), np.log(maxw), dlnlam))
    fout = [smoothspec(wave, f, resolution=R, outwave=owave, smoothtype="R")
            for f in flux]
    fout = np.array(fout)
    if nufnu:
        return to_nufnu(owave, fout)
    else:
        return owave, fout


def to_nufnu(ang, maggies):
    return ang / 1e4, maggies * 3631e-23 * 3e18 / ang
