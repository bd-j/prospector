#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
prospect/models/prior_data/__init__.py

This module handles the loading of static data files required for the
Prospector-β priors. These data files include the mass-metallicity relation,
the cosmic star formation rate density, and the redshift-age relation.

Data Sources:
1. Mass-Metallicity Relation: `Gallazzi et al. (2005)`_
2. Star Formation Rate Density: `Behroozi et al. (2019)`_
3. Redshift-Age Relation: WMAP9 cosmology from `Hinshaw et al. (2013)`_

.. _Gallazzi et al. (2005): https://ui.adsabs.harvard.edu/abs/2005MNRAS.362...41G
.. _Behroozi et al. (2019): https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.3143B
.. _Hinshaw et al. (2013): https://ui.adsabs.harvard.edu/abs/2013ApJS..208...19H
"""

import numpy as np
from importlib.resources import files
from scipy.interpolate import UnivariateSpline, interp1d
from typing import Union, Tuple

__all__ = ["MASSMET", "BEHROOZI_SFRD", "WMAP9_AGE", "SPL_TL_SFRD", "F_AGE_Z"]

# 1. Setup the resource path
_base_path = files(__name__)


# 2. Define a helper function to load text files
def _load_txt(filename: str, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Helper function to load text files from the package resources.

    Parameters
    ----------
    filename : str
        The name of the file to load.
    **kwargs
        Additional keyword arguments passed to `numpy.loadtxt`.

    Returns
    -------
    data : np.ndarray
        The loaded data.
    """
    with (_base_path / filename).open("rb") as f:
        return np.loadtxt(f, **kwargs)


# 3. Load Data as Module Constants

# Mass-Metallicity relation (Gallazzi et al. 2005)
# Columns: [logMass, mean_logZ, p16_logZ, p84_logZ]
MASSMET: np.ndarray = _load_txt("gallazzi_05_massmet.txt")

# Star Formation Rate Density (Behroozi et al. 2019)
# Returns (redshift, lookback_time, sfr_density)
_z_b19, _tl_b19, _sfrd_b19 = _load_txt("behroozi_19_sfrd.txt", unpack=True)
# We store the raw tuple if needed, or you can expose individual vectors
BEHROOZI_SFRD: Tuple[np.ndarray, np.ndarray, np.ndarray] = (_z_b19, _tl_b19, _sfrd_b19)

# Redshift-Age relation (WMAP9 cosmology)
# Returns (redshift, age_of_universe_Gyr)
_z_age, _age = _load_txt("wmap9_z_age.txt", unpack=True)
WMAP9_AGE: Tuple[np.ndarray, np.ndarray] = (_z_age, _age)

# 4. Create Derived Objects (Interpolators)

# Spline for lookback time vs SFRD
# x: Lookback time (Gyr)
# y: log10(SFRD)
SPL_TL_SFRD: UnivariateSpline = UnivariateSpline(_tl_b19, _sfrd_b19, s=0, ext=3)

# Interpolator for Age of Universe
# x: Age of universe (Gyr)
# y: Redshift
F_AGE_Z: interp1d = interp1d(_age, _z_age, bounds_error=False, fill_value="extrapolate")
