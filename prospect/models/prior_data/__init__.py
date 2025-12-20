#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
prospect/models/prior_data/__init__.py
Handles loading of static data files for priors.
"""

import numpy as np
from importlib.resources import files
from scipy.interpolate import UnivariateSpline, interp1d

__all__ = ["MASSMET", "BEHROOZI_SFRD", "WMAP9_AGE", "SPL_TL_SFRD", "F_AGE_Z"]

# 1. Setup the resource path
_base_path = files(__name__)


# 2. Define a helper function to load text files
def _load_txt(filename, **kwargs):
    with (_base_path / filename).open("rb") as f:
        return np.loadtxt(f, **kwargs)


# 3. Load Data as Module Constants

# Gallazzi 05: Mass-Metallicity relation
MASSMET = _load_txt("gallazzi_05_massmet.txt")

# Behroozi 19: Star Formation Rate Density
# Returns (redshift, lookback_time, sfr_density)
_z_b19, _tl_b19, _sfrd_b19 = _load_txt("behroozi_19_sfrd.txt", unpack=True)
# We store the raw tuple if needed, or you can expose individual vectors
BEHROOZI_SFRD = (_z_b19, _tl_b19, _sfrd_b19)

# WMAP9: Redshift-Age relation
_z_age, _age = _load_txt("wmap9_z_age.txt", unpack=True)
WMAP9_AGE = (_z_age, _age)

# 4. Create Derived Objects (Interpolators)

# Spline for lookback time vs SFRD
SPL_TL_SFRD = UnivariateSpline(_tl_b19, _sfrd_b19, s=0, ext=3)

# Interpolator for Age of Universe
F_AGE_Z = interp1d(_age, _z_age, bounds_error=False, fill_value="extrapolate")
