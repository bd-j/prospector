#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import UnivariateSpline

# We import the constants directly to verify they loaded correctly
from prospect.models import prior_data


def test_massmet_data_loaded():
    """Ensure MASSMET table is loaded and has expected shape."""
    assert prior_data.MASSMET is not None
    assert isinstance(prior_data.MASSMET, np.ndarray)
    # The file has 4 columns (mass, P50, P16, P84)
    assert prior_data.MASSMET.shape[1] == 4
    assert prior_data.MASSMET.shape[0] > 0


def test_behroozi_data_loaded():
    """Ensure Behroozi SFRD data is loaded."""
    assert prior_data.BEHROOZI_SFRD is not None
    z, tl, sfrd = prior_data.BEHROOZI_SFRD

    assert len(z) == len(tl) == len(sfrd)
    assert len(z) > 0
    # specific check: redshift should be ascending or descending, not random
    assert np.all(np.diff(z) > 0) or np.all(np.diff(z) < 0)


def test_wmap9_data_loaded():
    """Ensure WMAP9 Age data is loaded."""
    assert prior_data.WMAP9_AGE is not None
    z_age, age = prior_data.WMAP9_AGE
    assert len(z_age) == len(age)
    assert len(z_age) > 0


def test_derived_interpolators():
    """Ensure the splines and interpolators were built successfully."""

    # Check Spline
    assert isinstance(prior_data.SPL_TL_SFRD, UnivariateSpline)
    # Test a dummy evaluation (integral from 0 to 1 should be a float)
    val = prior_data.SPL_TL_SFRD.integral(0, 1)
    assert isinstance(val, float)

    # Check Interpolator
    # interp1d objects are callable
    assert callable(prior_data.F_AGE_Z)
    # Test a dummy evaluation (Age of universe at z=0 should be ~13.7 Gyr)
    z_at_now = prior_data.F_AGE_Z(13.7)
    assert np.isfinite(z_at_now)
