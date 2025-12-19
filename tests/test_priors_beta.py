#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from prospect.models import priors_beta
from scipy.integrate import simpson


def test_phi_met_instantiation():
    prior = priors_beta.PhiMet(
        zred_mini=0.1,
        zred_maxi=2.0,
        mass_mini=9.0,
        mass_maxi=12.0,
        z_mini=-2.0,
        z_maxi=0.5,
        const_phi=True,
    )
    assert prior is not None
    # Test __call__
    x = np.array([1.0, 10.0, 0.0])  # z, mass, met
    lnp = prior(x)
    assert np.all(np.isfinite(lnp))
    # Test sample
    samp = prior.sample()
    assert len(samp) == 3


def test_zred_mass_met_instantiation():
    prior = priors_beta.ZredMassMet(
        zred_mini=0.1,
        zred_maxi=2.0,
        mass_mini=9.0,
        mass_maxi=12.0,
        z_mini=-2.0,
        z_maxi=0.5,
        const_phi=True,
    )
    assert prior is not None
    x = np.array([1.0, 10.0, 0.0])
    lnp = prior(x)
    assert np.all(np.isfinite(lnp))


def test_nz_sfh_instantiation():
    prior = priors_beta.NzSFH(
        zred_mini=0.1,
        zred_maxi=2.0,
        mass_mini=9.0,
        mass_maxi=12.0,
        z_mini=-2.0,
        z_maxi=0.5,
        logsfr_ratio_mini=-5,
        logsfr_ratio_maxi=5,
        logsfr_ratio_tscale=0.3,
        nbins_sfh=7,
        const_phi=True,
    )
    assert prior is not None
    x = np.array(
        [1.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )  # length depends on nbins_sfh
    assert len(prior) == 9

    # x needs to be length 9: z, mass, met, 6 logsfr_ratios
    lnp = prior(x)
    assert np.all(np.isfinite(lnp))


def test_nz_sfh_callable_mass_mini():
    """Test NzSFH with a callable mass_mini, which is a feature supported by the class."""

    def mass_min_func(z):
        # Example function: mass min increases with redshift
        return 9.0 + 0.1 * z

    prior = priors_beta.NzSFH(
        zred_mini=0.1,
        zred_maxi=2.0,
        mass_mini=mass_min_func,
        mass_maxi=12.0,
        z_mini=-2.0,
        z_maxi=0.5,
        logsfr_ratio_mini=-5,
        logsfr_ratio_maxi=5,
        logsfr_ratio_tscale=0.3,
        nbins_sfh=7,
        const_phi=True,
    )
    assert prior is not None
    x = np.array([1.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    lnp = prior(x)
    assert np.all(np.isfinite(lnp))

    samp = prior.sample()
    assert len(samp) == 9


def test_consistency():
    """Test that the unified class (NzSFH logic) gives consistent results with itself.
    Specifically that p(z) is normalized."""

    # We use NzSFH as it has the 'fixed' logic for p(z)
    prior = priors_beta.NzSFH(
        zred_mini=0.5,
        zred_maxi=1.5,
        mass_mini=9.0,
        mass_maxi=11.0,
        z_mini=-2.0,
        z_maxi=0.5,
        logsfr_ratio_mini=-5,
        logsfr_ratio_maxi=5,
        logsfr_ratio_tscale=0.3,
        nbins_sfh=7,
        const_phi=True,
    )

    # Check normalization of p(z)
    z_grid = np.linspace(0.5, 1.5, 1000)
    # The first element of lnp returned by prior(x) is p(z)
    prob_z = np.array(
        [
            np.exp(prior(np.array([z, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))[0])
            for z in z_grid
        ]
    )

    # Integrate p(z)
    integral = simpson(prob_z, x=z_grid)
    assert np.isclose(integral, 1.0, atol=0.05)


def test_dynamic_constraint_strictness():
    """Ensure that p(z) becomes exactly zero when the mass limit
    exceeds the maximum mass (the 'cutoff' redshift)."""

    # m_min(z) = 8.0 + z.
    # Max mass is 13.0.
    # Therefore, at z > 5.0, m_min > 13.0, so p(z) should be 0.
    def cutoff_ramp(z):
        return 8.0 + 1.0 * z

    prior = priors_beta.NzSFH(
        zred_mini=0.1,
        zred_maxi=10.0,
        mass_mini=cutoff_ramp,
        mass_maxi=13.0,
        z_mini=-2.0,
        z_maxi=0.5,
        logsfr_ratio_mini=-5,
        logsfr_ratio_maxi=5,
        logsfr_ratio_tscale=0.3,
        nbins_sfh=7,
        const_phi=True,  # Use simpler mass func for speed
    )

    # Check Redshifts inside the valid region
    x_valid = np.array([4.0, 12.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    lnp_valid = prior(x_valid)
    assert np.all(np.isfinite(lnp_valid))
    assert np.all(lnp_valid > -np.inf)

    # Check Redshifts outside the valid region (z=6.0 -> m_min=14.0 > 13.0)
    # Note: We check the internal p(z) interpolator directly to isolate the z-prior logic
    prob_z_impossible = prior.finterp_z_pdf(6.0)
    assert prob_z_impossible == 0.0

    # Ensure no leakage in sampling
    # We sample 1000 times; max z should be <= 5.0
    samps = prior.sample(nsample=1000)
    z_samps = samps[0]
    assert np.max(z_samps) <= 5.0 + 1e-5  # Add tiny epsilon for float precision


def test_samples_obey_dynamic_bounds():
    """Ensure that every sampled mass is strictly above the redshift-dependent minimum."""

    def mass_ramp(z):
        return 9.0 + 0.5 * z

    prior = priors_beta.NzSFH(
        zred_mini=0.1,
        zred_maxi=4.0,
        mass_mini=mass_ramp,
        mass_maxi=12.0,
        z_mini=-2.0,
        z_maxi=0.5,
        logsfr_ratio_mini=-5,
        logsfr_ratio_maxi=5,
        logsfr_ratio_tscale=0.3,
        nbins_sfh=7,
        const_phi=True,
    )

    samps = prior.sample(nsample=2000)
    z_samps = samps[0]
    m_samps = samps[1]

    # Calculate the required minimum for every sampled redshift
    required_min = mass_ramp(z_samps)

    # Check bounds
    # Use a tiny epsilon for floating point comparisons
    assert np.all(m_samps >= required_min - 1e-7)
    assert np.all(m_samps <= 12.0 + 1e-7)
