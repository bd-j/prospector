#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from astropy.cosmology import WMAP9 as cosmo
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.stats import kstest
from unittest.mock import patch
from prospect.models import priors_beta


class TestInstantiation:
    def test_phi_met_instantiation(self):
        """
        PhiMet Instantiation

        Verifies that the PhiMet prior (Mass + Metallicity) can be instantiated,
        evaluated for log-probability, and sampled.
        """
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

    def test_zred_mass_met_instantiation(self):
        """
        ZredMassMet Instantiation

        Verifies that the ZredMassMet prior (Redshift + Mass + Metallicity) can be
        instantiated and evaluated for log-probability.
        """
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

    def test_nz_sfh_instantiation(self):
        """
        NzSFH Instantiation

        Verifies that the NzSFH prior (Redshift + Mass + Met + SFH) can be
        instantiated and evaluated with the correct input vector dimensions.
        """
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

    def test_nz_sfh_callable_mass_mini(self):
        """
        NzSFH Instantiation with Callable ``mass_mini``

        Verifies that NzSFH accepts a callable function for `mass_mini`, allowing
        for redshift-dependent mass limits during evaluation and sampling.
        """

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


class TestSMFPhysics:
    def test_leja20_continuity_validation(self):
        """
        Leja+20 Continuity Model

        Verifies that mass_func_at_z correctly reproduces the Double Schechter
        function defined in Leja et al. (2020) at z < 3.

        We independently implement the parameter interpolation and Schechter
        summation to ensure the module's implementation matches the paper's definition.
        """

        # 1. Independent Implementation of Leja+20 Logic

        # Raw parameters from Leja+20 Appendix B
        # Anchor points: z = 0.2, 1.6, 3.0
        pars_ref = {
            "logphi1": [-2.44, -3.08, -4.14],
            "logphi2": [-2.89, -3.29, -3.51],
            "logmstar": [10.79, 10.88, 10.84],
            "alpha1": -0.28,  # Constant
            "alpha2": -1.48,  # Constant
        }
        anchors = [0.2, 1.6, 3.0]

        def manual_interp(y_vals, z_target):
            # Quadratic interpolation between 3 points (y1, y2, y3) at (z1, z2, z3)
            z1, z2, z3 = anchors
            y1, y2, y3 = y_vals

            # Solving for y = az^2 + bz + c
            # (This algebra is derived from the standard solution for 3 points)
            denom = (z1 - z2) * (z1 - z3) * (z2 - z3)
            a = (z3 * (y2 - y1) + z2 * (y1 - y3) + z1 * (y3 - y2)) / denom
            b = (
                z3 * z3 * (y1 - y2) + z2 * z2 * (y3 - y1) + z1 * z1 * (y2 - y3)
            ) / denom
            c = (
                z2 * z3 * (z2 - z3) * y1
                + z3 * z1 * (z3 - z1) * y2
                + z1 * z2 * (z1 - z2) * y3
            ) / denom

            return a * z_target**2 + b * z_target + c

        def manual_schechter_log(logm, logphi, logmstar, alpha):
            # Phi(M) = ln(10) * Phi* * 10^((M-M*)(alpha+1)) * exp(-10^(M-M*))
            # Note: The result is dN/dlogM
            term1 = np.log(10) * (10**logphi)
            term2 = 10 ** ((logm - logmstar) * (alpha + 1))
            term3 = np.exp(-(10 ** (logm - logmstar)))
            return term1 * term2 * term3

        # 2. Setup Test Conditions
        z_test = 1.0
        mass_grid = np.linspace(8.0, 12.0, 100)

        # 3. Compute Expected Values (Manual)
        # Interpolate parameters to z=1.0
        lp1 = manual_interp(pars_ref["logphi1"], z_test)
        lp2 = manual_interp(pars_ref["logphi2"], z_test)
        lms = manual_interp(pars_ref["logmstar"], z_test)

        # Calculate Double Schechter
        phi1 = manual_schechter_log(mass_grid, lp1, lms, pars_ref["alpha1"])
        phi2 = manual_schechter_log(mass_grid, lp2, lms, pars_ref["alpha2"])
        expected_phi = phi1 + phi2

        # 4. Compute Actual Values (Module)
        # We use const_phi=False to enable the z-evolution logic, though at z<3
        # const_phi=False essentially just calls low_z_mass_func.
        actual_phi = priors_beta.mass_func_at_z(
            z=z_test,
            this_logm=mass_grid,
            const_phi=False,
            bounds=[6.0, 13.0],  # Ensure bounds don't cut off our grid
        )

        # 5. Assertions
        # We allow a small tolerance for floating point differences in the interpolation algebra
        np.testing.assert_allclose(
            actual_phi,
            expected_phi,
            rtol=1e-5,
            err_msg=f"Module mass function at z={z_test} does not match manual Leja+20 implementation.",
        )

    def test_tacchella18_highz_validation(self):
        """
        Tacchella+18 High-z Model

        Verifies that at integer redshifts (z=4..12), the mass function correctly
        retrieves the hardcoded parameters and computes the Schechter function
        defined in the module.
        """

        # 1. Re-define the raw data from priors_beta.py to serve as the "Truth"
        # This ensures we are testing against the specific values intended by the authors
        z_t18 = np.arange(4, 13, 1)

        # Original values from code
        phi_raw = np.array([261.9, 201.2, 140.5, 78.0, 38.4, 37.3, 8.1, 3.9, 1.1])
        logm_raw = np.array([10.16, 9.89, 9.62, 9.38, 9.18, 8.74, 8.79, 8.50, 8.50])
        alpha_raw = np.array(
            [-1.54, -1.59, -1.64, -1.70, -1.76, -1.80, -1.92, -2.00, -2.10]
        )

        # Apply transforms done in the module
        phi_t18 = phi_raw * 1e-5
        m_t18 = 10**logm_raw
        alpha_t18 = alpha_raw

        # 2. Define the exact Schechter form used in the module
        def manual_schechter_t18(m, phi_star, m_star, alpha):
            return phi_star * (m / m_star) ** (alpha + 1) * np.exp(-m / m_star)

        # 3. Test Loop over Integer Redshifts
        mass_grid_log = np.linspace(8.0, 12.0, 50)
        mass_grid_linear = 10**mass_grid_log

        for i, z_target in enumerate(z_t18):
            # A. Compute Expected
            # Retrieve parameters for this specific integer redshift
            p_star = phi_t18[i]
            m_star = m_t18[i]
            alp = alpha_t18[i]

            expected_phi = manual_schechter_t18(mass_grid_linear, p_star, m_star, alp)

            # B. Compute Actual
            # const_phi=False allows the code to switch to high-z logic
            actual_phi = priors_beta.mass_func_at_z(
                z=float(z_target),
                this_logm=mass_grid_log,
                const_phi=False,
                bounds=[6.0, 13.0],
            )

            # C. Assertion
            np.testing.assert_allclose(
                actual_phi,
                expected_phi,
                rtol=1e-7,
                err_msg=f"Mismatch in Tacchella+18 mass function at z={z_target}",
            )

    def test_transition_region_continuity(self):
        """
        Transition Region (3 < z < 4)

        Verifies that the mass function transitions smoothly between the Leja+20
        model (z <= 3) and the Tacchella+18 model (z >= 4).

        In the code, this is implemented as a cosine-weighted average of
        Phi_Leja(z=3) and Phi_Tacchella(z=4). Therefore, for a fixed mass,
        the evolution over 3 < z < 4 must be:
        1. Smooth (continuous)
        2. Bounded strictly between the values at z=3 and z=4
        3. Monotonic (since it mixes two constants)
        """

        # 1. Setup
        logm_test = 10.5  # A mass where both functions are well-defined
        z_start = 3.0
        z_end = 4.0

        # 2. Calculate Endpoints
        # const_phi=False is required to enable the transition logic
        phi_3 = priors_beta.mass_func_at_z(z_start, logm_test, const_phi=False)
        phi_4 = priors_beta.mass_func_at_z(z_end, logm_test, const_phi=False)

        # 3. Evaluate on a Fine Grid inside the transition
        z_grid = np.linspace(3.01, 3.99, 50)
        phi_grid = np.array(
            [priors_beta.mass_func_at_z(z, logm_test, const_phi=False) for z in z_grid]
        )

        # 4. Assertions

        # A. Boundedness
        # The interpolated value must assume values strictly between the endpoints
        min_val = min(phi_3, phi_4)
        max_val = max(phi_3, phi_4)

        assert np.all(phi_grid >= min_val), (
            "Interpolation dipped below minimum endpoint."
        )
        assert np.all(phi_grid <= max_val), "Interpolation exceeded maximum endpoint."

        # B. Monotonicity
        # Since we mix two constants with monotonic weights (cos^2),
        # the result must be monotonic.
        diffs = np.diff(phi_grid)

        if phi_4 > phi_3:
            # Should be increasing
            assert np.all(diffs > -1e-15), (
                "Transition should be monotonically increasing."
            )
        else:
            # Should be decreasing
            assert np.all(diffs < 1e-15), (
                "Transition should be monotonically decreasing."
            )

        # C. Smoothness / Continuity at boundaries
        # Check that values near boundaries are close to boundary values
        # z=3.01 should be close to z=3.0
        assert np.isclose(phi_grid[0], phi_3, rtol=0.05), (
            "Discontinuity detected near z=3 boundary."
        )

        # z=3.99 should be close to z=4.0
        assert np.isclose(phi_grid[-1], phi_4, rtol=0.05), (
            "Discontinuity detected near z=4 boundary."
        )


class TestMZRConstraints:
    def test_mzr_mean_relation_accuracy(self):
        """
        Mass-Metallicity Relation (Mean)

        Verifies that loc_massmet correctly interpolates the Gallazzi et al. (2005)
        mass-metallicity relation data.
        """
        # Access the source data directly from the module
        # The array structure is columns: [logM, logZ_50, logZ_16, logZ_84]
        massmet_data = priors_beta.MASSMET

        # 1. Test Exact Data Points
        # Verify that querying exactly at the grid points returns the tabulated values.
        # We test the first point, a middle point, and the last point.
        indices = [0, len(massmet_data) // 2, -1]

        for idx in indices:
            mass_input = massmet_data[idx, 0]
            expected_met = massmet_data[idx, 1]  # 50th percentile (mean/median)

            actual_met = priors_beta.loc_massmet(mass_input)

            # We expect exact matches since we are querying the knots of the interpolation
            np.testing.assert_allclose(
                actual_met,
                expected_met,
                err_msg=f"MZR Mean mismatch at mass={mass_input}",
            )

        # 2. Test Interpolation Logic
        # Pick a mass exactly halfway between two rows in the table to verify linear interpolation
        idx = 5
        m1, z1 = massmet_data[idx, 0], massmet_data[idx, 1]
        m2, z2 = massmet_data[idx + 1, 0], massmet_data[idx + 1, 1]

        target_mass = 0.5 * (m1 + m2)

        # Calculate expected value using linear interpolation formula
        expected_met_interp = z1 + (z2 - z1) * (target_mass - m1) / (m2 - m1)

        actual_met_interp = priors_beta.loc_massmet(target_mass)

        np.testing.assert_allclose(
            actual_met_interp,
            expected_met_interp,
            err_msg=f"MZR Interpolation mismatch at mass={target_mass}",
        )

    def test_mzr_scatter_inflation(self):
        """
        MZR Scatter Inflation

        The Wang et al. (2023) paper states that the Gallazzi et al. (2005)
        confidence intervals are "widened by a factor of 2" to account for
        systematic uncertainties.

        The code implements this by setting the prior's standard deviation (sigma)
        equal to the full 16th-84th percentile width of the original data:
            sigma_prior = (P84_original - P16_original)

        Since (P84 - P16) is approximately 2 * sigma_original, this effectively sets:
            sigma_prior = 2 * sigma_original

        This test verifies that 'scale_massmet' returns the full width (P84 - P16),
        confirming the factor of 2 inflation.
        """

        # 1. Access the source data
        # Columns: [logM, logZ_50, logZ_16, logZ_84]
        massmet_data = priors_beta.MASSMET

        # 2. Pick specific test points from the data rows
        # We test a few rows to ensure consistency
        rows_to_test = [0, 5, -1]

        for row_idx in rows_to_test:
            mass = massmet_data[row_idx, 0]
            p16_data = massmet_data[row_idx, 2]
            p84_data = massmet_data[row_idx, 3]

            # 3. Calculate the Full Width from data
            data_width = p84_data - p16_data

            # 4. Get the sigma used by the code
            code_sigma = priors_beta.scale_massmet(mass)

            # 5. Assert that sigma equals the full width
            # If the code were NOT inflating errors, it would likely return data_width / 2.
            # By returning data_width, it confirms the 2x inflation.
            np.testing.assert_allclose(
                code_sigma,
                data_width,
                err_msg=(
                    f"At mass={mass}, sigma should equal the full P84-P16 width "
                    "to satisfy the 'factor of 2' inflation requirement."
                ),
            )


class TestDynamicSFH:
    def test_sfh_downsizing_eq4(self):
        """
        Dynamic SFH Downsizing (Eq. 4)

        Verifies that delta_t_dex correctly implements the piecewise function
        defined in Equation 4 of Wang et al. (2023).

        The function defines a shift in log(lookback time) based on mass:
        - Mass < 9.0: Constant shift of -0.2 dex
        - Mass > 12.0: Constant shift of +0.8 dex
        - 9.0 < Mass < 12.0: Linear interpolation
        """

        # 1. Define Standard Parameters (from paper/default)
        m_min, m_max = 9.0, 12.0
        d_min, d_max = -0.2, 0.8

        # 2. Check Lower Bound Region (m <= 9.0)
        # Any mass below 9.0 should return exactly d_min
        assert priors_beta.delta_t_dex(8.0) == d_min, (
            "Mass < 9.0 should have constant shift of -0.2"
        )
        assert priors_beta.delta_t_dex(9.0) == d_min, (
            "Mass = 9.0 should have constant shift of -0.2"
        )

        # 3. Check Upper Bound Region (m >= 12.0)
        # Any mass above 12.0 should return exactly d_max
        assert priors_beta.delta_t_dex(12.0) == d_max, (
            "Mass = 12.0 should have constant shift of +0.8"
        )
        assert priors_beta.delta_t_dex(13.0) == d_max, (
            "Mass > 12.0 should have constant shift of +0.8"
        )

        # 4. Check Linear Transition Region
        # Test midpoint: Mass = 10.5
        # Expected: 0.5 * (d_min + d_max) = 0.5 * 0.6 = 0.3
        m_mid = 10.5
        expected_mid = d_min + (m_mid - m_min) * (d_max - d_min) / (m_max - m_min)

        actual_mid = priors_beta.delta_t_dex(m_mid)

        np.testing.assert_allclose(
            actual_mid,
            expected_mid,
            err_msg="Linear interpolation in transition region (m=10.5) is incorrect.",
        )

        # Test random point in transition
        m_rand = 11.0
        # shift = -0.2 + (11.0 - 9.0) * (1.0 / 3.0) = -0.2 + 2/3 = 0.4666...
        expected_rand = d_min + (m_rand - m_min) * (d_max - d_min) / (m_max - m_min)
        actual_rand = priors_beta.delta_t_dex(m_rand)

        np.testing.assert_allclose(
            actual_rand,
            expected_rand,
            err_msg="Linear interpolation in transition region (m=11.0) is incorrect.",
        )

    def test_cosmic_sfrd_integration(self):
        """
        Cosmic SFRD Integration

        Verifies that expe_logsfr_ratios correctly calculates the average SFR
        in each bin by integrating the Behroozi+19 spline (SPL_TL_SFRD) and
        then computes the correct log-ratios.

        We mock 'z_to_agebins_rescale' to return fixed, known time intervals
        so we can verify the integration math independently of the redshift/mass
        shifting logic.
        """

        # 1. Define Fixed Bins (in years, linear)
        # We choose arbitrary but valid ranges within the age of the universe
        # Bin 1: 0 to 1 Gyr
        # Bin 2: 1 Gyr to 2 Gyr
        # Bin 3: 2 Gyr to 5 Gyr
        fixed_bins = np.array([[0.0, 1.0e9], [1.0e9, 2.0e9], [2.0e9, 5.0e9]])

        # 2. Mock the bin generation to return our fixed bins
        # We patch the function where it is defined/used in priors_beta
        with patch(
            "prospect.models.priors_beta.z_to_agebins_rescale", return_value=fixed_bins
        ):
            # 3. Call the function
            # Arguments for z, m, etc. trigger the shifting logic, but since
            # we mock the *result* of that logic (the bins), they don't affect
            # the integration steps we are testing.
            actual_ratios = priors_beta.expe_logsfr_ratios(
                this_z=1.0,
                this_m=10.0,
                logsfr_ratio_mini=-5.0,
                logsfr_ratio_maxi=5.0,
                nbins_sfh=3,  # Matches our fixed_bins length
            )

            # 4. Perform Manual Integration of the Spline
            spline = priors_beta.SPL_TL_SFRD
            manual_sfrs = []

            for i in range(len(fixed_bins)):
                t_start, t_end = fixed_bins[i]
                # Calculate Average SFR = Integral / Duration
                # The spline class provided in the module has an .integral(a, b) method
                total_sfr = spline.integral(a=t_start, b=t_end)
                duration = t_end - t_start
                avg_sfr = total_sfr / duration
                manual_sfrs.append(avg_sfr)

            # 5. Compute Expected Ratios
            # Ratio[i] = log10( SFR[i] / SFR[i+1] )
            expected_ratios = []
            for i in range(len(manual_sfrs) - 1):
                ratio = np.log10(manual_sfrs[i] / manual_sfrs[i + 1])
                expected_ratios.append(ratio)

            expected_ratios = np.array(expected_ratios)

            # 6. Assert Equality
            np.testing.assert_allclose(
                actual_ratios,
                expected_ratios,
                rtol=1e-6,
                err_msg="expe_logsfr_ratios failed to correctly integrate the cosmic SFRD spline.",
            )

    def test_sfh_shape_downsizing(self):
        """
        SFH Shape Validation (Downsizing)

        Verifies that a high-mass galaxy has an 'older' star formation history
        (peak SFR occurs at a larger lookback time) compared to a low-mass galaxy
        at the same redshift.
        """

        # 1. Setup parameters
        z_obs = 0.5  # Observe at z=0.5
        m_high = 11.5  # Massive -> Old population (Declining SFH)
        m_low = 8.0  # Dwarf -> Young population (Rising SFH)
        nbins = 10  # Sufficient resolution

        # 2. Get Log-Ratios from the code
        ratios_high = priors_beta.expe_logsfr_ratios(
            this_z=z_obs,
            this_m=m_high,
            logsfr_ratio_mini=-10,
            logsfr_ratio_maxi=10,
            nbins_sfh=nbins,
        )
        ratios_low = priors_beta.expe_logsfr_ratios(
            this_z=z_obs,
            this_m=m_low,
            logsfr_ratio_mini=-10,
            logsfr_ratio_maxi=10,
            nbins_sfh=nbins,
        )

        # 3. Helper to reconstruct SFR(t) on the galaxy's timeline
        def analyze_sfh(z_in, ratios):
            # A. Reconstruct relative SFR from ratios
            sfrs = np.zeros(nbins)
            sfrs[0] = 1.0
            for i in range(len(ratios)):
                # Ratio[i] = log10(SFR[i] / SFR[i+1])
                sfrs[i + 1] = sfrs[i] / (10 ** ratios[i])

            # B. Get Physical Time Bins for the actual Galaxy
            # The prior's ratios are applied to the galaxy's physical age.
            agebins = priors_beta.z_to_agebins_rescale(zstart=z_in, nbins_sfh=nbins)

            # C. Calculate Peak Metrics
            # agebins are in Lookback Time
            t_mids = np.mean(agebins, axis=1)

            # Find time of peak SFR
            peak_idx = np.argmax(sfrs)
            peak_time = t_mids[peak_idx]

            return peak_time

        # 4. Analyze both galaxies
        peak_time_high = analyze_sfh(z_obs, ratios_high)
        peak_time_low = analyze_sfh(z_obs, ratios_low)

        # 5. Assert Downsizing
        # High mass peak should be at a larger lookback time (older) than low mass
        print(f"\nDownsizing Check at z={z_obs}:")
        print(
            f"High Mass (M={m_high}): Peak Lookback Time = {peak_time_high / 1e9:.2f} Gyr"
        )
        print(
            f"Low Mass  (M={m_low}): Peak Lookback Time = {peak_time_low / 1e9:.2f} Gyr"
        )

        assert peak_time_high > peak_time_low, (
            f"Downsizing failure: High mass galaxy peak ({peak_time_high:.2e}) "
            f"should be older (larger lookback) than low mass peak ({peak_time_low:.2e})."
        )


class TestDynamicRedshiftPrior:
    def test_redshift_prior_normalization(self):
        """
        Redshift Prior Normalization

        Verifies that the marginal redshift probability density function P(z)
        is correctly normalized (integrates to 1.0) over the defined redshift range.
        """

        # We use NzSFH as it has the fixed logic for p(z)
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
        z_grid = np.linspace(0.5, 1.5, 1_000)
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

    def test_prob_zero_when_mass_limit_violated(self):
        """
        Mass Limit Violation

        Verifies that the prior assigns exactly zero probability (or -inf log-prob)
        to redshifts where the minimum observable mass exceeds the maximum physical mass.
        """

        # Set a fixed seed for reproducibility
        np.random.seed(42)

        # m_min(z) = 8.0 + z.
        # Max mass is 13.0.
        # Therefore, at z > 5.0, m_min > 13.0, so p(z) should be 0.
        def cutoff_ramp(z):
            return 8.0 + 1.0 * z

        cutoff_z = 5.0
        mass_max = 13.0

        prior = priors_beta.NzSFH(
            zred_mini=0.1,
            zred_maxi=10.0,
            mass_mini=cutoff_ramp,
            mass_maxi=mass_max,
            z_mini=-2.0,
            z_maxi=0.5,
            logsfr_ratio_mini=-5,
            logsfr_ratio_maxi=5,
            logsfr_ratio_tscale=0.3,
            nbins_sfh=7,
            const_phi=True,  # Use simpler mass func for speed
        )

        # Case A: Valid Redshift (z=4.0 -> m_min=12.0 < 13.0)
        z_valid = 4.0
        x_valid = np.array([z_valid, 12.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        lnp_valid = prior(x_valid)
        assert np.all(np.isfinite(lnp_valid))
        assert np.all(lnp_valid > -np.inf)

        # Case B: Invalid Redshift (z=6.0 -> m_min=14.0 > 13.0)
        z_invalid = 6.0
        # Check internal p(z) interpolator directly to isolate the z-prior logic
        prob_z_impossible = prior.finterp_z_pdf(z_invalid)
        assert prob_z_impossible == 0.0

        # Check Boundary Precision
        # Should be valid just below cutoff, invalid just above
        p_near_valid = prior.finterp_z_pdf(cutoff_z - 0.05)
        p_near_invalid = prior.finterp_z_pdf(cutoff_z + 0.05)
        assert p_near_valid > 0.0, "Should be valid just below cutoff"
        assert p_near_invalid == 0.0, "Should be invalid just above cutoff"

        # Case C: Sampling
        samps = prior.sample(nsample=1_000)
        z_samps = samps[0]
        assert np.max(z_samps) <= cutoff_z + 1e-5, (
            "Samples should not exceed the cutoff redshift"
        )  # Add tiny epsilon for float precision

    def test_samples_obey_dynamic_bounds(self):
        """
        Dynamic Mass Bounds

        Verifies that random samples drawn from the prior strictly adhere to the
        redshift-dependent minimum mass boundaries.
        """

        # Set a fixed seed for reproducibility
        np.random.seed(42)

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

        samps = prior.sample(nsample=2_000)
        z_samps = samps[0]
        m_samps = samps[1]

        # Calculate the required minimum for every sampled redshift
        required_min = mass_ramp(z_samps)

        # Check bounds
        # Use a tiny epsilon for floating point comparisons
        assert np.all(m_samps >= required_min - 1e-7)
        assert np.all(m_samps <= 12.0 + 1e-7)

    def test_volume_scaling_consistency(self):
        """
        Volume Scaling

        Verifies that if the galaxy number density N(z) is constant, the resulting
        redshift prior P(z) scales exactly with the differential comoving volume dV/dz.

        P(z) ~ N(z) * dV/dz
        If N(z) = const, then P(z) ~ dV/dz.
        """

        # 1. Instantiate the prior
        # We use a range that avoids z=0 to prevent divergence in some volume calc edge cases
        z_min, z_max = 0.5, 3.0
        prior = priors_beta.ZredMassMet(
            zred_mini=z_min,
            zred_maxi=z_max,
            mass_mini=9.0,
            mass_maxi=12.0,
            z_mini=-2.0,
            z_maxi=0.2,
            const_phi=True,
        )

        # 2. Mock a "Flat" Mass Function (Constant Number Density)
        # The prior calculates N(z) during __init__ and stores it in self._n_gal_z.
        # It also stores the redshift grid used for this calculation in self._z_grid_norm.
        # We overwrite _n_gal_z with ones to simulate a constant number density.
        prior._n_gal_z = np.ones_like(prior._z_grid_norm)

        # 3. Force re-calculation of the PDF interpolators
        # This uses our mocked _n_gal_z combined with the real cosmology to create P(z)
        prior._setup_dynamic_z_prior_from_norm()

        # 4. Compute the Expected Distribution (dV/dz)
        # We evaluate on the same grid used internally to minimize interpolation errors
        z_grid = prior._z_grid_norm
        dvol_physical = cosmo.differential_comoving_volume(z_grid).value

        # Normalize dV/dz so it integrates to 1.0 (to match the PDF)
        # The code uses simpson integration internally, so we do the same.
        norm_factor = simpson(y=dvol_physical, x=z_grid)
        pdf_expected = dvol_physical / norm_factor

        # 5. Compute the Actual Distribution from the Code
        # finterp_z_pdf is the normalized P(z) constructed by the class
        pdf_actual = prior.finterp_z_pdf(z_grid)

        # 6. Compare
        # We exclude the very edges which can have interpolation artifacts
        mask = (z_grid > z_min + 0.05) & (z_grid < z_max - 0.05)

        # Check that they match within a small tolerance (e.g., 1%)
        # This proves the code correctly multiplies N(z) by the comoving volume.
        np.testing.assert_allclose(
            pdf_actual[mask],
            pdf_expected[mask],
            rtol=0.01,
            err_msg="P(z) does not scale with Comoving Volume when N(z) is constant.",
        )


class TestStatisticalConsistency:
    def test_mass_function_sampling_consistency(self):
        """
        Mass Function

        Verifies that samples drawn from the mass prior at a fixed redshift
        statistically match the analytic PDF of the mass function.

        We use a Kolmogorov-Smirnov (KS) test.
        """

        # Set a fixed seed for reproducibility
        np.random.seed(42)

        # 1. Setup a prior with fixed redshift to isolate the mass sampling
        test_z = 1.0
        mass_min = 9.0
        mass_max = 12.0

        prior = priors_beta.BetaPrior(
            z_prior="fixed",
            zred=test_z,
            mass_prior="mass_function",
            mass_mini=mass_min,
            mass_maxi=mass_max,
            z_mini=-2.0,
            z_maxi=0.2,
            const_phi=True,  # Use simple Leja+20 model
            name="test_mass_consistency",
        )

        # 2. Generate a large number of samples
        n_samples = 10_000
        # Returns shape (n_samples, n_params) -> [z, mass, met]
        samples = prior.sample(nsample=n_samples)
        mass_samples = samples[:, 1]

        # 3. Construct the Reference CDF
        # We must integrate the PDF used by the code to get the "Truth" CDF
        m_grid = np.linspace(mass_min, mass_max, 1_000)

        # priors_beta.cdf_mass_func_at_z returns the CDF normalized to [0, 1]
        # Note: We must pass the same bounds as the prior
        cdf_grid = priors_beta.cdf_mass_func_at_z(
            z=test_z, logm=m_grid, const_phi=True, bounds=[mass_min, mass_max]
        )

        # Create a callable CDF function for the KS test
        # bounds_error=False/fill_value ensures numerical noise at edges doesn't crash test
        cdf_func = interp1d(
            m_grid, cdf_grid, kind="linear", bounds_error=False, fill_value=(0.0, 1.0)
        )

        # 4. Perform Kolmogorov-Smirnov Test
        # H0: The samples come from the distribution defined by cdf_func
        # We reject H0 if p_value is very small (e.g. < 0.001)
        d_statistic, p_value = kstest(mass_samples, cdf_func)

        # Debug info in case of failure
        print(f"\nMass KS Test: D={d_statistic:.4f}, p={p_value:.4f}")

        # Assert that we cannot reject the null hypothesis
        # (i.e., the samples are consistent with the distribution)
        assert p_value > 0.001, (
            f"Mass sampling inconsistent with PDF (p={p_value:.4f}). "
            "The sampling logic likely diverges from the probability definition."
        )

    def test_dynamic_z_sampling_consistency(self):
        """
        Dynamic Redshift

        Verifies that samples drawn from the dynamic redshift prior (ZredMassMet)
        statistically match the calculated marginal P(z) distribution.
        """

        # Set a fixed seed for reproducibility
        np.random.seed(42)

        z_min = 0.5
        z_max = 3.0
        mass_min = 9.0
        mass_max = 12.0

        # 1. Setup the dynamic prior
        prior = priors_beta.ZredMassMet(
            zred_mini=z_min,
            zred_maxi=z_max,
            mass_mini=mass_min,
            mass_maxi=mass_max,
            z_mini=-2.0,
            z_maxi=0.2,
            const_phi=True,
        )

        # 2. Generate samples
        n_samples = 15_000
        samples = prior.sample(nsample=n_samples)
        z_samples = samples[:, 0]

        # 3. Reconstruct the Reference Forward CDF
        # The prior calculates P(z) internally as `pdf_z_unnorm`.
        # We can access the internal interpolator `finterp_z_pdf` to reconstruct the CDF.

        z_grid = np.linspace(z_min, z_max, 2_000)
        pdf_values = prior.finterp_z_pdf(z_grid)

        # Numerically integrate to get CDF
        from scipy.integrate import simpson

        cdf_values = np.zeros_like(z_grid)
        for i in range(1, len(z_grid)):
            cdf_values[i] = simpson(y=pdf_values[: i + 1], x=z_grid[: i + 1])

        # Normalize
        cdf_values /= cdf_values[-1]

        # Create callable CDF
        cdf_func = interp1d(
            z_grid, cdf_values, kind="linear", bounds_error=False, fill_value=(0.0, 1.0)
        )

        # 4. Perform KS Test
        d_statistic, p_value = kstest(z_samples, cdf_func)

        print(f"\nRedshift KS Test: D={d_statistic:.4f}, p={p_value:.4f}")

        assert p_value > 0.001, (
            f"Redshift sampling inconsistent with P(z) (p={p_value:.4f}). "
            "Check _setup_dynamic_z_prior_from_norm logic."
        )


class TestEdgeCaseStability:
    def test_sfr_ratio_nan_recovery(self):
        """
        NaN Handling in SFH

        Verifies that expe_logsfr_ratios correctly handles cases where SFRs are zero,
        producing NaNs (0/0) or Infs.

        We accomplish this by temporarily patching the SPL_TL_SFRD object
        used inside the module to return zeros, forcing 0/0 conditions.
        """

        # 1. Define a Mock Spline to control integral outputs
        # We want to force a sequence that triggers the specific NaN logic.
        # The logic requires first_nan_idx > 0 to work (it copies the left neighbor).
        # So we need:
        # Bin 0: Integral > 0
        # Bin 1: Integral = 0  --> Ratio 0 = Bin0/Bin1 = Inf (Handled by clip)
        # Bin 2: Integral = 0  --> Ratio 1 = Bin1/Bin2 = 0/0 = NaN (Handled by fill)

        class MockSplineSeq:
            def __init__(self):
                self.call_count = 0

            def integral(self, a, b):
                self.call_count += 1
                if self.call_count == 1:
                    return 1.0  # First bin has mass
                if self.call_count == 2:
                    return 0.0  # Second bin empty
                if self.call_count == 3:
                    return 0.0  # Third bin empty -> Ratio (Bin1/Bin2) is 0/0 NaN
                return 1.0  # Others valid

        # 2. Patch the module global
        # We must save the original to restore it after the test
        original_spline = priors_beta.SPL_TL_SFRD
        priors_beta.SPL_TL_SFRD = MockSplineSeq()

        try:
            # 3. Trigger the function
            # Parameters don't strictly matter as the Spline ignores them,
            # but we need enough bins to hit our sequence.
            ratio_max = 5.0

            res = priors_beta.expe_logsfr_ratios(
                this_z=1.0,
                this_m=10.0,
                logsfr_ratio_mini=-5.0,
                logsfr_ratio_maxi=ratio_max,
                nbins_sfh=5,
            )

            # 4. Assertions

            # A. Finite Check
            assert np.all(np.isfinite(res)), f"Result contained NaNs or Infs: {res}"

            # B. Logic Check
            # Ratio 0: log10(1.0 / 0.0) = Inf -> Clipped to ratio_max
            assert res[0] == ratio_max, (
                "First bin should be clipped max (Inf handling)."
            )

            # Ratio 1: log10(0.0 / 0.0) = NaN -> Forward filled from Ratio 0
            assert res[1] == res[0], (
                f"Forward fill failed. Expected {res[0]}, got {res[1]}"
            )

        finally:
            # 5. Restore the original spline
            priors_beta.SPL_TL_SFRD = original_spline

    def test_logprob_is_neg_inf_outside_bounds(self):
        """
        Out of Bounds Inputs

        Verifies that the prior returns -inf (or the log-zero clamp) when parameters
        are outside the defined physical bounds.
        """

        # 1. Setup a standard prior
        z_min, z_max = 0.5, 2.0
        mass_min, mass_max = 9.0, 11.0
        met_min, met_max = -1.0, 0.5

        prior = priors_beta.NzSFH(
            zred_mini=z_min,
            zred_maxi=z_max,
            mass_mini=mass_min,
            mass_maxi=mass_max,
            z_mini=met_min,
            z_maxi=met_max,
            logsfr_ratio_mini=-5,
            logsfr_ratio_maxi=5,
            logsfr_ratio_tscale=0.3,
            nbins_sfh=5,
            const_phi=True,
        )

        # 2. Define Inputs [z, mass, met, ratio1, ratio2, ratio3, ratio4]
        x_valid = np.array([1.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        x_mass_high = x_valid.copy()
        x_mass_high[1] = 12.0  # > 11.0

        x_z_low = x_valid.copy()
        x_z_low[0] = 0.1  # < 0.5

        x_met_high = x_valid.copy()
        x_met_high[2] = 1.0  # > 0.5

        # 3. Evaluate
        with np.errstate(divide="ignore"):
            lnp_valid = prior(x_valid)
            lnp_mass_high = prior(x_mass_high)
            lnp_z_low = prior(x_z_low)
            lnp_met_high = prior(x_met_high)

        # 4. Assertions

        # A. Valid Check
        assert np.all(np.isfinite(lnp_valid)), (
            "Valid input should return finite log-prob."
        )

        # B. Mass Check (Index 1)
        # Check specific component is -inf
        assert lnp_mass_high[1] == -np.inf, (
            f"Mass component (idx 1) should be -inf. Got {lnp_mass_high}"
        )
        # Check total probability is -inf
        assert np.sum(lnp_mass_high) == -np.inf, (
            "Total log-prob should be -inf for invalid mass"
        )

        # C. Redshift Check (Index 0)
        # Dynamic priors clamp 0 probability to ~1e-300 (-690 log)
        assert lnp_z_low[0] < -600.0, (
            f"Redshift component (idx 0) should be < -600. Got {lnp_z_low[0]}"
        )
        assert np.sum(lnp_z_low) < -600.0, (
            "Total log-prob should be effectively -inf for invalid z"
        )

        # D. Metallicity Check (Index 2)
        assert lnp_met_high[2] == -np.inf, (
            f"Metallicity component (idx 2) should be -inf. Got {lnp_met_high}"
        )
        assert np.sum(lnp_met_high) == -np.inf, (
            "Total log-prob should be -inf for invalid metallicity"
        )
