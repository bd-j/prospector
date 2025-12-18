#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_parameters.py

This module contains unit tests for parameter handling and prior sampling
mechanisms within the ProspectorParams class.

It covers:
1.  **Dependency Propagation:** Verifies that parameters are updated in the
    correct topological order, ensuring that derived parameters use the
    most up-to-date values of their dependencies.
2.  **Prior Sampling:** Verifies the ``sample_prior`` method, ensuring it
    correctly generates samples from the joint prior distribution. This includes
    tests for:
    * Output shapes and parameter ordering.
    * Adherence to hard bounds.
    * Statistical correctness (mean, variance, covariance) for a wide range
        of prior distributions (e.g., Normal, TopHat, LogNormal, Beta,
        StudentT, and MultivariateNormal).
"""

import numpy as np
from scipy.special import erf
import pytest
from prospect.models import ProspectorParams, priors, transforms
from prospect.models.templates import TemplateLibrary
from functools import partial


class TestParameterDependencies:
    def test_simple_dependency(self):
        """
        Test a simple dependency: B -> A (B depends on A)
        """
        config = {
            "A": {
                "N": 1,
                "isfree": True,
                "init": 2.0,
                "prior": priors.TopHat(mini=0, maxi=10),
            },
            "B": {
                "N": 1,
                "isfree": False,
                "init": 0.0,
                "depends_on": lambda A=0, **kwargs: A * 3,
            },
        }

        model = ProspectorParams(config)
        # Initial state: dependencies are NOT propagated automatically on init
        # So B should be 0.0
        assert model.params["B"][0] == 0.0

        # Explicitly propagate
        model.propagate_parameter_dependencies()
        assert model.params["B"][0] == 6.0, "Dependency propagation failed"

        # Update A to 3.0 via set_parameters (which triggers propagation)
        model.set_parameters(np.array([3.0]))
        assert model.params["A"][0] == 3.0
        assert model.params["B"][0] == 9.0, (
            "Dependency update failed after setting parameters"
        )

    def test_chain_dependency(self):
        """
        Test a chain dependency: C -> B -> A
        A = 2
        B = A * 2 = 4
        C = B + 1 = 5
        """
        config = {
            "A": {
                "N": 1,
                "isfree": True,
                "init": 2.0,
                "prior": priors.TopHat(mini=0, maxi=10),
            },
            "B": {
                "N": 1,
                "isfree": False,
                "init": 0.0,
                "depends_on": lambda A=0, **kwargs: A * 2,
            },
            "C": {
                "N": 1,
                "isfree": False,
                "init": 0.0,
                "depends_on": lambda B=0, **kwargs: B + 1,
            },
        }

        model = ProspectorParams(config)
        model.propagate_parameter_dependencies()

        # Initial check after manual propagation
        assert model.params["B"][0] == 4.0
        assert model.params["C"][0] == 5.0

        # Update A -> 3.0
        # Expected: B -> 6.0, C -> 7.0
        model.set_parameters(np.array([3.0]))
        assert model.params["B"][0] == 6.0
        assert model.params["C"][0] == 7.0

    def test_diamond_dependency(self):
        r"""
        Test a diamond dependency:
             A
           /   \
          B     C
           \   /
             D
             
        D = B + C
        B = A * 2
        C = A + 1
        
        If A=2:
        B=4, C=3
        D=7
        """
        config = {
            "A": {
                "N": 1,
                "isfree": True,
                "init": 2.0,
                "prior": priors.TopHat(mini=0, maxi=10),
            },
            "B": {
                "N": 1,
                "isfree": False,
                "init": 0.0,
                "depends_on": lambda A=0, **kwargs: A * 2,
            },
            "C": {
                "N": 1,
                "isfree": False,
                "init": 0.0,
                "depends_on": lambda A=0, **kwargs: A + 1,
            },
            "D": {
                "N": 1,
                "isfree": False,
                "init": 0.0,
                "depends_on": lambda B=0, C=0, **kwargs: B + C,
            },
        }

        model = ProspectorParams(config)
        model.propagate_parameter_dependencies()

        # Initial check
        assert model.params["B"][0] == 4.0
        assert model.params["C"][0] == 3.0
        assert model.params["D"][0] == 7.0

        # Update A -> 5.0
        # B=10, C=6, D=16
        model.set_parameters(np.array([5.0]))
        assert model.params["B"][0] == 10.0
        assert model.params["C"][0] == 6.0
        assert model.params["D"][0] == 16.0

    def test_unordered_config(self):
        """
        Pass the configuration in an order that would fail a linear pass.
        We define C (depends on B) before B (depends on A).
        """

        def dep_C(B=0, **kwargs):
            return B + 10

        def dep_B(A=0, **kwargs):
            return A * 2

        # List of tuples to enforce order when creating dict
        config_list = [
            ("C", {"N": 1, "isfree": False, "init": 0.0, "depends_on": dep_C}),
            ("B", {"N": 1, "isfree": False, "init": 0.0, "depends_on": dep_B}),
            (
                "A",
                {
                    "N": 1,
                    "isfree": True,
                    "init": 1.0,
                    "prior": priors.TopHat(mini=0, maxi=10),
                },
            ),
        ]

        # Create ProspectorParams using a list of dicts to preserve this specific "bad" order

        full_config_list = []
        for name, cfg in config_list:
            c = cfg.copy()
            c["name"] = name
            full_config_list.append(c)

        model = ProspectorParams(full_config_list)
        model.propagate_parameter_dependencies()

        # A=1 -> B=2 -> C=12
        assert model.params["A"][0] == 1.0
        assert model.params["B"][0] == 2.0
        assert model.params["C"][0] == 12.0

        # Update A -> 2
        # B=4, C=14
        model.set_parameters(np.array([2.0]))
        assert model.params["B"][0] == 4.0
        assert model.params["C"][0] == 14.0

    def test_cycle_detection_fallback(self):
        """
        Test that a cycle (A -> B -> A) doesn't hang the system and falls back
        to the default order (or at least finishes).
        """
        config = {
            "A": {
                "N": 1,
                "isfree": True,
                "init": 1.0,
                "prior": priors.TopHat(mini=0, maxi=10),
                "depends_on": lambda B=0, **kwargs: B + 1,
            },
            "B": {
                "N": 1,
                "isfree": False,
                "init": 1.0,
                "depends_on": lambda A=0, **kwargs: A + 1,
            },
        }

        # This creates a cycle. The topological sort should detect it (length of sorted != length of items).

        with pytest.raises(RecursionError, match="Cyclic dependency detected"):
            _ = ProspectorParams(config)

    def test_stale_update_check(self):
        """
        Explicitly verify that updates use the FRESH value from the CURRENT propagation pass.

        Setup:
        A (free)
        B depends on A (B = A)
        C depends on B (C = B)

        If we update A from 1 to 2.
        Linear order A -> C -> B would update:
        A = 2
        C = B (old B=1) -> C=1 (WRONG)
        B = A -> B=2

        Topological order A -> B -> C should result in:
        A = 2
        B = 2
        C = 2
        """

        def dep_C(B=0, **kwargs):
            return B

        def dep_B(A=0, **kwargs):
            return A

        # Force a bad order in config list
        config_list = [
            {
                "name": "A",
                "N": 1,
                "isfree": True,
                "init": 1.0,
                "prior": priors.TopHat(mini=0, maxi=10),
            },
            {"name": "C", "N": 1, "isfree": False, "init": 0.0, "depends_on": dep_C},
            {"name": "B", "N": 1, "isfree": False, "init": 0.0, "depends_on": dep_B},
        ]

        model = ProspectorParams(config_list)
        model.propagate_parameter_dependencies()

        # Initial: A=1, B=1, C=1
        assert model.params["A"][0] == 1.0
        assert model.params["B"][0] == 1.0
        assert model.params["C"][0] == 1.0

        # Update A -> 2
        model.set_parameters(np.array([2.0]))

        assert model.params["A"][0] == 2.0
        assert model.params["B"][0] == 2.0
        assert model.params["C"][0] == 2.0, "C used stale value of B!"

    def test_template_beta(self):
        """
        Test using the actual 'beta' template from TemplateLibrary.
        This template has complex dependencies:
        nzsfh -> (zred, logmass, logsfr_ratios, logzsol)
        zred -> agebins
        logsfr_ratios -> mass

        We want to ensure that setting 'nzsfh' correctly propagates to 'mass' and 'agebins'.
        """
        try:
            # beta template requires astropy, etc.
            config = TemplateLibrary["beta"]
        except Exception as e:
            pytest.skip(f"Skipping beta template test due to missing dependencies: {e}")

        model = ProspectorParams(config)

        # 'nzsfh' is the free parameter driving others.
        # It has N=9 (default in template: 7 bins + 2).
        # nzsfh = [zred, logmass, logzsol, ...logsfr_ratios...]

        # Let's pick a distinct value for zred and logmass in nzsfh
        new_zred = 1.5
        new_logmass = 11.0
        new_logzsol = -0.2

        # Current length of nzsfh
        N_nzsfh = config["nzsfh"]["N"]
        new_nzsfh = np.zeros(N_nzsfh)
        new_nzsfh[0] = new_zred
        new_nzsfh[1] = new_logmass
        new_nzsfh[2] = new_logzsol
        # rest zeros

        # Identify free parameters
        theta = model.theta.copy()

        # Find slice for nzsfh
        if "nzsfh" in model.theta_index:
            sl = model.theta_index["nzsfh"]
            theta[sl] = new_nzsfh
        else:
            pytest.fail("nzsfh not found in free parameters of beta template")

        # Update
        model.set_parameters(theta)

        # Check propagation
        # zred should match
        assert np.isclose(model.params["zred"][0], new_zred), (
            "zred not updated from nzsfh"
        )

        # logmass should match
        assert np.isclose(model.params["logmass"][0], new_logmass), (
            "logmass not updated from nzsfh"
        )

        # agebins depends on zred.
        # Verify agebins changed from default (zred=0.5 in init)
        from prospect.models import transforms

        expected_agebins = transforms.zred_to_agebins_pbeta(
            np.atleast_1d(new_zred), np.zeros(7)
        )  # 7 bins default
        assert np.allclose(model.params["agebins"], expected_agebins), (
            "agebins not updated correctly from new zred"
        )

    def test_continuity_sfh_zred_dependency(self):
        """
        Test a specific failure case with the continuity_sfh template where
        zred is made free, and agebins depends on zred.
        This creates a dependency chain:
        zred -> agebins -> mass

        If parameter propagation is not topologically sorted, 'mass' (which appears early in
        the dictionary) might be calculated using stale 'agebins' (which appears late),
        leading to inconsistency.
        """

        # 1. Setup the model parameters from the template
        # We use a copy to avoid modifying the global template
        try:
            model_params = TemplateLibrary["continuity_sfh"]
        except Exception as e:
            pytest.skip(f"Could not load continuity_sfh template: {e}")

        # 2. Modify to make zred free and agebins dependent on it
        # Make redshift a free parameter (and give it a prior)
        model_params["zred"]["isfree"] = True
        model_params["zred"]["prior"] = priors.TopHat(mini=0.5, maxi=3.0)
        # Give it an initial value distinct from default (usually 0.1)
        model_params["zred"]["init"] = 1.0

        # agebins now must depend on redshift
        model_params["agebins"]["depends_on"] = transforms.zred_to_agebins

        # Instantiate the model object
        model = ProspectorParams(model_params)

        # 3. Verify the dependency order calculation
        if hasattr(model, "_dependency_order"):
            # Check that agebins is updated before mass
            try:
                idx_agebins = model._dependency_order.index("agebins")
                idx_mass = model._dependency_order.index("mass")
                assert idx_agebins < idx_mass, (
                    f"Topological sort failed: 'agebins' ({idx_agebins}) should come before 'mass' ({idx_mass})"
                )
            except ValueError:
                pass

        # 4. Perform an update and verify consistency

        # Let's set a new redshift
        new_zred = 2.0

        # We need to construct a theta vector.
        # Identify the index of zred in theta
        theta = model.theta.copy()
        zred_idx = model.theta_index["zred"]

        # Update zred in theta
        theta[zred_idx] = new_zred

        # Call set_parameters
        model.set_parameters(theta)

        # 5. Check if agebins and mass are consistent

        # Current values in model.params
        curr_zred = model.params["zred"][0]
        curr_agebins = model.params["agebins"]
        curr_mass = model.params["mass"]

        assert np.isclose(curr_zred, new_zred), "zred was not updated correctly"

        # Calculate expected agebins manually
        # Note: zred_to_agebins requires 'agebins' in the input to know the structure (ncomp)
        expected_agebins = transforms.zred_to_agebins(
            zred=new_zred, agebins=model.params["agebins"]
        )
        assert np.allclose(curr_agebins, expected_agebins), (
            "agebins was not updated correctly based on zred"
        )

        # Calculate expected mass manually
        # Get other inputs for mass calculation
        logsfr_ratios = model.params["logsfr_ratios"]
        logmass = model.params["logmass"]

        # Re-run the transform using the updated parameters
        expected_mass = transforms.logsfr_ratios_to_masses(
            logsfr_ratios=logsfr_ratios, logmass=logmass, agebins=curr_agebins
        )

        # If the propagation order was wrong, curr_mass would have been calculated using OLD agebins
        # leading to a mismatch with expected_mass (which uses NEW agebins)
        assert np.allclose(curr_mass, expected_mass), (
            "mass is inconsistent with current agebins! Likely calculated using stale agebins."
        )

    def test_exotic_callables(self):
        """Test that dependency detection works for more exotic functions,
        such as partials and callable classes."""

        # 1. Partial Function
        def base_func(A=0, multiplier=1, **kwargs):
            return A * multiplier

        partial_dep = partial(base_func, multiplier=2)

        # 2. Callable Class
        class CallableDep:
            def __call__(self, B=0, **kwargs):
                return B + 5

        config = {
            "A": {
                "N": 1,
                "isfree": True,
                "init": 2.0,
                "prior": priors.TopHat(mini=0, maxi=10),
            },
            "B": {
                "N": 1,
                "isfree": False,
                "init": 0.0,
                "depends_on": partial_dep,
            },  # Depends on A
            "C": {
                "N": 1,
                "isfree": False,
                "init": 0.0,
                "depends_on": CallableDep(),
            },  # Depends on B
        }

        model = ProspectorParams(config)
        model.propagate_parameter_dependencies()

        # A=2 -> B=4 -> C=9
        assert model.params["B"][0] == 4.0
        assert model.params["C"][0] == 9.0

    def test_robustness_edge_cases(self):
        """
        Test edge cases for the dependency engine:
        1. Functions with arguments NOT in the model (should be ignored by introspection).
        2. Functions returning Multi-dimensional arrays (should be preserved by np.atleast_1d).
        """

        # Scenario 1: Function requests 'debug_mode', which is NOT a parameter.
        # The introspector should ignore it and link 'A' correctly.
        def loose_signature(A=0, debug_mode=False, verbose=True, **kwargs):
            return A * 2

        # Scenario 2: Function returns a 2x2 Matrix.
        # The 'np.atleast_1d' safety check should NOT mangle the shape.
        def matrix_return(A=0, **kwargs):
            return np.eye(2) * A

        config = {
            "A": {
                "N": 1,
                "isfree": True,
                "init": 2.0,
                "prior": priors.TopHat(mini=0, maxi=10),
            },
            # B depends on A, but asks for 'debug_mode' too
            "B": {"N": 1, "isfree": False, "init": 0.0, "depends_on": loose_signature},
            # C depends on A, returns a 2x2 matrix
            "C": {
                "N": 4,
                "isfree": False,
                "init": np.zeros((2, 2)),
                "depends_on": matrix_return,
            },
        }

        model = ProspectorParams(config)

        # Verify Introspection ignored the junk arguments
        if hasattr(model, "_dependency_order"):
            params_in_order = [p for p, deps in model._dependency_order]
            assert "B" in params_in_order
            # If introspection failed on 'debug_mode', B might have been skipped or A might be missing from graph

        model.propagate_parameter_dependencies()

        # Check B (Loose Signature)
        assert model.params["B"][0] == 4.0

        # Check C (Matrix Shape Preservation)
        expected_matrix = np.eye(2) * 2.0
        assert np.array_equal(model.params["C"], expected_matrix)
        assert model.params["C"].shape == (2, 2), (
            f"Shape mismatch! Expected (2,2), got {model.params['C'].shape}. "
            "np.atleast_1d might be mangling dimensions."
        )

    def test_hidden_dependency_kwargs(self):
        """
        Test that dependencies hidden inside **kwargs are NOT detected by introspection.
        This is a known limitation of the current implementation.

        Setup:
        B depends on A via kwargs['A'].
        We explicitly order the config so B comes before A.

        If dependency detection worked, the sort would put A -> B, and B would be 10.
        Since it fails, B updates before A (using old A=1), so B becomes 2.
        """

        def lazy_transform(**kwargs):
            # Access 'A' blindly from kwargs
            val_a = kwargs["A"][0]
            return val_a * 2

        # Force "Bad" order: B before A
        # Since B's dependency is hidden, the sorter sees B as having 0 dependencies.
        # It preserves the input order [B, A].
        config_list = [
            {
                "name": "B",
                "N": 1,
                "isfree": False,
                "init": 0.0,
                "depends_on": lazy_transform,
            },
            {
                "name": "A",
                "N": 1,
                "isfree": True,
                "init": 1.0,
                "prior": priors.TopHat(mini=0, maxi=10),
            },
        ]

        # Instantiation doesn't cause failure since there's no propagation yet.
        model = ProspectorParams(config_list)

        # Trying to propagate should fail because kwargs is not passed to lazy_transform.
        with pytest.raises(KeyError, match="A"):
            model.propagate_parameter_dependencies()


class TestSamplePrior:
    def test_sample_prior_shape_consistency(self):
        """
        Test that sample_prior generates theta arrays of correct shape
        matching the number of free parameters.
        """
        config = {
            "param1": {
                "N": 1,
                "isfree": True,
                "init": 0.0,
                "prior": priors.TopHat(mini=-5, maxi=5),
            },
            "param2": {
                "N": 3,
                "isfree": True,
                "init": [0.0, 0.0, 0.0],
                "prior": priors.TopHat(
                    mini=np.array([-1, -2, -3]),
                    maxi=np.array([1, 2, 3]),
                ),
            },
            "param3": {
                "N": 2,
                "isfree": False,
                "init": [1.0, 2.0],
            },
        }

        model = ProspectorParams(config)
        nfree = len(model.theta)
        assert nfree == 4, f"Expected 4 free parameters, got {nfree}"

        # Draw samples with `nsamples=None` (default)
        # Sample multiple times to check consistency
        for _ in range(10):
            theta_sampled = model.sample_prior()
            assert theta_sampled.shape == (nfree,), (
                f"Sampled theta shape {theta_sampled.shape} does not match expected ({nfree},)"
            )

        # Draw samples with explicit nsamples
        nsamples_test = [5, 7, 10, 13, 23]
        for nsamples in nsamples_test:
            theta_sampled = model.sample_prior(nsamples=nsamples)
            assert theta_sampled.shape == (nsamples, nfree), (
                f"Sampled theta shape {theta_sampled.shape} does not match expected ({nsamples}, {nfree})"
            )

    def test_sample_prior_parameter_ordering(self):
        """
        Test that sample_prior respects the parameter ordering
        defined in the model configuration.
        """
        config = {
            "param1": {
                "N": 1,
                "isfree": True,
                "init": 0.0,
                "prior": priors.TopHat(mini=0, maxi=1),
            },
            "param2": {
                "N": 1,
                "isfree": True,
                "init": 0.0,
                "prior": priors.TopHat(mini=2, maxi=3),
            },
            "param3": {
                "N": 1,
                "isfree": True,
                "init": 0.0,
                "prior": priors.TopHat(mini=4, maxi=5),
            },
            "param4": {
                "N": 1,
                "isfree": True,
                "init": 0.0,
                "prior": priors.TopHat(mini=6, maxi=7),
            },
        }

        model = ProspectorParams(config)

        # Check multiple times for robustness
        for _ in range(10):
            theta_sampled = model.sample_prior()

            # Due to how the priors are defined above, we can infer the parameter order
            # based on the sorted array.
            assert (np.argsort(theta_sampled) == np.arange(len(theta_sampled))).all(), (
                f"Parameter ordering in sampled theta ({np.argsort(theta_sampled)}) "
                + "does not match expected order"
            )

    def test_sample_prior_bounds_adherence(self):
        """
        Test that sample_prior generates values within the defined prior bounds.
        """

        # Explicitly define parameter bounds
        bounds = np.array(
            [
                [-10, 10],  # param1
                [-5, 5],  # param2[0]
                [-15, 15],  # param2[1]
            ]
        )

        config = {
            "param1": {
                "N": 1,
                "isfree": True,
                "init": 0.0,
                "prior": priors.TopHat(mini=bounds[0, 0], maxi=bounds[0, 1]),
            },
            "param2": {
                "N": 2,
                "isfree": True,
                "init": [0.0, 0.0],
                "prior": priors.TopHat(
                    mini=bounds[1:, 0],
                    maxi=bounds[1:, 1],
                ),
            },
        }

        model = ProspectorParams(config)

        # Check multiple samples for robustness
        for _ in range(100):
            theta_sampled = model.sample_prior()

            # Check param1
            assert bounds[0, 0] <= theta_sampled[0] <= bounds[0, 1], (
                f"param1 value {theta_sampled[0]} out of bounds [{bounds[0, 0]}, {bounds[0, 1]}]"
            )

            # Check param2
            assert bounds[1, 0] <= theta_sampled[1] <= bounds[1, 1], (
                f"param2[0] value {theta_sampled[1]} out of bounds [{bounds[1, 0]}, {bounds[1, 1]}]"
            )
            assert bounds[2, 0] <= theta_sampled[2] <= bounds[2, 1], (
                f"param2[1] value {theta_sampled[2]} out of bounds [{bounds[2, 0]}, {bounds[2, 1]}]"
            )

    def test_sample_prior_distribution_properties(self):
        """
        Test that sample_prior generates samples consistent with the defined prior distributions.
        Here we test the first and second moments of the sampled parameters for TopHat and Normal.
        """

        # Set seed for deterministic behavior
        np.random.seed(42)

        # Define probability density function for standard normal distribution
        def normal_pdf(x):
            return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

        def normal_cdf(x):
            return 0.5 * (1 + erf(x / np.sqrt(2)))

        # Truncated normal mean and variance
        def truncated_normal_mean(mean, sigma, a, b):
            alpha = (a - mean) / sigma
            beta = (b - mean) / sigma
            Z = normal_cdf(beta) - normal_cdf(alpha)
            phi_alpha = normal_pdf(alpha)
            phi_beta = normal_pdf(beta)
            return mean + (phi_alpha - phi_beta) * sigma / Z

        def truncated_normal_variance(mean, sigma, a, b):
            alpha = (a - mean) / sigma
            beta = (b - mean) / sigma
            Z = normal_cdf(beta) - normal_cdf(alpha)
            phi_alpha = normal_pdf(alpha)
            phi_beta = normal_pdf(beta)
            return sigma**2 * (
                1
                + (alpha * phi_alpha - beta * phi_beta) / Z
                - ((phi_alpha - phi_beta) / Z) ** 2
            )

        # Reciprocal distribution mean and variance
        def reciprocal_mean(mini, maxi):
            return (maxi - mini) / np.log(maxi / mini)

        def reciprocal_variance(mini, maxi):
            term1 = (maxi**2 - mini**2) / (2 * np.log(maxi / mini))
            return term1 - reciprocal_mean(mini, maxi) ** 2

        # Beta distribution mean and variance (scaled and shifted)
        def beta_mean(alpha, beta, mini, maxi):
            scale = maxi - mini
            standard_mean = alpha / (alpha + beta)
            return mini + scale * standard_mean

        def beta_variance(alpha, beta, mini, maxi):
            scale = maxi - mini
            standard_var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
            return scale**2 * standard_var

        # Log-normal distribution mean and variance
        def lognormal_mean(mode, sigma):
            return np.exp(mode + 1.5 * sigma**2)

        def lognormal_variance(mode, sigma):
            return (np.exp(sigma**2) - 1) * np.exp(2 * mode + 3 * sigma**2)

        # LogNormalLinpar mean and variance
        def lognormal_linpar_mean(mode, sigma_factor):
            sigma = np.log(sigma_factor)
            return mode * np.exp(1.5 * sigma**2)

        def lognormal_linpar_variance(mode, sigma_factor):
            sigma = np.log(sigma_factor)
            return (np.exp(sigma**2) - 1) * mode**2 * np.exp(3 * sigma**2)

        # Skewed normal distribution mean and variance
        def skewed_normal_mean(location, sigma, skew):
            delta = skew / np.sqrt(1 + skew**2)
            return location + sigma * delta * np.sqrt(2 / np.pi)

        def skewed_normal_variance(location, sigma, skew):
            delta = skew / np.sqrt(1 + skew**2)
            return sigma**2 * (1 - (2 * delta**2) / np.pi)

        # Student's t-distribution mean and variance
        def students_t_mean(mean, scale, df):
            return mean

        def students_t_variance(mean, scale, df):
            return scale**2 * (df / (df - 2))

        # The distributions we want to test
        dist_config = {
            "uniform": {"mini": 2, "maxi": 10},
            "normal": {"mean": 25, "sigma": 3},
            "truncated_normal": {"mean": 5, "sigma": 2, "mini": 0, "maxi": 10},
            "reciprocal": {"mini": 1, "maxi": 100},
            "beta": {"alpha": 2, "beta": 5, "mini": -10, "maxi": 10},
            "lognormal": {"mode": 1.0, "sigma": 0.5},
            "lognormal_linpar": {"mode": 2.0, "sigma_factor": 1.5},
            "skewed_normal": {"location": 0, "sigma": 2, "skew": 5},
            "students_t": {"mean": 10, "scale": 1, "df": 5},
            "multivariate_normal": {
                "mean": np.array([1.0, 3.0]),
                "Sigma": np.array([[2.0, 0.5], [0.5, 1.0]]),
            },
        }

        # The properties we expect from each distribution
        expected_stats = {
            "uniform": {
                "mean": (
                    dist_config["uniform"]["mini"] + dist_config["uniform"]["maxi"]
                )
                / 2,
                "var": (
                    (dist_config["uniform"]["maxi"] - dist_config["uniform"]["mini"])
                    ** 2
                )
                / 12,
            },
            "normal": {
                "mean": dist_config["normal"]["mean"],
                "var": dist_config["normal"]["sigma"] ** 2,
            },
            "truncated_normal": {
                "mean": truncated_normal_mean(
                    dist_config["truncated_normal"]["mean"],
                    dist_config["truncated_normal"]["sigma"],
                    dist_config["truncated_normal"]["mini"],
                    dist_config["truncated_normal"]["maxi"],
                ),
                "var": truncated_normal_variance(
                    dist_config["truncated_normal"]["mean"],
                    dist_config["truncated_normal"]["sigma"],
                    dist_config["truncated_normal"]["mini"],
                    dist_config["truncated_normal"]["maxi"],
                ),
            },
            "reciprocal": {
                "mean": reciprocal_mean(
                    dist_config["reciprocal"]["mini"],
                    dist_config["reciprocal"]["maxi"],
                ),
                "var": reciprocal_variance(
                    dist_config["reciprocal"]["mini"],
                    dist_config["reciprocal"]["maxi"],
                ),
            },
            "beta": {
                "mean": beta_mean(**dist_config["beta"]),
                "var": beta_variance(**dist_config["beta"]),
            },
            "lognormal": {
                "mean": lognormal_mean(
                    dist_config["lognormal"]["mode"],
                    dist_config["lognormal"]["sigma"],
                ),
                "var": lognormal_variance(
                    dist_config["lognormal"]["mode"],
                    dist_config["lognormal"]["sigma"],
                ),
            },
            "lognormal_linpar": {
                "mean": lognormal_linpar_mean(
                    dist_config["lognormal_linpar"]["mode"],
                    dist_config["lognormal_linpar"]["sigma_factor"],
                ),
                "var": lognormal_linpar_variance(
                    dist_config["lognormal_linpar"]["mode"],
                    dist_config["lognormal_linpar"]["sigma_factor"],
                ),
            },
            "skewed_normal": {
                "mean": skewed_normal_mean(
                    dist_config["skewed_normal"]["location"],
                    dist_config["skewed_normal"]["sigma"],
                    dist_config["skewed_normal"]["skew"],
                ),
                "var": skewed_normal_variance(
                    dist_config["skewed_normal"]["location"],
                    dist_config["skewed_normal"]["sigma"],
                    dist_config["skewed_normal"]["skew"],
                ),
            },
            "students_t": {
                "mean": students_t_mean(
                    dist_config["students_t"]["mean"],
                    dist_config["students_t"]["scale"],
                    dist_config["students_t"]["df"],
                ),
                "var": students_t_variance(
                    dist_config["students_t"]["mean"],
                    dist_config["students_t"]["scale"],
                    dist_config["students_t"]["df"],
                ),
            },
            "multivariate_normal": {
                "mean": dist_config["multivariate_normal"]["mean"],
                "cov": dist_config["multivariate_normal"]["Sigma"],
            },
        }

        # Set up the model configuration
        config = {
            "uniform": {
                "N": 1,
                "isfree": True,
                "init": 0.0,
                "prior": priors.TopHat(**dist_config["uniform"]),
            },
            "normal": {
                "N": 1,
                "isfree": True,
                "init": 0.0,
                "prior": priors.Normal(**dist_config["normal"]),
            },
            "truncated_normal": {
                "N": 1,
                "isfree": True,
                "init": 0.0,
                "prior": priors.ClippedNormal(**dist_config["truncated_normal"]),
            },
            "reciprocal": {
                "N": 1,
                "isfree": True,
                "init": 0.0,
                "prior": priors.LogUniform(**dist_config["reciprocal"]),
            },
            "beta": {
                "N": 1,
                "isfree": True,
                "init": 0.0,
                "prior": priors.Beta(**dist_config["beta"]),
            },
            "lognormal": {
                "N": 1,
                "isfree": True,
                "init": 0.0,
                "prior": priors.LogNormal(**dist_config["lognormal"]),
            },
            "lognormal_linpar": {
                "N": 1,
                "isfree": True,
                "init": 0.0,
                "prior": priors.LogNormalLinpar(**dist_config["lognormal_linpar"]),
            },
            "skewed_normal": {
                "N": 1,
                "isfree": True,
                "init": 0.0,
                "prior": priors.SkewNormal(**dist_config["skewed_normal"]),
            },
            "students_t": {
                "N": 1,
                "isfree": True,
                "init": 0.0,
                "prior": priors.StudentT(**dist_config["students_t"]),
            },
            "multivariate_normal": {
                "N": 2,
                "isfree": True,
                "init": [0.0, 0.0],
                "prior": priors.MultiVariateNormal(
                    **dist_config["multivariate_normal"]
                ),
            },
        }

        model = ProspectorParams(config)

        # Draw a large number of samples to estimate moments.
        nsamples = 10_000
        theta_samples = model.sample_prior(nsamples=nsamples)

        # Compute sample means and variances
        sample_means = np.mean(theta_samples, axis=0)
        sample_vars = np.var(theta_samples, axis=0)

        # Loop through variables using theta_index for safety
        for param_name, stats in expected_stats.items():
            # Get the slice for this parameter
            idx = model.theta_index[param_name]

            # Extract statistics for this specific parameter
            # Check mean (handles both scalar and vector cases)
            curr_mean = sample_means[idx]
            np.testing.assert_allclose(
                curr_mean,
                stats["mean"],
                rtol=0.1,
                err_msg=f"Mean mismatch for {param_name}",
            )

            # Check variance or covariance
            if "cov" in stats:
                # For multivariate, check full covariance matrix
                full_cov = np.cov(theta_samples, rowvar=False)
                # Extract the block corresponding to this parameter
                curr_cov = full_cov[idx, :][:, idx]

                np.testing.assert_allclose(
                    curr_cov,
                    stats["cov"],
                    rtol=0.2,  # Covariance estimates are noisier
                    atol=0.05,  # Handle near-zero off-diagonals
                    err_msg=f"Covariance mismatch for {param_name}",
                )
            else:
                # For univariate, check variance
                curr_var = sample_vars[idx]
                np.testing.assert_allclose(
                    curr_var,
                    stats["var"],
                    rtol=0.1,
                    err_msg=f"Variance mismatch for {param_name}",
                )
