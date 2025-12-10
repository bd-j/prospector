#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_parameter_dependencies.py

This module tests the dependency propagation logic in ProspectorParams.
It verifies that parameters are updated in the correct topological order,
ensuring that derived parameters use the most up-to-date values of their
dependencies.
"""

import numpy as np
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
            model = ProspectorParams(config)

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
