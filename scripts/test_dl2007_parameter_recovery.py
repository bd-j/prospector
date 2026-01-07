#!/usr/bin/env python
"""
Test parameter recovery: FSPS DL2007 vs CIGALE DL2007.

This script:
1. Generates a mock spectrum using FSPS with known dust parameters
2. Fits it with FSPS-based dust emission (standard Prospector)
3. Fits it with CIGALE-based dust emission (DL2007CigaleSSPBasis)
4. Compares recovered parameters

Usage:
    python test_dl2007_parameter_recovery.py
"""

import numpy as np
import sys
import os
from functools import partial

# Add project to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

import warnings
warnings.filterwarnings('ignore')

# Check for required packages
try:
    import fsps
    import prospect
    from prospect.models import SpecModel
    from prospect.models.templates import TemplateLibrary
    from prospect.sources import FastStepBasis
    from prospect.observation import Photometry
    from prospect.fitting import lnprobfn
    from sedpy.observate import load_filters, getSED
    HAS_DEPS = True
except ImportError as e:
    print(f"Missing dependency: {e}")
    HAS_DEPS = False
    sys.exit(1)

# Import CIGALE-based source
from prospect.sources import DL2007CigaleSSPBasis


def build_mock_obs(wave, spectrum, zred, uncertainty_frac=0.03):
    """Build a mock observation using Photometry class."""

    # Select filters that span UV to FIR
    filter_names = [
        'sdss_u0', 'sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0',  # Optical
        'twomass_J', 'twomass_H', 'twomass_Ks',  # NIR
        'wise_w1', 'wise_w2', 'wise_w3', 'wise_w4',  # MIR
        'herschel_pacs_100', 'herschel_pacs_160',  # FIR
        'herschel_spire_250',  # Sub-mm
    ]

    try:
        filters = load_filters(filter_names)
    except Exception as e:
        print(f"Could not load all filters: {e}")
        print("Using subset of filters...")
        filter_names = ['sdss_g0', 'sdss_r0', 'sdss_i0', 'wise_w3', 'wise_w4']
        filters = load_filters(filter_names)

    # Compute photometry from spectrum
    mags = getSED(wave, spectrum, filterlist=filters)
    maggies = 10**(-0.4 * mags)

    # Add noise
    maggies_unc = uncertainty_frac * maggies
    np.random.seed(42)
    maggies_obs = maggies + np.random.normal(0, maggies_unc)

    # Create Photometry observation object
    obs = Photometry(
        filters=filters,
        flux=maggies_obs,
        uncertainty=maggies_unc,
        redshift=zred,
    )

    return obs, filters


def build_fsps_model(free_dust=True):
    """Build model using FSPS dust emission with non-parametric SFH."""
    # Use continuity SFH (non-parametric) since DL2007CigaleSSPBasis uses FastStepBasis
    model_params = TemplateLibrary["continuity_sfh"]
    model_params.update(TemplateLibrary["dust_emission"])

    # Fix most parameters
    model_params["zred"]["init"] = 0.1
    model_params["zred"]["isfree"] = False

    model_params["logzsol"]["init"] = 0.0
    model_params["logzsol"]["isfree"] = False

    model_params["dust2"]["init"] = 0.3
    model_params["dust2"]["isfree"] = False

    # Fix SFH parameters
    model_params["logmass"]["init"] = 10.0
    model_params["logmass"]["isfree"] = False

    # Fix continuity SFH parameters (logsfr_ratios)
    if "logsfr_ratios" in model_params:
        model_params["logsfr_ratios"]["isfree"] = False

    # Dust emission parameters - free or fixed
    if free_dust:
        model_params["duste_qpah"]["isfree"] = True
        model_params["duste_qpah"]["init"] = 2.0
        model_params["duste_qpah"]["prior"] = prospect.models.priors.TopHat(mini=0.5, maxi=5.0)

        model_params["duste_umin"]["isfree"] = True
        model_params["duste_umin"]["init"] = 1.0
        model_params["duste_umin"]["prior"] = prospect.models.priors.TopHat(mini=0.1, maxi=25.0)

        model_params["duste_gamma"]["isfree"] = True
        model_params["duste_gamma"]["init"] = 0.1
        model_params["duste_gamma"]["prior"] = prospect.models.priors.TopHat(mini=0.001, maxi=0.5)

    return SpecModel(model_params)


def build_cigale_model(free_dust=True):
    """Build model using CIGALE DL2007 dust emission with non-parametric SFH."""
    # Use continuity SFH to match DL2007CigaleSSPBasis (which uses FastStepBasis)
    model_params = TemplateLibrary["continuity_sfh"]
    model_params.update(TemplateLibrary["dl2007_cigale_dust_emission"])

    # Fix most parameters (same as FSPS model)
    model_params["zred"]["init"] = 0.1
    model_params["zred"]["isfree"] = False

    model_params["logzsol"]["init"] = 0.0
    model_params["logzsol"]["isfree"] = False

    model_params["dust2"]["init"] = 0.3
    model_params["dust2"]["isfree"] = False

    # Fix SFH parameters
    model_params["logmass"]["init"] = 10.0
    model_params["logmass"]["isfree"] = False

    # Fix continuity SFH parameters (logsfr_ratios)
    if "logsfr_ratios" in model_params:
        model_params["logsfr_ratios"]["isfree"] = False

    # Dust emission parameters - free or fixed
    if free_dust:
        model_params["dl2007_cigale_qpah"]["isfree"] = True
        model_params["dl2007_cigale_qpah"]["init"] = 2.0
        model_params["dl2007_cigale_qpah"]["prior"] = prospect.models.priors.TopHat(mini=0.5, maxi=5.0)

        model_params["dl2007_cigale_umin"]["isfree"] = True
        model_params["dl2007_cigale_umin"]["init"] = 1.0
        model_params["dl2007_cigale_umin"]["prior"] = prospect.models.priors.TopHat(mini=0.1, maxi=25.0)

        model_params["dl2007_cigale_gamma"]["isfree"] = True
        model_params["dl2007_cigale_gamma"]["init"] = 0.1
        model_params["dl2007_cigale_gamma"]["prior"] = prospect.models.priors.TopHat(mini=0.001, maxi=0.5)

        # Fix umax to match FSPS default behavior
        model_params["dl2007_cigale_umax"]["init"] = 1e6
        model_params["dl2007_cigale_umax"]["isfree"] = False

    return SpecModel(model_params)


def run_optimization(observations, model, sps):
    """Run optimization to find best-fit parameters."""
    from scipy.optimize import minimize, differential_evolution

    # Get initial theta and bounds from model
    theta_init = model.theta.copy()

    # Get bounds from priors
    bounds = []
    for i, p in enumerate(model.free_params):
        prior = model.config_dict[p].get('prior', None)
        if prior is not None and hasattr(prior, 'params'):
            if hasattr(prior, 'mini') and hasattr(prior, 'maxi'):
                bounds.append((prior.mini, prior.maxi))
            elif 'mini' in prior.params and 'maxi' in prior.params:
                bounds.append((prior.params['mini'], prior.params['maxi']))
            else:
                bounds.append((None, None))
        else:
            bounds.append((None, None))

    # Use Prospector's lnprobfn with negative=True for minimization
    neg_lnprob = partial(lnprobfn, model=model, observations=observations, sps=sps, negative=True)

    # Initial test
    init_val = neg_lnprob(theta_init)
    print(f"  Initial theta: {theta_init}")
    print(f"  Initial -ln(prob): {init_val:.2f}")
    print(f"  Bounds: {bounds}")

    # Debug: check model prediction vs observations
    model.set_parameters(theta_init)
    predictions, _ = model.predict(theta_init, observations=observations, sps=sps)
    print(f"  Mock photometry (first 5): {observations[0].flux[:5]}")
    print(f"  Model prediction (first 5): {predictions[0][:5]}")

    # Run bounded optimization (L-BFGS-B respects bounds)
    result = minimize(neg_lnprob, theta_init, method='L-BFGS-B',
                      bounds=bounds,
                      options={'maxiter': 500, 'ftol': 1e-6})

    print(f"  Optimization converged: {result.success}")
    print(f"  Final theta: {result.x}")

    return result.x, -result.fun


def main():
    print("=" * 70)
    print("DL2007 Parameter Recovery Test")
    print("FSPS vs CIGALE Implementation")
    print("=" * 70)

    # True parameters for mock spectrum
    true_params = {
        'qpah': 2.5,
        'umin': 1.0,
        'gamma': 0.1,
    }

    print(f"\nTrue dust parameters:")
    print(f"  qpah = {true_params['qpah']}")
    print(f"  umin = {true_params['umin']}")
    print(f"  gamma = {true_params['gamma']}")

    # Generate mock spectrum using FSPS with FastStepBasis (non-parametric SFH)
    print("\n--- Generating mock spectrum with FSPS DL2007 ---")
    sps_fsps = FastStepBasis(zcontinuous=1)

    # Build model for mock data generation
    model_fsps_gen = build_fsps_model(free_dust=False)

    # Set true parameters
    model_fsps_gen.params["duste_qpah"] = true_params['qpah']
    model_fsps_gen.params["duste_umin"] = true_params['umin']
    model_fsps_gen.params["duste_gamma"] = true_params['gamma']

    # Generate spectrum using the model predict workflow
    zred = float(np.atleast_1d(model_fsps_gen.params['zred'])[0])

    # Create a dummy photometry observation for prediction
    filter_names = [
        'sdss_u0', 'sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0',
        'twomass_J', 'twomass_H', 'twomass_Ks',
        'wise_w1', 'wise_w2', 'wise_w3', 'wise_w4',
        'herschel_pacs_100', 'herschel_pacs_160',
        'herschel_spire_250',
    ]
    filters = load_filters(filter_names)

    # Use model.predict to generate mock photometry properly
    # This ensures the normalization is consistent
    obs_dummy = Photometry(
        filters=filters,
        flux=np.ones(len(filters)),
        uncertainty=0.03 * np.ones(len(filters)),
        redshift=zred,
    )

    # Get model predictions (which handles mass normalization internally)
    predictions, mfrac = model_fsps_gen.predict(model_fsps_gen.theta, observations=[obs_dummy], sps=sps_fsps)
    maggies_true = predictions[0]

    wave = sps_fsps.wavelengths
    wave_obs = wave * (1 + zred)

    logmass = float(np.atleast_1d(model_fsps_gen.params['logmass'])[0])
    mass = 10**logmass

    print(f"  Generated spectrum with {len(wave)} wavelength points")
    print(f"  Mass: {mass:.2e} Msun")
    print(f"  True photometry (first 3): {maggies_true[:3]}")

    # Build mock observation by adding noise to true photometry
    print("\n--- Building mock photometry ---")
    uncertainty_frac = 0.03
    maggies_unc = uncertainty_frac * maggies_true
    np.random.seed(42)
    maggies_obs = maggies_true + np.random.normal(0, maggies_unc)

    obs_phot = Photometry(
        filters=filters,
        flux=maggies_obs,
        uncertainty=maggies_unc,
        redshift=zred,
    )
    observations = [obs_phot]

    print(f"  Using {len(filters)} photometric bands")
    for i, f in enumerate(filters):
        print(f"    {f.name}: {obs_phot.flux[i]:.4e} +/- {obs_phot.uncertainty[i]:.4e}")

    # Fit with FSPS model
    print("\n--- Fitting with FSPS DL2007 ---")
    model_fsps_fit = build_fsps_model(free_dust=True)
    theta_fsps, lnp_fsps = run_optimization(observations, model_fsps_fit, sps_fsps)

    # Extract best-fit parameters
    model_fsps_fit.set_parameters(theta_fsps)
    fsps_results = {
        'qpah': float(np.atleast_1d(model_fsps_fit.params['duste_qpah'])[0]),
        'umin': float(np.atleast_1d(model_fsps_fit.params['duste_umin'])[0]),
        'gamma': float(np.atleast_1d(model_fsps_fit.params['duste_gamma'])[0]),
    }

    print(f"  Best-fit parameters:")
    print(f"    qpah = {fsps_results['qpah']:.3f} (true: {true_params['qpah']})")
    print(f"    umin = {fsps_results['umin']:.3f} (true: {true_params['umin']})")
    print(f"    gamma = {fsps_results['gamma']:.4f} (true: {true_params['gamma']})")
    print(f"  ln(prob) = {lnp_fsps:.2f}")

    # Fit with CIGALE model
    print("\n--- Fitting with CIGALE DL2007 ---")
    sps_cigale = DL2007CigaleSSPBasis(zcontinuous=1)
    model_cigale_fit = build_cigale_model(free_dust=True)
    theta_cigale, lnp_cigale = run_optimization(observations, model_cigale_fit, sps_cigale)

    # Extract best-fit parameters
    model_cigale_fit.set_parameters(theta_cigale)
    cigale_results = {
        'qpah': float(np.atleast_1d(model_cigale_fit.params['dl2007_cigale_qpah'])[0]),
        'umin': float(np.atleast_1d(model_cigale_fit.params['dl2007_cigale_umin'])[0]),
        'gamma': float(np.atleast_1d(model_cigale_fit.params['dl2007_cigale_gamma'])[0]),
    }

    print(f"  Best-fit parameters:")
    print(f"    qpah = {cigale_results['qpah']:.3f} (true: {true_params['qpah']})")
    print(f"    umin = {cigale_results['umin']:.3f} (true: {true_params['umin']})")
    print(f"    gamma = {cigale_results['gamma']:.4f} (true: {true_params['gamma']})")
    print(f"  ln(prob) = {lnp_cigale:.2f}")

    # Compare results
    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    print(f"\n{'Parameter':<10} {'True':<10} {'FSPS':<10} {'CIGALE':<10} {'FSPS err':<10} {'CIGALE err':<10}")
    print("-" * 60)

    for param in ['qpah', 'umin', 'gamma']:
        true_val = true_params[param]
        fsps_val = fsps_results[param]
        cigale_val = cigale_results[param]
        fsps_err = (fsps_val - true_val) / true_val * 100
        cigale_err = (cigale_val - true_val) / true_val * 100
        print(f"{param:<10} {true_val:<10.3f} {fsps_val:<10.3f} {cigale_val:<10.3f} {fsps_err:<+10.1f}% {cigale_err:<+10.1f}%")

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
