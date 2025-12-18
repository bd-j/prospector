#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_priors.py

This module provides specific unit tests for the statistical properties of
prior distributions used in Prospector. It validates that sampling from these
priors yields the expected means, variances, and covariances.
"""

import numpy as np
from scipy.special import erf
from prospect.models import ProspectorParams, priors

# ------------------------------------------------------------------------------
# 1. Math Helper Functions (Analytical Moments)
# ------------------------------------------------------------------------------


# Standard normal distribution PDF and CDF
def normal_pdf(x):
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)


def normal_cdf(x):
    return 0.5 * (1 + erf(x / np.sqrt(2)))


# Truncated normal mean and variance
def truncated_normal_mean(mean=None, sigma=None, mini=None, maxi=None):
    alpha = (mini - mean) / sigma
    beta = (maxi - mean) / sigma
    Z = normal_cdf(beta) - normal_cdf(alpha)
    phi_alpha = normal_pdf(alpha)
    phi_beta = normal_pdf(beta)
    return mean + (phi_alpha - phi_beta) * sigma / Z


def truncated_normal_variance(mean=None, sigma=None, mini=None, maxi=None):
    alpha = (mini - mean) / sigma
    beta = (maxi - mean) / sigma
    Z = normal_cdf(beta) - normal_cdf(alpha)
    phi_alpha = normal_pdf(alpha)
    phi_beta = normal_pdf(beta)
    return sigma**2 * (
        1
        + (alpha * phi_alpha - beta * phi_beta) / Z
        - ((phi_alpha - phi_beta) / Z) ** 2
    )


# Reciprocal distribution mean and variance
def reciprocal_mean(mini=None, maxi=None):
    return (maxi - mini) / np.log(maxi / mini)


def reciprocal_variance(mini=None, maxi=None):
    term1 = (maxi**2 - mini**2) / (2 * np.log(maxi / mini))
    return term1 - reciprocal_mean(mini, maxi) ** 2


# Beta distribution mean and variance (scaled and shifted)
def beta_mean(alpha=None, beta=None, mini=None, maxi=None):
    scale = maxi - mini
    standard_mean = alpha / (alpha + beta)
    return mini + scale * standard_mean


def beta_variance(alpha=None, beta=None, mini=None, maxi=None):
    scale = maxi - mini
    standard_var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
    return scale**2 * standard_var


# Log-normal distribution mean and variance
def lognormal_mean(mode=None, sigma=None):
    return np.exp(mode + 1.5 * sigma**2)


def lognormal_variance(mode=None, sigma=None):
    return (np.exp(sigma**2) - 1) * np.exp(2 * mode + 3 * sigma**2)


# LogNormalLinpar mean and variance
def lognormal_linpar_mean(mode=None, sigma_factor=None):
    sigma = np.log(sigma_factor)
    return mode * np.exp(1.5 * sigma**2)


def lognormal_linpar_variance(mode=None, sigma_factor=None):
    sigma = np.log(sigma_factor)
    return (np.exp(sigma**2) - 1) * mode**2 * np.exp(3 * sigma**2)


# Skewed normal distribution mean and variance
def skewed_normal_mean(location=None, sigma=None, skew=None):
    delta = skew / np.sqrt(1 + skew**2)
    return location + sigma * delta * np.sqrt(2 / np.pi)


def skewed_normal_variance(location=None, sigma=None, skew=None):
    delta = skew / np.sqrt(1 + skew**2)
    return sigma**2 * (1 - (2 * delta**2) / np.pi)


# Student's t-distribution mean and variance
def students_t_mean(mean=None, scale=None, df=None):
    return mean


def students_t_variance(mean=None, scale=None, df=None):
    return scale**2 * (df / (df - 2))


# ------------------------------------------------------------------------------
# 2. Test Infrastructure
# ------------------------------------------------------------------------------


def check_prior_moments(
    prior, expected_mean, expected_var, expected_cov=None, nsamples=10_000
):
    """
    Helper function to sample from a prior and verify statistical moments.
    """
    # Determine N based on whether mean is scalar or array
    mean_arr = np.atleast_1d(expected_mean)
    N = len(mean_arr)

    # 1. Setup Model
    config = {
        "p": {
            "N": N,
            "isfree": True,
            "init": expected_mean,  # Init doesn't matter for sampling
            "prior": prior,
        }
    }
    model = ProspectorParams(config)

    # 2. Sample
    np.random.seed(42)
    samples = model.sample_prior(nsamples=nsamples)

    # 3. Verify Mean
    sample_mean = np.mean(samples, axis=0)

    # Handle scalar vs vector output from sample_mean
    if N == 1:
        # Squeeze to scalar for comparison if N=1
        sample_mean = sample_mean[0]

    np.testing.assert_allclose(
        sample_mean,
        expected_mean,
        rtol=0.1,
        err_msg=f"Mean mismatch for {prior.__class__.__name__}",
    )

    # 4. Verify Variance / Covariance
    if expected_cov is not None:
        # Multivariate check
        full_cov = np.cov(samples, rowvar=False)
        np.testing.assert_allclose(
            full_cov,
            expected_cov,
            rtol=0.2,
            atol=0.05,
            err_msg=f"Covariance mismatch for {prior.__class__.__name__}",
        )
    else:
        # Univariate check
        sample_var = np.var(samples, axis=0)
        if N == 1:
            sample_var = sample_var[0]

        np.testing.assert_allclose(
            sample_var,
            expected_var,
            rtol=0.1,
            err_msg=f"Variance mismatch for {prior.__class__.__name__}",
        )


# ------------------------------------------------------------------------------
# 4. Individual Tests
# ------------------------------------------------------------------------------


def test_uniform_prior():
    mini, maxi = 2, 10
    prior = priors.TopHat(mini=mini, maxi=maxi)

    exp_mean = (mini + maxi) / 2
    exp_var = ((maxi - mini) ** 2) / 12

    check_prior_moments(prior, exp_mean, exp_var)


def test_normal_prior():
    mean, sigma = 25, 3
    prior = priors.Normal(mean=mean, sigma=sigma)

    exp_mean = mean
    exp_var = sigma**2

    check_prior_moments(prior, exp_mean, exp_var)


def test_truncated_normal_prior():
    params = {"mean": 5, "sigma": 2, "mini": 0, "maxi": 10}
    prior = priors.ClippedNormal(**params)

    exp_mean = truncated_normal_mean(**params)
    exp_var = truncated_normal_variance(**params)

    check_prior_moments(prior, exp_mean, exp_var)


def test_reciprocal_prior():
    params = {"mini": 1, "maxi": 100}
    prior = priors.LogUniform(**params)

    exp_mean = reciprocal_mean(**params)
    exp_var = reciprocal_variance(**params)

    check_prior_moments(prior, exp_mean, exp_var)


def test_beta_prior():
    params = {"alpha": 2, "beta": 5, "mini": -10, "maxi": 10}
    prior = priors.Beta(**params)

    exp_mean = beta_mean(**params)
    exp_var = beta_variance(**params)

    check_prior_moments(prior, exp_mean, exp_var)


def test_lognormal_prior():
    params = {"mode": 1.0, "sigma": 0.5}
    prior = priors.LogNormal(**params)

    exp_mean = lognormal_mean(**params)
    exp_var = lognormal_variance(**params)

    check_prior_moments(prior, exp_mean, exp_var)


def test_lognormal_linpar_prior():
    params = {"mode": 2.0, "sigma_factor": 1.5}
    prior = priors.LogNormalLinpar(**params)

    exp_mean = lognormal_linpar_mean(**params)
    exp_var = lognormal_linpar_variance(**params)

    check_prior_moments(prior, exp_mean, exp_var)


def test_skewed_normal_prior():
    params = {"location": 0, "sigma": 2, "skew": 5}
    prior = priors.SkewNormal(**params)

    exp_mean = skewed_normal_mean(**params)
    exp_var = skewed_normal_variance(**params)

    check_prior_moments(prior, exp_mean, exp_var)


def test_students_t_prior():
    params = {"mean": 10, "scale": 1, "df": 5}
    prior = priors.StudentT(**params)

    exp_mean = students_t_mean(**params)
    exp_var = students_t_variance(**params)

    check_prior_moments(prior, exp_mean, exp_var)


def test_multivariate_normal_prior():
    mean = np.array([1.0, 3.0])
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    prior = priors.MultiVariateNormal(mean=mean, Sigma=cov)

    # Pass 'cov' as the 4th argument to trigger the multivariate check in the helper
    check_prior_moments(prior, mean, expected_var=None, expected_cov=cov)
