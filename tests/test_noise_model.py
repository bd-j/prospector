import numpy as np
from scipy.stats import norm
from prospect.likelihood.noise_model import NoiseModel, NoiseModelCov


def test_diagonal_gaussian_lnlike_matches_scipy():
    """Diagonal Gaussian ln-likelihood must equal the sum of scipy normal logpdfs.

    Regression test: the normalisation term N*log(2*pi) was previously written as a
    multiplicative factor on the chi^2 sum, inflating chi^2 by log(2*pi) = 1.8379.
    """
    rng = np.random.default_rng(0)
    n = 256
    sigma = rng.uniform(0.5, 2.0, n)
    data = rng.normal(0.0, sigma)
    pred = np.zeros(n)

    expected = norm.logpdf(data, loc=pred, scale=sigma).sum()

    nm = NoiseModel()
    nm.Sigma = sigma ** 2
    nm.log_det = np.sum(np.log(nm.Sigma))
    assert np.isclose(nm.lnlikelihood(pred, data), expected, rtol=1e-12)

    cov = NoiseModelCov()
    cov.Sigma = sigma ** 2
    cov.log_det = np.sum(np.log(cov.Sigma))
    assert np.isclose(cov.lnlikelihood(pred, data), expected, rtol=1e-12)