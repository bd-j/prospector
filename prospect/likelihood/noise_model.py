# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import cho_factor, cho_solve
try:
    from statsmodels.nonparametric.kernel_density import KDEMultivariate
except(ImportError):
    pass

__all__ = ["NoiseModel", "NoiseModel2D", "FittableNoiseModel", "NoiseModelKDE"]


class NoiseModel:

    """This base class allows for 1-d noise models without any special kernels
    for covariance matrix construction, but with possibility for outliers.
    """

    f_outlier = 0
    n_sigma_outlier = 50

    def __init__(self,
                 frac_out_name="f_outlier",
                 nsigma_out_name="nsigma_outlier"):
        self.frac_out_name = frac_out_name
        self.nsigma_out_name = nsigma_out_name
        self.kernels = []

    def _available_parameters(self):
        new_pars = [(self.frac_out_name, "Fraction of data points that are outliers"),
                    (self.nsigma_out_name, "Dispersion of the outlier distribution, in units of chi")]
        for kernel in self.kernels:
            new_pars += getattr(kernel, "_available_parameters", [])
        return new_pars

    def update(self, **params):
        self.f_outlier = params.get(self.frac_out_name, 0)
        self.n_sigma_outlier = params.get(self.nsigma_out_name, 50)
        [k.update(**params) for k in self.kernels]

    def populate_vectors(self, obs, vectors={}):
        # update vectors
        vectors["mask"] = obs.mask
        vectors["uncertainty"] = obs.uncertainty
        return vectors

    def factorize_1d(self, Sigma):
        self.Sigma = Sigma
        self.factorized_Sigma = np.sqrt(Sigma)
        self.log_det = np.sum(np.log(self.Sigma))
        if self.f_outlier > 0:
            self.factorized_Sigma_outlier = self.factorized_Sigma * self.n_sigma_outlier
            #self.Sigma_outlier = self.Sigma * (self.n_sigma_outlier**2)

    def construct_covariance(self, uncertainty=[], mask=slice(None), **other_vectors):
        """Simple 1D uncertainty vector.

        Must produce:
        * self.factorized_Sigma
        * self.Sigma
        * self.log_det
        Optionally produce
        * self.factorized_Sigma_outlier
        """
        Sigma = uncertainty[mask]**2
        self.factorize_1d(Sigma)

    def lnlikelihood_1d(self, pred, data, factorized_Sigma):
        """Simple ln-likelihood for diagonal covariance matrix.

        Returns
        -------
        lnp : ndarray of shape (ndata,)
        """
        delta = data - pred
        exparg = -0.5 * (delta / factorized_Sigma)**2
        log_norm = np.log(np.sqrt(2*np.pi) * factorized_Sigma)
        return exparg - log_norm

    def lnlike(self, pred, obs, vectors={}):
        """This is the method called during fitting, after a call to self.update()

        Returns
        -------
        lnlike : float
            The total likelihood of the observed data given the prediction.
        """
        # populatate vectors used as metrics and weight functions.
        vectors = self.populate_vectors(obs)
        # Construct Sigma (and Sigma_outlier) and their sqrt/factorizations
        self.construct_covariance(**vectors)

        # Compute likelihood of inliers
        lnp = self.lnlikelihood_1d(pred[obs.mask], obs.flux[obs.mask],
                                     self.factorized_Sigma)
        if (self.f_outlier == 0.0):
            return np.sum(lnp)

        elif self.f_outlier > 0:
            # Mixture model
            lnp_out = self.lnlikelihood_1d(pred[obs.mask], obs.flux[obs.mask],
                                             self.factorized_Sigma_outlier)
            lnp_tot = np.logaddexp(lnp + np.log(1 - self.f_outlier),
                                   lnp_out + np.log(self.f_outlier))
            return np.sum(lnp_tot)
        else:
            raise ValueError(f"Outlier fraction ({self.f_outlier}) cannot be negative")


class NoiseModel2D(NoiseModel):

    """Fixed 2D covariance matrix for data
    """

    def __init__(self, Sigma=None, mask=slice(None), **kwargs):
        super(NoiseModel2D, self).__init__(**kwargs)
        if Sigma is not None:
            self.factorize_2d(Sigma[mask, :][:, mask])

    def factorize_2d(self, Sigma):
        self.Sigma = Sigma
        #self.factorized_Sigma = np.linalg.cholesky(self.Sigma)
        self.factorized_Sigma = cho_factor(self.Sigma, check_finite=True)
        self.log_det = np.log(np.prod(np.diag(self.factorized_Sigma[0])))

    def construct_covariance(self, **other_vectors):
        """Sigma is fixed at instantiation
        """
        pass

    def lnlikelihood_2d(self, y, mu, factorized_Sigma):
        """Compute the Gaussian likelihood of the data points y given the mean mu
        and factorized covariance matrix L.
        """
        n = y.shape[0]
        delta = y - mu

        # L = factorized_Sigma
        # Solve L * z = diff for z (forward substitution)
        #z = np.linalg.solve(L, delta.T)
        # Solve L^T * w = z for w (backward substitution)
        #w = np.linalg.solve(L.T, z)
        # Compute the exponent term: exp(-0.5 * w.T * w)
        #exparg = -0.5 * np.sum(w**2, axis=0)

        # faster
        exparg = -0.5 * np.dot(delta, cho_solve(factorized_Sigma, delta))

        # Normalization factor
        log_norm = 0.5 * (n * np.log(np.sqrt(2 * np.pi)) + self.log_det)

        return exparg - log_norm

    def lnlike(self, pred, obs, vectors={}):
        """This is the method called during fitting
        """
        # populate vectors used as metrics and weight functions.
        vectors = self.populate_vectors(obs)
        # Construct Sigma and factorize
        self.construct_covariance(**vectors)

        lnp = self.lnlikelihood_2d(pred[obs.mask], obs.flux[obs.mask], self.factorized_Sigma)
        if self.f_outlier > 0:
            raise ValueError("Outlier modeling not available with 2d covariances")

        return lnp


class FittableNoiseModel(NoiseModel):
    """This object allows for fitting noise properties constructed from kernels
    """

    # TODO: metric names should be the responsibility of kernels, not noise models
    def __init__(self,
                 metric_name='',
                 mask_name='mask',
                 kernels=[],
                 **kwargs):

        super(FittableNoiseModel, self).__init__(**kwargs)
        self.kernels = kernels
        self.metric_name = metric_name
        self.mask_name = mask_name

    def populate_vectors(self, obs, vectors={}):
        # update vectors
        vectors["mask"] = obs.mask
        vectors["uncertainty"] = obs.uncertainty
        vectors["wavelength"] = obs.wavelength
        vectors["flux"] = obs.flux
        if obs.kind == "photometry":
            vectors["filternames"] = obs.filternames
            vectors["phot_samples"] = obs.get("phot_samples", None)
        return vectors

    def construct_covariance(self, **vectors):
        """Construct a covariance matrix from a metric, a list of kernel
        objects, and a list of weight vectors (of same length as the metric)
        """
        metric = vectors[self.metric_name]
        mask = vectors.get(self.mask_name, slice(None))

        # 1 = uncorrelated errors, 2 = covariance matrix, >2 undefined
        ndmax = np.array([k.ndim for k in self.kernels]).max()
        Sigma = np.zeros(ndmax * [metric[mask].shape[0]])

        # loop over Kernels
        for kernel in self.kernels:
            wght = vectors.get(kernel.weight_by, None)
            Sigma += kernel(metric[mask], weights=wght[mask], ndim=ndmax)

        # Compute the things we need
        if Sigma.ndim == 1:
            self.factorize_1d(Sigma)

        elif Sigma.ndim == 2:
            self.factorize_2d(Sigma)



class NoiseModelKDE:

    def __init__(self, metric_name="phot_samples", mask_name="mask"):
        # , kernel=None, weight_by=None):
        #  self.kernel = kernel
        #  self.weight_names = weight_by
        self.metric_name = metric_name
        self.mask_name = mask_name
        self.lnl = None

    def update(self, **params):
        pass

    def compute(self, check_finite=False, **vectors):
        """Identify and cache the lnlikelihood function using the photometry
        posterior samples.  This will look for `self.metric_name` in vectors and
        use that as a set of samples to initialize a multivariate KDE
        """
        # need an assert statement, in case it is a new object
        if self.lnl is None:
            metric = vectors[self.metric_name]
            mask = vectors.get('mask', slice(None))
            samples = metric[:, mask]

            self.metric_lims = np.percentile(samples, [0, 100], axis=0)
            # KDE - use if possible
            self.lnl = KDEMultivariate(data=samples, var_type='c'*samples.shape[1]).pdf

            # Correlated normals (use if trial point is out of bounds)
            self.mu = np.mean(samples, axis=0)
            self.cov = np.cov(samples, rowvar=0)
            self.factorized_Sigma = cho_factor(self.cov, overwrite_a=True,
                                               check_finite=check_finite)
            self.log_det = 2 * np.sum(np.log(np.diag(self.factorized_Sigma[0])))
            assert np.isfinite(self.log_det)
            self.n = samples.shape[1]

    def lnlikelihood(self, phot_mu, phot_obs=None, check_finite=False, **extras):
        """Compute the ln of the likelihood

        :param phot_mu:
            Model photometry, same units as the photometry in `phot_obs`.
        :param phot_obs:
            Observed photometry, in linear flux units (i.e. maggies).
        """
        # check bounds of trial point relative to phot samples
        lo_check = np.min(phot_mu - self.metric_lims[0]) >= 0
        hi_check = np.max(phot_mu - self.metric_lims[1]) <= 0
        if lo_check * hi_check:
            return np.log(self.lnl(phot_mu))
        # use correlated normals if trial point is out of bounds
        else:
            residual = phot_mu - self.mu
            first_term = np.dot(residual, cho_solve(self.factorized_Sigma,
                                residual, check_finite=check_finite))
            lnlike = -0.5 * (first_term + self.log_det + self.n * np.log(2.*np.pi))
            return lnlike

