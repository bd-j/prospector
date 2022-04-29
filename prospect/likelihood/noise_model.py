# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import cho_factor, cho_solve
try:
    from statsmodels.nonparametric.kernel_density import KDEMultivariate
except(ImportError):
    pass

__all__ = ["NoiseModel", "NoiseModelCov", "NoiseModelKDE"]


class NoiseModel:

    """This class allows for 1-d covariance matrix noise models without any
    special kernels for covariance matrix construction.
    """

    f_outlier = 0
    n_sigma_outlier = 50

    def __init__(self, f_outlier_name="f_outlier", n_sigma_name="nsigma_outlier"):
        self.f_outlier_name = f_outlier_name
        self.n_sigma_name = n_sigma_name
        self.kernels = []

    def update(self, **params):
        self.f_outlier = params.get(self.f_outlier_name, 0)
        self.n_sigma_outlier = params.get(self.n_sigma_name, 50)
        [k.update(**params) for k in self.kernels]

    def lnlike(self, pred, obs, vectors={}):

        # Construct Sigma (and factorize if 2d)
        vectors = self.populate_vectors(obs)
        self.compute(**vectors)

        # Compute likelihood
        if (self.f_outlier == 0.0):
            # Let the noise model do it
            lnp = self.lnlikelihood(pred[obs.mask], obs.flux[obs.mask])
            return lnp
        elif self.f_outlier > 0:
            # Use the noise model variance, but otherwise compute on our own
            assert self.Sigma.ndim == 1, "Outlier modeling only available for uncorrelated errors"
            delta = obs.flux[obs.mask] - pred[obs.mask]
            var = self.Sigma
            lnp = -0.5*((delta**2 / var) + np.log(2*np.pi*var))
            var_bad = var * (self.n_sigma_outlier**2)
            lnp_bad = -0.5*((delta**2 / var_bad) + np.log(2*np.pi*var_bad))
            lnp_tot = np.logaddexp(lnp + np.log(1 - self.f_outlier), lnp_bad + np.log(self.f_outlier))
            return lnp_tot
        else:
            raise ValueError("f_outlier must be >= 0")

    def populate_vectors(self, obs, vectors={}):
        # update vectors
        vectors["mask"] = obs.mask
        vectors["unc"] = obs.uncertainty
        if obs.kind == "photometry":
            vectors["filternames"] = obs.filternames
            vectors["phot_samples"] = obs.get("phot_samples", None)
        return vectors

    def construct_covariance(self, unc=[], mask=slice(None), **vectors):
        self.Sigma = np.atleast_1d(unc[mask]**2)

    def compute(self, **vectors):
        """Make a boring diagonal Covariance array
        """
        self.construct_covariance(**vectors)
        self.log_det = np.sum(np.log(self.Sigma))

    def lnlikelihood(self, pred, data):
        """Simple ln-likihood for diagonal covariance matrix.
        """
        delta = data - pred
        lnp = -0.5*(np.dot(delta**2, np.log(2*np.pi) / self.Sigma) +
                    self.log_det)
        return lnp.sum()


class NoiseModelCov(NoiseModel):
    """This object allows for 1d or 2d covariance matrices constructed from kernels
    """

    def __init__(self, f_outlier_name="f_outlier", n_sigma_name="nsigma_outlier",
                 metric_name='', mask_name='mask', kernels=[], weight_by=[]):

        super().__init__(f_outlier_name=f_outlier_name,
                         n_sigma_name=n_sigma_name)
        assert len(kernels) == len(weight_by)
        self.kernels = kernels
        self.weight_names = weight_by
        self.metric_name = metric_name
        self.mask_name = mask_name

    def populate_vectors(self, vectors, obs):
        # update vectors
        vectors["mask"] = obs.mask
        vectors["wavelength"] = obs.wavelength
        vectors["unc"] = obs.uncertainty
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

        weight_vectors = self.get_weights(**vectors)
        for i, (kernel, wght) in enumerate(zip(self.kernels, weight_vectors)):
            Sigma += kernel(metric[mask], weights=wght, ndim=ndmax)
        return Sigma

    def get_weights(self, **vectors):
        """From a dictionary of vectors that give weights, pull the vectors
        that correspond to each kernel, as stored in the `weight_names`
        attribute.  A None vector will result in None weights
        """
        mask = vectors.get(self.mask_name, slice(None))
        wghts = []
        for w in self.weight_names:
            if w is None:
                wghts += [None]
            elif vectors[w] is None:
                wghts += [None]
            else:
                wghts.append(vectors[w][mask])
        return wghts

    def compute(self, check_finite=False, **vectors):
        """Build and cache the covariance matrix, and if it is 2-d factorize it
        and cache that.  Also cache ``log_det``.
        """
        self.Sigma = self.construct_covariance(**vectors)
        if self.Sigma.ndim == 1:
            self.log_det = np.sum(np.log(self.Sigma))
        else:
            self.factorized_Sigma = cho_factor(self.Sigma, overwrite_a=True, check_finite=check_finite)
            self.log_det = 2 * np.sum(np.log(np.diag(self.factorized_Sigma[0])))
            assert np.isfinite(self.log_det)

    def lnlikelihood(self, prediction, data, check_finite=False):
        """Compute the ln of the likelihood, using the current cached (and
        factorized if non-diagonal) covariance matrix.

        Parameters
        ----------
        prediction : ndarray of float
            Model flux, same units as `data`.

        data : ndarray of float
            Observed flux, in linear flux units (i.e. maggies).

        Returns
        -------
        lnlike : float
            The likelihood fo the data
        """
        residual = data - prediction
        n = len(residual)
        assert n == self.Sigma.shape[0]
        if self.Sigma.ndim == 1:
            first_term = np.dot(residual**2, 1.0 / self.Sigma)
        else:
            CinvD = cho_solve(self.factorized_Sigma, residual, check_finite=check_finite)
            first_term = np.dot(residual, CinvD)

        lnlike = -0.5 * (first_term + self.log_det + n * np.log(2.*np.pi))

        return lnlike


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

