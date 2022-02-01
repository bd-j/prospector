import numpy as np
from scipy.linalg import cho_factor, cho_solve
try:
    from statsmodels.nonparametric.kernel_density import KDEMultivariate
except(ImportError):
    pass

__all__ = ["NoiseModel", "NoiseModelKDE"]


class NoiseModel(object):

    def __init__(self, metric_name='', mask_name='mask', kernels=[],
                 weight_by=[]):
        assert len(kernels) == len(weight_by)
        self.kernels = kernels
        self.weight_names = weight_by
        self.metric_name = metric_name
        self.mask_name = mask_name

    def update(self, **params):
        [k.update(**params) for k in self.kernels]

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
        if self.Sigma.ndim > 1:
            self.factorized_Sigma = cho_factor(self.Sigma, overwrite_a=True,
                                               check_finite=check_finite)
            self.log_det = 2 * np.sum(np.log(np.diag(self.factorized_Sigma[0])))
            assert np.isfinite(self.log_det)
        else:
            self.log_det = np.sum(np.log(self.Sigma))

    def lnlikelihood(self, phot_mu, phot_obs, check_finite=False, **extras):
        """Compute the ln of the likelihood, using the current factorized
        covariance matrix.

        :param phot_mu:
            Model photometry, same units as the photometry in `phot_obs`.
        :param phot_obs:
            Observed photometry, in linear flux units (i.e. maggies).
        """
        residual = phot_obs - phot_mu
        n = len(residual)
        assert n == self.Sigma.shape[0]
        if self.Sigma.ndim > 1:
            first_term = np.dot(residual, cho_solve(self.factorized_Sigma,
                                residual, check_finite=check_finite))
        else:
            first_term = np.dot(residual**2, 1.0/self.Sigma)

        lnlike = -0.5 * (first_term + self.log_det + n * np.log(2.*np.pi))

        return lnlike


class NoiseModelKDE(object):

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

