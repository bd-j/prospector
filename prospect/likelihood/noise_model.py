import numpy as np
from scipy.linalg import cho_factor, cho_solve

__all__ = ["NoiseModel"]


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
        mask = vectors.get('mask', slice(None))

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
            if vectors[w] is None:
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

    def lnlikelihood(self, residual, check_finite=False, **extras):
        """Compute the ln of the likelihood, using the current factorized
        covariance matrix.

        :param residual: ndarray, shape (nwave,)
            Vector of residuals (y_data - mean_model).
        """
        n = len(residual)
        assert n == self.Sigma.shape[0]
        if self.Sigma.ndim > 1:
            first_term = np.dot(residual, cho_solve(self.factorized_Sigma,
                                residual, check_finite=check_finite))
        else:
            first_term = np.dot(residual**2, 1.0/self.Sigma)

        lnlike = -0.5 * (first_term + self.log_det + n * np.log(2.*np.pi))

        return lnlike
