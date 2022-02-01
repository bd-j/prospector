import numpy as np

__all__ = ["Kernel", "Uncorrelated", "ExpSquared", "Matern", "PhotoCal", 
           "PhotSamples_MVN"]


class Kernel(object):

    def __init__(self, parnames=[], name=''):
        """
        :param parnames:
            A list of names of the kernel params, used to alias the intrinsic
            parameter names.  This way different instances of the same kernel
            can have different parameter names.
        """
        if len(parnames) == 0:
            parnames = self.kernel_params
        assert len(parnames) == len(self.kernel_params)
        self.param_alias = dict(zip(self.kernel_params, parnames))
        self.params = {}
        self.name = name

    def __repr__(self):
        return '{}({})'.format(self.__class__, self.param_alias.items())

    def update(self, **kwargs):
        """Take a dictionary of parameters, pick out the properly named
        parameters according to the alias, and put them in the param state
        dictionary.
        """
        for k in self.kernel_params:
            self.params[k] = kwargs[self.param_alias[k]]

    def __call__(self, metric, weights=None, ndim=2, **extras):
        """Return a covariance matrix, given a metric.  Optionally, multiply
        the output kernel by a weight function to induce non-stationarity.
        """
        k = self.construct_kernel(metric)
        if ndim != k.ndim:
            # Either promote to 2 dimensions or demote to 1.
            # The latter should never happen...
            k = np.diag(k)
        if weights is None:
            return k
        elif ndim == 2:
            Sigma = weights[None, :] * k * weights[:, None]
        else:
            Sigma = k * weights**2
        return Sigma


class Uncorrelated(Kernel):

    # Simple uncorrelated noise model
    ndim = 1
    kernel_params = ['amplitude']

    def construct_kernel(self, metric):
        s = metric.shape[0]
        jitter = self.params['amplitude']**2 * np.ones(s)
        if metric.ndim == 2:
            return np.diag(jitter)
        elif metric.ndim == 1:
            return jitter
        else:
            raise(NotImplementedError)


class ExpSquared(Kernel):

    ndim = 2
    npars = 2
    kernel_params = ['amplitude', 'length']

    def construct_kernel(self, metric):
        """Construct an exponential squared covariance matrix.
        """
        a, l = self.params['amplitude'], self.params['length']
        Sigma = a**2 * np.exp(-(metric[:, None] - metric[None, :])**2 / (2 * l**2))
        return Sigma


class Matern(Kernel):

    ndim = 2
    npars = 2
    kernel_params = ['amplitude', 'length']

    def construct_kernel(self, metric):
        """Construct a Matern kernel covariance matrix, for \nu=3/2.
        """
        a, l = self.params['amplitude'], self.params['length']
        Sigma = np.sqrt(3) * np.abs(metric[:, None] - metric[None, :]) / l
        Sigma = a**2 * (1 + Sigma) * np.exp(-Sigma)
        return Sigma


class PhotoCal(Kernel):

    ndim = 2
    npars = 2
    kernel_params = ['amplitude', 'filter_names']

    def construct_kernel(self, metric):
        """ This adds correlated noise in specified bands of photometry
        """
        k = np.array([f in self.params["filter_names"] for f in metric])
        K = k[:, None] * k[None, :]     # select off-diagonal elements
        return K * self.params["amplitude"]**2


class PhotSamples_MVN(Kernel):
    npars = 0
    kernel_params = []

    def __init__(self, cov, filter_names, parnames=[], name=''):

        super().__init__(parnames=parnames, name=name)
        assert cov.shape[0] == len(filter_names)
        # if no covariance, set ndim = 1
        if not np.count_nonzero(cov - np.diag(np.diagonal(cov))):
            self.ndim = 1
        else:
            self.ndim = 2
        self.cov_mat = cov
        self.params["filter_names"] = filter_names

    def construct_kernel(self, metric):
        # we pull the rows of the covariance matrix corresponding to the filters listed in `metric`
        band_index = np.array([self.params["filter_names"].index(f) for f in metric])
        return self.cov_mat[band_index[:, None], band_index]

    def __call__(self, metric, weights=None, ndim=2, **extras):
        assert weights is None, "PhotCorrelated is not meant to be weighted by anything"
        return super().__call__(metric, ndim=ndim, **extras)

