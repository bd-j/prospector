# Module containg various functions to be used as priors
import numpy as np

__all__ = ["normal", "tophat", "normal_clipped", "positive",
           "lognormal", "logarithmic",
           "plotting_range"]


def zeros(theta, **extras):
    return np.zeros_like(theta)


def positive(theta, **extras):
    """
    Require that a parameter be positive.
    """
    p = 1.0 * np.zeros_like(theta)
    n = theta < 0
    p[n] = -np.infty
    return p


def tophat(theta, mini=0.0, maxi=1.0, **extras):
    """
    A simple tophat function.  Input can be scalar or matched vectors
    """
    lnp = 1.0 * np.zeros_like(theta)
    n = (theta < mini) | (theta > maxi)
    lnp[n] = -np.infty
    return lnp


def normal(theta, mean=0.0, sigma=1.0, **extras):
    """
    A simple gaussian.  should make sure it can be vectorized.
    """
    return np.log((2*np.pi)**(-0.5)/sigma) - (theta - mean)**2/(2*sigma**2)


def normal_clipped(theta, mean=0.0, sigma=1.0, mini=0.0, maxi=1.0, **extras):
    """
    A clipped gaussian.
    """
    lnp = np.log((2*np.pi)**(-0.5)/sigma) - (theta - mean)**2/(2*sigma**2)
    n = (theta < mini) | (theta > maxi)
    lnp[n] = -np.infty

    return lnp


def lognormal(theta, log_mean=0.0, sigma=1.0, **extras):
    """
    A lognormal  gaussian.  should make sure it can be vectorized.
    """
    if np.all(theta > 0):
        return ( np.log((2*np.pi)**(-0.5)/(theta*sigma)) -
                (np.log(theta) - log_mean)**2/(2*sigma**2) )
    else:
        return np.zeros(np.size(theta))-np.infty


def logarithmic(theta, mini=0.0, maxi=np.inf, **extras):
    """A logarithmic (1/x) prior, with optional bounds.
    """
    lnp = -np.log(theta)
    n = (theta < mini) | (theta > maxi)
    lnp[n] = -np.infty
    return lnp


def plotting_range(prior_args):
    if 'mini' in prior_args:
        return prior_args['mini'], prior_args['maxi']
    if 'log_mean' in prior_args:
        mini = np.atleast_1d(prior_args['log_mean']) - 10 * np.array(prior_args['sigma'])
        maxi = np.atleast_1d(prior_args['log_mean']) + 10 * np.array(prior_args['sigma'])
        return np.exp(mini).tolist(), np.exp(maxi).tolist()
    if 'mean' in prior_args:
        mini = np.array(prior_args['mean']) - 10 * np.array(prior_args['sigma'])
        maxi = np.array(prior_args['mean']) + 10 * np.array(prior_args['sigma'])
        return mini.tolist(), maxi.tolist()


class Prior(object):
    """Encapsulate the priors in an object.  Each prior should have a
    distribution name and optional parameters specifying scale and location
    (e.g. min/max or mean/sigma).  These can be aliased. When called, the
    argument should be a variable and it should return the ln-prior-probability
    of that value.

    Should be able to sample from the prior, and to get the gradient of the
    prior at any variable value.  Methods should also be avilable to give a
    useful plotting range and, if there are bounds, to return them.
    """

    def __init__(self, parnames=[], name='', **kwargs):
        """
        :param parnames:
            A list of names of the parameters params, used to alias the intrinsic
            parameter names.  This way different instances of the same Prior
            can (must) have different parameter names.
        """
        if len(parnames) == 0:
            parnames = self.prior_params
        assert len(parnames) == len(self.prior_params)
        self.alias = dict(zip(self.prior_params, parnames))
        self.params = {}

        self.name = name
        self.update(**kwargs)

    def update(self, **kwargs):
        """
        """
        for k in self.prior_params:
            try:
                self.params[k] = kwargs[self.alias[k]]
            except(KeyError):
                pass

    def __call__(self, x, **kwargs):
        """Compute the value of the probability desnity function at x and
        return the ln of that.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        p = self.distribution.pdf(x, loc=self.loc, scale=self.scale)
        return np.log(p)
        
    def sample(self, nsample, **kwargs):
        """Draw nsample values from the prior distribution.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.distribution.rvs(size=nsample, loc=self.loc, scale=self.scale)

    def unit_transform(self, x, **kwargs):
        """Go from a value of the CDF (between 0 and 1) to the corresponding
        parameter value.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.distribution.ppf(x, loc=self.loc, scale=self.scale)

    def inverse_unit_transform(self, x, **kwargs):
        """Go from the parameter value to the unit coordinate using the cdf.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.distribution.cdf(x, loc=self.loc, scale=self.scale)
        
        
    def gradient(self, theta):
        raise(NotImplementedError)

    @property
    def range(self):
        raise(NotImplementedError)

    @property
    def bounds(self):
        raise(NotImplementedError)

    def serialize(self):
        raise(NotImplementedError)


class TopHat(Prior):

    prior_params = ['mini', 'maxi']
    distribution = scipy.stats.uniform

    @property
    def scale(self):
        return self.params['maxi'] - self.params['mini']

    @property
    def loc(self):
        return self.params['mini']

    @property
    def range(self):
        return (self.params['mini'], self.params['maxi'])

    def bounds(self, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range


class Normal(Prior):

    prior_params = ['mean', 'sigma']
    distribution = scipy.stats.norm

    @property
    def scale(self):
        return self.params['sigma']

    @property
    def loc(self):
        return self.params['mean']

    @property
    def range(self):
        nsig = 4
        return (self.params['mean'] - nsig * self.params['sigma'],
                self.params['mean'] + self.params['sigma'])

    def bounds(self, **kwargs):
        #if len(kwargs) > 0:
        #    self.update(**kwargs)
        return (-np.inf, np.inf)
