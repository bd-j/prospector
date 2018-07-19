# Module containg various functions (or objects) to be used as priors.
# These return the ln-prior-probability
import numpy as np
import scipy.stats

__all__ = ["plotting_range",
           "Prior", "TopHat", "Normal", "ClippedNormal",
           "LogNormal", "LogUniform", "Beta"]


def zeros(theta, **extras):
    return np.zeros_like(theta)


def tophat(theta, mini=0.0, maxi=1.0, **extras):
    """A simple tophat function.  Input can be scalar or matched vectors
    """
    lnp = 1.0 * np.zeros_like(theta)
    n = (theta < mini) | (theta > maxi)
    lnp[n] = -np.infty
    return lnp


def normal(theta, mean=0.0, sigma=1.0, **extras):
    """A simple gaussian.  should make sure it can be vectorized.
    """
    return np.log((2*np.pi)**(-0.5)/sigma) - (theta - mean)**2/(2*sigma**2)


def normal_clipped(theta, mean=0.0, sigma=1.0, mini=0.0, maxi=1.0, **extras):
    """A clipped gaussian.
    """
    lnp = np.log((2*np.pi)**(-0.5)/sigma) - (theta - mean)**2/(2*sigma**2)
    n = (theta < mini) | (theta > maxi)
    lnp[n] = -np.infty

    return lnp


def lognormal(theta, log_mean=0.0, sigma=1.0, **extras):
    """A lognormal  gaussian.  should make sure it can be vectorized.
    """
    if np.all(theta > 0):
        return (np.log((2*np.pi)**(-0.5)/(theta*sigma)) -
                (np.log(theta) - log_mean)**2/(2*sigma**2))
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
        mini = (np.atleast_1d(prior_args['log_mean']) -
                10 * np.array(prior_args['sigma']))
        maxi = (np.atleast_1d(prior_args['log_mean']) +
                10 * np.array(prior_args['sigma']))
        return np.exp(mini).tolist(), np.exp(maxi).tolist()
    if 'mean' in prior_args:
        mini = (np.array(prior_args['mean']) -
                10 * np.array(prior_args['sigma']))
        maxi = (np.array(prior_args['mean']) +
                10 * np.array(prior_args['sigma']))
        return mini.tolist(), maxi.tolist()


class Prior(object):
    """Encapsulate the priors in an object.  Each prior should have a
    distribution name and optional parameters specifying scale and location
    (e.g. min/max or mean/sigma).  These can be aliased at instantiation using
    the ``parnames`` keyword. When called, the argument should be a variable
    and the object should return the ln-prior-probability of that value.

    .. code-block:: python
        
        ln_prior_prob = Prior()(value)

    Should be able to sample from the prior, and to get the gradient of the
    prior at any variable value.  Methods should also be avilable to give a
    useful plotting range and, if there are bounds, to return them.

    :param parnames:
        A list of names of the parameters, used to alias the intrinsic
        parameter names.  This way different instances of the same Prior can
        have different parameter names, in case they are being fit for....
    """

    def __init__(self, parnames=[], name='', **kwargs):
        """Constructor.

        :param parnames:
            A list of names of the parameters, used to alias the intrinsic
            parameter names.  This way different instances of the same Prior
            can have different parameter names, in case they are being fit for....
        """
        if len(parnames) == 0:
            parnames = self.prior_params
        assert len(parnames) == len(self.prior_params)
        self.alias = dict(zip(self.prior_params, parnames))
        self.params = {}

        self.name = name
        self.update(**kwargs)

    def __repr__(self):
        argstring = ['{}={}'.format(k, v) for k, v in list(self.params.items())] 
        return '{}({})'.format(self.__class__, ",".join(argstring))

    def update(self, **kwargs):
        """Update `params` values using alias.
        """
        for k in self.prior_params:
            try:
                self.params[k] = kwargs[self.alias[k]]
            except(KeyError):
                pass
        # FIXME: Should add a check for unexpected kwargs.

    def __len__(self):
        """The length is set by the maximum size of any of the prior_params.
        Note that the prior params must therefore be scalar of same length as
        the maximum size of any of the parameters.  This is not checked.
        """
        return max([np.size(self.params.get(k, 1)) for k in self.prior_params])

    def __call__(self, x, **kwargs):
        """Compute the value of the probability desnity function at x and
        return the ln of that.

        :param x:
            Value of the parameter, scalar or iterable of same length as the
            Prior object.

        :param kwargs: optional
            All extra keyword arguments are sued to update the `prior_params`.

        :returns lnp:
            The natural log of the prior probability at x, scalar or ndarray of
            same length as the prior object.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        p = self.distribution.pdf(x, *self.args,
                                  loc=self.loc, scale=self.scale)
        with np.errstate(invalid='ignore'):
            lnp = np.log(p)
        return lnp

    def sample(self, nsample=None, **kwargs):
        """Draw a sample from the prior distribution.

        :param nsample: (optional)
            Unused
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.distribution.rvs(*self.args, size=len(self),
                                     loc=self.loc, scale=self.scale)

    def unit_transform(self, x, **kwargs):
        """Go from a value of the CDF (between 0 and 1) to the corresponding
        parameter value.

        :param x:
            A scalar or vector of same length as the Prior with values between
            zero and one corresponding to the value of the CDF.

        :returns theta:
            The parameter value corresponding to the value of the CDF given by
            `x`.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.distribution.ppf(x, *self.args,
                                     loc=self.loc, scale=self.scale)

    def inverse_unit_transform(self, x, **kwargs):
        """Go from the parameter value to the unit coordinate using the cdf.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.distribution.cdf(x, *self.args,
                                     loc=self.loc, scale=self.scale)

    def gradient(self, theta):
        raise(NotImplementedError)

    @property
    def loc(self):
        """This should be overridden.
        """
        return 0

    @property
    def scale(self):
        """This should be overridden.
        """
        return 1

    @property
    def args(self):
        return []

    @property
    def range(self):
        raise(NotImplementedError)

    @property
    def bounds(self):
        raise(NotImplementedError)

    def serialize(self):
        raise(NotImplementedError)


class TopHat(Prior):
    """A simple uniform prior, described by two parameters

    :param mini:
        Minimum of the distribution

    :param maxi:
        Maximum of the distribution
    """
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
    """A simple gaussian prior.


    :param mean:
        Mean of the distribution

    :param sigma:
        Standard deviation of the distribution
    """
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


class ClippedNormal(Prior):
    """A Gaussian prior clipped to some range.

    :param mean:
        Mean of the normal distribution

    :param sigma:
        Standard deviation of the normal distribution

    :param mini:
        Minimum of the distribution

    :param maxi:
        Maximum of the distribution
    """
    prior_params = ['mean', 'sigma', 'mini', 'maxi']
    distribution = scipy.stats.truncnorm

    @property
    def scale(self):
        return self.params['sigma']

    @property
    def loc(self):
        return self.params['mean']

    @property
    def range(self):
        return (self.params['mini'], self.params['maxi'])

    @property
    def args(self):
        a = (self.params['mini'] - self.params['mean']) / self.params['sigma']
        b = (self.params['maxi'] - self.params['mean']) / self.params['sigma']
        return [a, b]

    def bounds(self, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range


class LogUniform(Prior):
    """Like log-normal, but the distribution of natural log of the variable is
    distributed uniformly instead of normally.

    :param mini:
        Minimum of the distribution

    :param maxi:
        Maximum of the distribution
    """
    prior_params = ['mini', 'maxi']
    distribution = scipy.stats.reciprocal

    @property
    def args(self):
        a = self.params['mini']
        b = self.params['maxi']
        return [a, b]

    @property
    def range(self):
        return (self.params['mini'], self.params['maxi'])

    def bounds(self, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range


class Beta(Prior):
    """A Beta distribution.

    :param mini:
        Minimum of the distribution

    :param maxi:
        Maximum of the distribution

    :param alpha:

    :param beta:
    """
    prior_params = ['mini', 'maxi', 'alpha', 'beta']
    distribution = scipy.stats.beta

    @property
    def scale(self):
        return self.params.get('maxi', 1) - self.params.get('mini', 0)

    @property
    def loc(self):
        return self.params.get('mini', 0)

    @property
    def args(self):
        a = self.params['alpha']
        b = self.params['beta']
        return [a, b]

    @property
    def range(self):
        return (self.params.get('mini',0), self.params.get('maxi',1))

    def bounds(self, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range


class LogNormal(Prior):
    """A log-normal prior, where the natural log of the variable is distributed
    normally.  Useful for parameters that cannot be less than zero.

    Note that ``LogNormal(np.exp(mode) / f) == LogNormal(np.exp(mode) * f)``
    and ``f = np.exp(sigma)`` corresponds to "one sigma" from the peak.

    :param mode:
        Natural log of the variable value at which the probability density is
        highest.

    :param sigma:
        Standard deviation of the distribution of the natural log of the
        variable.
    """
    prior_params = ['mode', 'sigma']
    distribution = scipy.stats.lognorm

    @property
    def args(self):
        return [self.params["sigma"]]

    @property
    def scale(self):
        return  np.exp(self.params["mode"] + self.params["sigma"]**2)

    @property
    def loc(self):
        return 0

    @property
    def range(self):
        nsig = 4
        return (np.exp(self.params['mode'] + (nsig * self.params['sigma'])),
                np.exp(self.params['mode'] - (nsig * self.params['sigma'])))

    def bounds(self, **kwargs):
        return (0, np.inf)


class LogNormalLinpar(Prior):
    """A log-normal prior, where the natural log of the variable is distributed
    normally.  Useful for parameters that cannot be less than zero.

    LogNormal(mode=x, sigma=y) is equivalent to
    LogNormalLinpar(mode=np.exp(x), sigma_factor=np.exp(y))

    :param mode:
        The (linear) value of the variable where the probability density is
        highest. Must be > 0.

    :param sigma_factor:
        The (linear) factor describing the dispersion of the log of the
        variable.  Must be > 0
    """
    prior_params = ['mode', 'sigma_factor']
    distribution = scipy.stats.lognorm

    @property
    def args(self):
        return [np.log(self.params["sigma_factor"])]

    @property
    def scale(self):
        k = self.params["sigma_factor"]**np.log(self.params["sigma_factor"])
        return  self.params["mode"] * k

    @property
    def loc(self):
        return 0

    @property
    def range(self):
        nsig = 4
        return (self.params['mode'] * (nsig * self.params['sigma_factor']),
                self.params['mode'] /  (nsig * self.params['sigma_factor']))

    def bounds(self, **kwargs):
        return (0, np.inf)

class SkewNormal(Prior):
    """A normal distribution including a skew parameter

    :param location:
        Center (*not* mean, mode, or median) of the distribution. 
        The center will approach the mean as skew approaches zero.

    :param sigma:
        Standard deviation of the distribution

    :param skew:
        Skewness of the distribution
    """
    prior_params = ['location', 'sigma', 'skew']
    distribution = scipy.stats.skewnorm

    @property
    def args(self):
        return [self.params['skew']]

    @property
    def scale(self):
        return self.params['sigma']

    @property
    def loc(self):
        return self.params['location']

    @property
    def range(self):
        nsig = 4
        return (self.params['location'] - nsig * self.params['sigma'],
                self.params['location'] + nsig * self.params['sigma'])

    def bounds(self, **kwargs):
        return (-np.inf, np.inf)


class StudentT(Prior):
    """A Student's T distribution

    :param mean:
        Mean of the distribution

    :param scale:
        Size of the distribution, analogous to the standard deviation

    :param df:
        Number of degrees of freedom
    """
    prior_params = ['mean', 'scale', 'df']
    distribution = scipy.stats.t

    @property
    def args(self):
        return [self.params['df']]

    @property
    def scale(self):
        return self.params['scale']

    @property
    def loc(self):
        return self.params['mean']

    @property
    def range(self):
        nsig = 4
        return (self.params['location'] - nsig * self.params['scale'],
                self.params['location'] + nsig * self.params['scale'])

    def bounds(self, **kwargs):
        return (-np.inf, np.inf)
