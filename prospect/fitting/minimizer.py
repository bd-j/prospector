import numpy as np
from numpy.random import normal, multivariate_normal
from scipy.optimize import minimize

try:
    import multiprocessing
except ImportError:
    pass


__all__ = ["Pminimize", "pminimize", "minimizer_ball", "reinitialize"]


class Pminimize(object):
    """A class for performing minimizations using scipy's minimize function.
    This class enables minimizations from different initial positions to be
    parallelized.

    If a pool object is passed, that object's map function is used to
    parallelize the minimizations.  Otherwise, if nthreads > 1, we
    generate a pool using multiprocessing and use that map function.

    :param chi2fn:
        Function to be minimized.

    :param opts:
        Dictionary of options for the minimization, as described in
        scipy's minimize docs.

    :param args:
        A sequence of objects that are passed as
        an arguments to your chi2 function

    :param method:  (Default: 'powell')
        Minimization method.  This should probably be 'powell', since
        I haven't tried anything else and there's no way to pass
        hessians or jacobians.

    :param pool: (Default: None)
        A pool object which contains a map method that will be used
        for distributing tasks.

    :param nthreads: (Default: 1)
        If pool is None and nthreads > 1, create a mutiprocessing pool
        with nthreads threads.  Otherwise, this is ignored.
    """

    def __init__(self, chi2fn, args, opts, method='powell', pool=None, nthreads=1):
        self.method = method
        self.opts = opts
        self.args = args
        self._size = None

        # Wrap scipy's minimize to make it pickleable
        self.minimize = _minimize_wrapper(chi2fn, args, method, opts)

        self.pool = pool
        if nthreads > 1 and self.pool is None:
            self.pool = multiprocessing.Pool(nthreads)
            self._size = nthreads

    def run(self, pinit):
        """Actually run the minimizations, in parallel if pool has been set up.

        :param pinit:
           An iterable where each element is a parameter vector for a
           starting condition.

        :returns results:
            A list of scipy-style minimization results objects,
            corresponding to the initial locations.
        """
        if self.pool is not None:
            M = self.pool.map
        else:
            M = map

        results = list(M(self.minimize,  [np.array(p) for p in pinit]))
        return results

    @property
    def size(self):
        if self.pool is None:
            return 1
        elif self._size is not None:
        #    print('uhoh')
            return self._size
        else:
            return self.pool.size


class _minimize_wrapper(object):
    """This is a hack to make the minimization function pickleable (for MPI)
    even though it requires many arguments.  Ripped off from emcee.
    """
    def __init__(self, function, args, method, options):
        self.f = minimize
        self.func = function
        self.args = tuple(args)
        self.meth = method
        self.opts = options

    def __call__(self, x):
        try:
            return self.f(self.func, x, args=self.args,
                          method=self.meth, options=self.opts)
        except:
            import traceback
            print("minimizer: Exception while trying to minimize the function:")
            print("  params:", x)
            print("  args:", self.args)
            print("  exception:")
            traceback.print_exc()
            raise


def pminimize(chi2fn, initial, args=None, model=None,
              method='powell', opts=None,
              pool=None, nthreads=1):
    """Do as many minimizations as you have threads, in parallel.  Always use
    initial_center for one of the minimization streams, the rest will be
    sampled from the prior for each parameter.  Returns each of the
    minimization result dictionaries, as well as the starting positions.
    """
    # Instantiate the minimizer
    mini = Pminimize(chi2fn, args, opts,
                     method=method, pool=pool, nthreads=1)
    size = mini.size
    pinitial = minimizer_ball_fromprior(initial, size, model)
    powell_guesses = mini.run(pinitial)

    return [powell_guesses, pinitial]


def reinitialize(best_guess, model, edge_trunc=0.1, reinit_params=[],
                 **extras):
    """Check if the Powell minimization found a minimum close to the edge of
    the prior for any parameter. If so, reinitialize to the center of the
    prior.

    This is only done for parameters where ``'reinit':True`` in the model
    configuration dictionary, or for parameters in the supplied
    ``reinit_params`` list.

    :param buest_guess:
        The result of some sort of optimization step, iterable of length
        model.ndim.

    :param model:
        A ..models.parameters.ProspectorParams() object.

    :param edge_trunc: (optional, default 0.1)
        The fractional distance from the edge of the priors that triggers
        reinitialization.

    :param reinit_params: optional
        A list of model parameter names to reinitialize, overrides the value or
        presence of the ``reinit`` key in the model configuration dictionary.

    :returns output:
        The best_guess with parameters near the edge reset to be at the center
        of the prior.  ndarray of shape (ndim,)
    """
    edge = edge_trunc
    bounds = model.theta_bounds()
    output = np.array(best_guess)
    reinit = np.zeros(model.ndim, dtype=bool)
    for p, inds in list(model.theta_index.items()):
        reinit[inds] = (model._config_dict[p].get('reinit', False) or
                        (p in reinit_params))

    for k, (guess, bound) in enumerate(zip(best_guess, bounds)):
        # Normalize the guess and the bounds
        prange = bound[1] - bound[0]
        g, b = guess/prange, bound/prange
        if ((g - b[0] < edge) or (b[1] - g < edge)) and (reinit[k]):
            output[k] = b[0] + prange/2
    return output


def minimizer_ball(center, nminimizers, model):
    """Setup a 'grid' of parameter values uniformly distributed between min and
    max More generally, this should sample from the prior for each parameter.
    """
    size = nminimizers
    pinitial = [center]
    if size > 1:
        ginitial = np.zeros([size - 1, model.ndim])
        for i, (lo, hi) in enumerate(model.theta_bounds()):  # this is a dumb loop to have
            ginitial[:, i] = np.random.uniform(lo, hi, size - 1)
        pinitial += ginitial.tolist()
    return pinitial


def minimizer_ball_fromprior(center, nminimizers, model):
    """Draw initial values from the (1d, separable, independent) priors for
    each parameter.  Requires that priors have the `sample` method available.
    """
    pinitial = [center]
    if nminimizers > 1:
        ginitial = np.zeros([nminimizers - 1, model.ndim])
        for p, inds in list(model.theta_index.items()):
            for j in range(nminimizers-1):
                ginitial[j, inds] = model._config_dict[p]['prior'].sample()
        pinitial += ginitial.tolist()
    return pinitial
