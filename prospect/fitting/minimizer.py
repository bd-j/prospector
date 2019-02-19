import warnings
import numpy as np
from numpy.random import normal, multivariate_normal


__all__ = ["minimize_wrapper", "minimizer_ball", "reinitialize"]


class minimize_wrapper(object):
    """This is a hack to make the minimization function pickleable (for MPI)
    even though it requires many arguments.  Ripped off from emcee.
    """
    def __init__(self, algorithm, function, args, method, options):
        self.f = algorithm
        self.func = function
        self.args = tuple(args)
        self.meth = method
        self.opts = options

    def __call__(self, x):
        try:
            return self.f(self.func, x, args=self.args,
                          method=self.meth, **self.opts)
        except:
            import traceback
            print("minimizer: Exception while trying to minimize the function:")
            print("  params:", x)
            print("  args:", self.args)
            print("  exception:")
            traceback.print_exc()
            raise


def minimizer_ball(center, nminimizers, model, seed=None):
    """Draw initial values from the (1d, separable, independent) priors for
    each parameter.  Requires that priors have the `sample` method available.
    If priors are old-style, draw randomly between min and max.
    """
    rand = np.random.RandomState(seed)

    size = nminimizers
    pinitial = [center]
    if size > 1:
        ginitial = np.zeros([size - 1, model.ndim])
        for p, inds in list(model.theta_index.items()):
            for j in range(size-1):
                ginitial[j, inds] = model._config_dict[p]['prior'].sample()
        pinitial += ginitial.tolist()
    return pinitial


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
    warnings.warn("minimizer.reintialize is deprecated", DeprecationWarning)
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
