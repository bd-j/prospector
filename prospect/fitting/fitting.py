#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""fitting.py -- Default posterior probability function and high-level fitting
methods for prospector
"""


import time
from functools import partial as argfix

import numpy as np
from scipy.optimize import minimize, least_squares
import warnings

from .minimizer import minimize_wrapper, minimizer_ball
from .ensemble import run_emcee_sampler
from .nested import run_dynesty_sampler
from ..likelihood.likelihood import compute_chi, compute_lnlike


__all__ = ["lnprobfn", "fit_model",
           "run_minimize", "run_emcee", "run_dynesty"
           ]


<<<<<<< HEAD
def lnprobfn(theta, model=None, observations=None, sps=None, noises=None,
             residuals=False, nested=False, negative=False, verbose=False):
=======
def lnprobfn(theta, model=None, observations=None, sps=None,
             residuals=False, nested=False, verbose=False):
>>>>>>> 5617c8c (fitting ubdates for observation lists; dosctring modernization.)
    """Given a parameter vector and optionally a dictionary of observational
    ata and a model object, return the matural log of the posterior. This
    requires that an sps object (and if using spectra and gaussian processes, a
    NoiseModel) be instantiated.

    Parameters
    ----------
    theta :  ndarray of shape ``(ndim,)``
        Input parameter vector

    model : instance of the :py:class:`prospect.models.SedModel`
        The model parameterization and parameter state. Must have
        :py:meth:`predict()` defined

    observations : A list of :py:class:`observation.Observation` instances
        The data to be fit.

    sps : instance of a :py:class:`prospect.sources.SSPBasis` (sub-)class.
        The object used to construct the basic physical spectral model.
        Anything with a compatible :py:func:`get_galaxy_spectrum` can
        be used here. It will be passed to ``lnprobfn``

    residuals : bool (optional, default: False)
        A switch to allow vectors of :math:`\chi` values to be returned instead
        of a scalar posterior probability.  This can be useful for
        least-squares optimization methods. Note that prior probabilities are
        not included in this calculation.

    nested : bool (optional, default: False)
        If ``True``, do not add the ln-prior probability to the ln-likelihood
        when computing the ln-posterior.  For nested sampling algorithms the
        prior probability is incorporated in the way samples are drawn, so
        should not be included here.

    negative: bool (optional, default: False)
        If ``True`` return the negative on the ln-probability for minimization
        purposes.

    Returns
    -------
    lnp : float or ndarry of shape `(ndof,)`
        Ln-probability, unless ``residuals=True`` in which case a vector of
        :math:`\chi` values is returned.
    """
    if residuals:
        ndof = np.sum([obs["ndof"] for obs in observations])
        lnnull = np.zeros(ndof) - 1e18  # -np.infty
    else:
        lnnull = -np.infty

    # --- Calculate prior probability and exit if not within prior ---
    lnp_prior = model.prior_product(theta, nested=nested)
    if not np.isfinite(lnp_prior):
        return lnnull

    # set parameters
    model.set_parameters(theta)

    #  --- Update Noise Model Parameters ---
    [obs.noise.update(**model.params) for obs in observations
     if obs.noise is not None]

    # --- Generate mean model ---
    try:
        predictions, x = model.predict(theta, observations, sps=sps)
    except(ValueError):
        return lnnull
    except:
        print("There was an error during the likelihood call at parameters {}".format(theta))
        raise

    # --- Optionally return chi vectors for least-squares ---
    # note this does not include priors!
    if residuals:
        chi = [compute_chi(spec, obs) for pred, obs in zip(predictions, observations)]
        return np.concatenate(chi)

    # --- Emission Lines ---
    lnp_eline = getattr(model, "_ln_eline_penalty", 0.0)

    # --- Calculate likelihoods ---
    lnp_data = [compute_lnlike(pred, obs, vectors={}) for pred, obs
                in zip(predictions, observations)]

    lnp = lnp_prior + np.sum(lnp_data) + lnp_eline
    if negative:
        lnp *= -1

    return lnp


def wrap_lnp(lnpfn, observations, model, sps, **lnp_kwargs):
    return argfix(lnpfn, observations=observations, model=model, sps=sps,
                  **lnp_kwargs)


def fit_model(observations, model, sps, lnprobfn=lnprobfn,
              optimize=False, emcee=False, dynesty=True, **kwargs):
    """Fit a model to observations using a number of different methods

    Parameters
    ----------
    observations : list of :py:class:`observate.Observation` instances
        The data to be fit.

    model : instance of the :py:class:`prospect.models.SedModel`
        The model parameterization and parameter state.  It will be
        passed to ``lnprobfn``.

    sps : instance of a :py:class:`prospect.sources.SSPBasis` (sub-)class.
        The object used to construct the basic physical spectral model.
        Anything with a compatible :py:func:`get_galaxy_spectrum` can
        be used here. It will be passed to ``lnprobfn``

    lnprobfn : callable (optional, default: :py:meth:`lnprobfn`)
        A posterior probability function that can take ``observations``,
        ``model``, and ``sps`` as keywords. By default use the
        :py:func:`lnprobfn` defined above.

    optimize : bool (optional, default: False)
        If ``True``, conduct a round of optimization before sampling from the
        posterior.  The model state will be set to the best value at the end of
        optimization before continuing on to sampling or returning.  Parameters
        controlling the optimization can be passed via ``kwargs``, including

        + ``min_method``: 'lm' | 'powell'
        + ``nmin``: number of minimizations to do.  Beyond the first, minimizations
          will be started from draws from the prior.
        + ``min_opts``: dictionary of minimization options passed to the
          scipy.optimize.minimize method.

        See :py:func:`run_minimize` for details.

    emcee : bool  (optional, default: False)
        If ``True``, sample from the posterior using emcee.  Additonal
        parameters controlling emcee can be passed via ``**kwargs``.  These include

        + ``initial_positions``: A set of initial positions for the walkers
        + ``hfile``: an open h5py.File file handle for writing result incrementally

        Many additional emcee parameters can be provided here, see
        :py:func:`run_emcee` for details.

    dynesty : bool (optional, default: True)
        If ``True``, sample from the posterior using dynesty.  Additonal
        parameters controlling dynesty can be passed via ``**kwargs``. See
        :py:func:`run_dynesty` for details.

    Returns
    -------
    output : dictionary
        A dictionary with two keys, ``"optimization"`` and ``"sampling"``.  The
        value of each of these is a 2-tuple with results in the first element
        and durations (in seconds) in the second element.
    """
    # Make sure obs has required keys
    [obs.rectify() for obs in observations]

    if emcee & dynesty:
        msg = ("Cannot run both emcee and dynesty fits "
               "in a single call to fit_model")
        raise(ValueError, msg)
    if (not emcee) & (not dynesty) & (not optimize):
        msg = ("No sampling or optimization routine "
               "specified by user; returning empty results")
        warnings.warn(msg)

    output = {"optimization": (None, 0.),
              "sampling": (None, 0.)}

    if optimize:
        optres, topt, best = run_minimize(observations, model, sps,
                                          lnprobfn=lnprobfn, **kwargs)
        # set to the best
        model.set_parameters(optres[best].x)
        output["optimization"] = (optres, topt)

    if emcee:
        run_sampler = run_emcee
    elif dynesty:
        run_sampler = run_dynesty
    else:
        return output

    output["sampling"] = run_sampler(observations, model, sps,
                                     lnprobfn=lnprobfn, **kwargs)
    return output


def run_minimize(observations=None, model=None, sps=None, lnprobfn=lnprobfn,
                 min_method='lm', min_opts={}, nmin=1, pool=None, **extras):
    """Run a minimization.  This wraps the lnprobfn fixing the ``obs``,
    ``model``, ``noise``, and ``sps`` objects, and then runs a minimization of
    -lnP using scipy.optimize methods.

    Parameters
    ----------
    observations : list of :py:class:`observate.Observation` instances
        The data to be fit.

    model : instance of the :py:class:`prospect.models.SedModel`
        The model parameterization and parameter state.  It will be
        passed to ``lnprobfn``.

    sps : instance of a :py:class:`prospect.sources.SSPBasis` (sub-)class.
        The object used to construct the basic physical spectral model.
        Anything with a compatible :py:func:`get_galaxy_spectrum` can
        be used here. It will be passed to ``lnprobfn``

    lnprobfn : callable (optional, default: :py:meth:`lnprobfn`)
        A posterior probability function that can take ``observations``,
        ``model``, and ``sps`` as keywords. By default use the
        :py:func:`lnprobfn` defined above.

    min_method : string (optional, default: 'lm')
        Method to use for minimization
        * 'lm': Levenberg-Marquardt
        * 'powell': Powell line search method

    nmin : int (optional, default: 1)
        Number of minimizations to do.  Beyond the first, minimizations will be
        started from draws from the prior.

    min_opts : dict (optional, default: {})
        Dictionary of minimization options passed to the scipy.optimize method.
        These include things like 'xtol', 'ftol', etc..

    pool : object (optional, default: None)
        A pool to use for parallel optimization from multiple initial positions.

    Returns
    -------
    results :
        A list of `scipy.optimize.OptimizeResult` objects.

    t_wall : float
        Wall time used for the minimization, in seconds.

    best : int
        The index of the results list containing the lowest chi-square result.
    """
    initial = model.theta.copy()

    lsq = ["lm"]
    scalar = ["powell"]

    # --- Set some options based on minimization method ---
    if min_method in lsq:
        algorithm = least_squares
        residuals = True
        min_opts["x_scale"] = "jac"
    elif min_method in scalar:
        algorithm = minimize
        residuals = False

    args = []
<<<<<<< HEAD
    loss = argfix(lnprobfn, obs=obs, model=model, sps=sps,
                  noise=noise, residuals=residuals, negative=True)
=======
    loss = argfix(lnprobfn, observations=observations, model=model, sps=sps, residuals=residuals)
>>>>>>> 5617c8c (fitting ubdates for observation lists; dosctring modernization.)
    minimizer = minimize_wrapper(algorithm, loss, [], min_method, min_opts)
    qinit = minimizer_ball(initial, nmin, model)

    if pool is not None:
        M = pool.map
    else:
        M = map

    t = time.time()
    results = list(M(minimizer, [np.array(q) for q in qinit]))
    tm = time.time() - t

    if min_method in lsq:
        chisq = [np.sum(r.fun**2) for r in results]
        best = np.argmin(chisq)
    elif min_method in scalar:
        best = np.argmin([p.fun for p in results])

    return results, tm, best


def run_emcee(observations, model, sps, lnprobfn=lnprobfn,
              hfile=None, initial_positions=None, **kwargs):
    """Run emcee, optionally including burn-in and convergence checking.  Thin
    wrapper on :py:class:`prospect.fitting.ensemble.run_emcee_sampler`

    Parameters
    ----------
    observations : list of :py:class:`observate.Observation` instances
        The data to be fit.

    model : instance of the :py:class:`prospect.models.SedModel`
        The model parameterization and parameter state.  It will be
        passed to ``lnprobfn``.

    sps : instance of a :py:class:`prospect.sources.SSPBasis` (sub-)class.
        The object used to construct the basic physical spectral model.
        Anything with a compatible :py:func:`get_galaxy_spectrum` can
        be used here. It will be passed to ``lnprobfn``

    lnprobfn : callable (optional, default: :py:meth:`lnprobfn`)
        A posterior probability function that can take ``observations``,
        ``model``, and ``sps`` as keywords. By default use the
        :py:func:`lnprobfn` defined above.

    hfile : :py:class:`h5py.File()` instance (optional, default: None)
        A file handle for a :py:class:`h5py.File` object that will be written
        to incremantally during sampling.

    initial_positions : ndarray of shape ``(nwalkers, ndim)`` (optional, default: None)
        If given, a set of initial positions for the emcee walkers.  Rounds of
        burn-in will be skipped if this parameter is present.

    Extra Parameters
    --------
    nwalkers : int
        The number of walkers to use.  If None, use the nearest power of two to
        ``ndim * walker_factor``.

    niter : int
        Number of iterations for the production run

    nburn : list of int
        List of the number of iterations to run in each round of burn-in (for
        removing stuck walkers.) E.g. `nburn=[32, 64]` will run the sampler for
        32 iterations before reinitializing and then run the sampler for
        another 64 iterations before starting the production run.

    storechain : bool (default: True)
        If using HDF5 output, setting this to False will keep the chain from
        being held in memory by the sampler object.

    :param pool: (optional)
        A ``Pool`` object, either from ``multiprocessing`` or from
        ``emcee.mpi_pool``.

    :param interval:
        Fraction of the full run at which to flush to disk, if using hdf5 for
        output.

    :param convergence_check_interval:
        How often to assess convergence, in number of iterations. If this is
        not `None`, then the KL convergence test is run.

    :param convergence_chunks:
        The number of iterations to combine when creating the marginalized
        parameter probability functions.

    :param convergence_stable_points_criteria:
        The number of stable convergence checks that the chain must pass before
        being declared stable.

    Returns
    --------
    sampler :
        An instance of :py:class:`emcee.EnsembleSampler`.

    t_wall : float
        Duration of sampling (including burn-in) in seconds of wall time.
    """
    q = model.theta.copy()

    postkwargs = {"observations": observations,
                  "model": model,
                  "sps": sps,
                  "nested": False,
                  }

    # Could try to make signatures for these two methods the same....
    if initial_positions is not None:
        meth = restart_emcee_sampler
        t = time.time()
        out = meth(lnprobfn, initial_positions, hdf5=hfile,
                   postkwargs=postkwargs, **kwargs)
        sampler = out
        ts = time.time() - t
    else:
        meth = run_emcee_sampler
        t = time.time()
        out = meth(lnprobfn, q, model, hdf5=hfile,
                   postkwargs=postkwargs, **kwargs)
        sampler, burn_p0, burn_prob0 = out
        ts = time.time() - t

    return sampler, ts


<<<<<<< HEAD
def run_dynesty(obs, model, sps, noise, lnprobfn=lnprobfn,
                pool=None, nested_target_n_effective=10000, **kwargs):
=======
def run_dynesty(obs, model, sps, lnprobfn=lnprobfn,
                pool=None, nested_posterior_thresh=0.05, **kwargs):
>>>>>>> 5617c8c (fitting ubdates for observation lists; dosctring modernization.)
    """Thin wrapper on :py:class:`prospect.fitting.nested.run_dynesty_sampler`

    Parameters
    ----------
    observations : list of :py:class:`observate.Observation` instances
        The data to be fit.

    model : instance of the :py:class:`prospect.models.SedModel`
        The model parameterization and parameter state.  It will be
        passed to ``lnprobfn``.

    sps : instance of a :py:class:`prospect.sources.SSPBasis` (sub-)class.
        The object used to construct the basic physical spectral model.
        Anything with a compatible :py:func:`get_galaxy_spectrum` can
        be used here. It will be passed to ``lnprobfn``

    lnprobfn : callable (optional, default: :py:meth:`lnprobfn`)
        A posterior probability function that can take ``observations``,
        ``model``, and ``sps`` as keywords. By default use the
        :py:func:`lnprobfn` defined above.

    Extra Parameters
    --------
    nested_bound: (optional, default: 'multi')

    nested_sample: (optional, default: 'unif')

    nested_nlive_init: (optional, default: 100)

    nested_nlive_batch: (optional, default: 100)

    nested_dlogz_init: (optional, default: 0.02)

    nested_maxcall: (optional, default: None)

    nested_walks: (optional, default: 25)

    Returns
    --------
    result:
        An instance of :py:class:`dynesty.results.Results`.

    t_wall : float
        Duration of sampling in seconds of wall time.
    """
    from dynesty.dynamicsampler import stopping_function, weight_function
    nested_stop_kwargs = {"target_n_effective": nested_target_n_effective}

    lnp = wrap_lnp(lnprobfn, observations, model, sps, noise=noise,
                   nested=True)

    # Need to deal with postkwargs...

    t = time.time()
    dynestyout = run_dynesty_sampler(lnp, model.prior_transform, model.ndim,
                                     stop_function=stopping_function,
                                     wt_function=weight_function,
                                     nested_stop_kwargs=nested_stop_kwargs,
                                     pool=pool, **kwargs)
    ts = time.time() - t

    return dynestyout, ts
