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
from ..likelihood import lnlike_spec, lnlike_phot, chi_spec, chi_phot, write_log
from ..utils.obsutils import fix_obs


__all__ = ["lnprobfn", "fit_model",
           "run_minimize", "run_emcee", "run_dynesty"
           ]


def lnprobfn(theta, model=None, obs=None, sps=None, noise=(None, None),
             residuals=False, nested=False, negative=False, verbose=False):
    """Given a parameter vector and optionally a dictionary of observational
    ata and a model object, return the matural log of the posterior. This
    requires that an sps object (and if using spectra and gaussian processes, a
    NoiseModel) be instantiated.

    :param theta:
        Input parameter vector, ndarray of shape (ndim,)

    :param model:
        SedModel model object, with attributes including ``params``, a
        dictionary of model parameter state.  It must also have
        :py:func:`prior_product`, and :py:func:`predict` methods
        defined.

    :param obs:
        A dictionary of observational data.  The keys should be

        + ``"wavelength"``  (angstroms)
        + ``"spectrum"``    (maggies)
        + ``"unc"``         (maggies)
        + ``"maggies"``     (photometry in maggies)
        + ``"maggies_unc"`` (photometry uncertainty in maggies)
        + ``"filters"``     (:py:class:`sedpy.observate.FilterSet` or iterable of :py:class:`sedpy.observate.Filter`)
        +  and optional spectroscopic ``"mask"`` and ``"phot_mask"`` (same
           length as ``spectrum`` and ``maggies`` respectively, True means use
           the data points)

    :param sps:
        A :py:class:`prospect.sources.SSPBasis` object or subclass thereof, or
        any object with a ``get_spectrum`` method that will take a dictionary
        of model parameters and return a spectrum, photometry, and ancillary
        information.

    :param noise: (optional, default: (None, None))
        A 2-element tuple of :py:class:`prospect.likelihood.NoiseModel` objects.

    :param residuals: (optional, default: False)
        A switch to allow vectors of :math:`\chi` values to be returned instead
        of a scalar posterior probability.  This can be useful for
        least-squares optimization methods. Note that prior probabilities are
        not included in this calculation.

    :param nested: (optional, default: False)
        If ``True``, do not add the ln-prior probability to the ln-likelihood
        when computing the ln-posterior.  For nested sampling algorithms the
        prior probability is incorporated in the way samples are drawn, so
        should not be included here.

    :param negative: (optiona, default: False)
        If ``True`` return the negative on the ln-probability for minimization
        purposes.

    :returns lnp:
        Ln-probability, unless ``residuals=True`` in which case a vector of
        :math:`\chi` values is returned.
    """
    if residuals:
        lnnull = np.zeros(obs["ndof"]) - 1e18  # np.infty
        #lnnull = -np.infty
    else:
        lnnull = -np.infty

    # --- Calculate prior probability and exit if not within prior ---
    lnp_prior = model.prior_product(theta, nested=nested)
    if not np.isfinite(lnp_prior):
        return lnnull

    #  --- Update Noise Model ---
    spec_noise, phot_noise = noise
    vectors, sigma_spec = {}, None
    model.set_parameters(theta)
    if spec_noise is not None:
        spec_noise.update(**model.params)
        vectors.update({"unc": obs.get('unc', None)})
        sigma_spec = spec_noise.construct_covariance(**vectors)
    if phot_noise is not None:
        phot_noise.update(**model.params)
        vectors.update({'phot_unc': obs.get('maggies_unc', None),
                        'phot': obs.get('maggies', None),
                        'filter_names': obs.get('filter_names', None)})

    # --- Generate mean model ---
    try:
        t1 = time.time()
        spec, phot, x = model.predict(theta, obs, sps=sps, sigma_spec=sigma_spec)
        d1 = time.time() - t1
    except(ValueError):
        return lnnull
    except:
        print("There was an error during the likelihood call at parameters {}".format(theta))
        raise

    # --- Optionally return chi vectors for least-squares ---
    # note this does not include priors!
    if residuals:
        chispec = chi_spec(spec, obs)
        chiphot = chi_phot(phot, obs)
        return np.concatenate([chispec, chiphot])

    #  --- Mixture Model ---
    f_outlier_spec = model.params.get('f_outlier_spec', 0.0)
    if (f_outlier_spec != 0.0):
        sigma_outlier_spec = model.params.get('nsigma_outlier_spec', 10)
        vectors.update({'nsigma_outlier_spec': sigma_outlier_spec})
    f_outlier_phot = model.params.get('f_outlier_phot', 0.0)
    if (f_outlier_phot != 0.0):
        sigma_outlier_phot = model.params.get('nsigma_outlier_phot', 10)
        vectors.update({'nsigma_outlier_phot': sigma_outlier_phot})

    # --- Emission Lines ---

    # --- Calculate likelihoods ---
    t1 = time.time()
    lnp_spec = lnlike_spec(spec, obs=obs,
                           f_outlier_spec=f_outlier_spec,
                           spec_noise=spec_noise,
                           **vectors)
    lnp_phot = lnlike_phot(phot, obs=obs,
                           f_outlier_phot=f_outlier_phot,
                           phot_noise=phot_noise, **vectors)
    lnp_eline = getattr(model, '_ln_eline_penalty', 0.0)

    d2 = time.time() - t1
    if verbose:
        write_log(theta, lnp_prior, lnp_spec, lnp_phot, d1, d2)

    lnp = lnp_prior + lnp_phot + lnp_spec + lnp_eline
    if negative:
        lnp *= -1

    return lnp


def wrap_lnp(lnpfn, obs, model, sps, **lnp_kwargs):
    return argfix(lnpfn, obs=obs, model=model, sps=sps,
                  **lnp_kwargs)


def fit_model(obs, model, sps, noise=(None, None), lnprobfn=lnprobfn,
              optimize=False, emcee=False, dynesty=True, **kwargs):
    """Fit a model to observations using a number of different methods

    :param obs:
        The ``obs`` dictionary containing the data to fit to, which will be
        passed to ``lnprobfn``.

    :param model:
        An instance of the :py:class:`prospect.models.SedModel` class
        containing the model parameterization and parameter state.  It will be
        passed to ``lnprobfn``.

    :param sps:
        An instance of a :py:class:`prospect.sources.SSPBasis` (sub-)class.
        Alternatively, anything with a compatible :py:func:`get_spectrum` can
        be used here. It will be passed to ``lnprobfn``

    :param noise: (optional, default: (None, None))
        A tuple of NoiseModel objects for the spectroscopy and photometry
        respectively.  Can also be (None, None) in which case simple chi-square
        will be used.

    :param lnprobfn: (optional, default: lnprobfn)
        A posterior probability function that can take ``obs``, ``model``,
        ``sps``, and ``noise`` as keywords. By default use the
        :py:func:`lnprobfn` defined above.

    :param optimize: (optional, default: False)
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

    :param emcee:  (optional, default: False)
        If ``True``, sample from the posterior using emcee.  Additonal
        parameters controlling emcee can be passed via ``**kwargs``.  These include

        + ``initial_positions``: A set of initial positions for the walkers
        + ``hfile``: an open h5py.File file handle for writing result incrementally

        Many additional emcee parameters can be provided here, see
        :py:func:`run_emcee` for details.

    :param dynesty:
        If ``True``, sample from the posterior using dynesty.  Additonal
        parameters controlling dynesty can be passed via ``**kwargs``. See
        :py:func:`run_dynesty` for details.

    :returns output:
        A dictionary with two keys, ``"optimization"`` and ``"sampling"``.  The
        value of each of these is a 2-tuple with results in the first element
        and durations (in seconds) in the second element.
    """
    # Make sure obs has required keys
    obs = fix_obs(obs)

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
        optres, topt, best = run_minimize(obs, model, sps, noise,
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

    output["sampling"] = run_sampler(obs, model, sps, noise,
                                     lnprobfn=lnprobfn, **kwargs)
    return output


def run_minimize(obs=None, model=None, sps=None, noise=None, lnprobfn=lnprobfn,
                 min_method='lm', min_opts={}, nmin=1, pool=None, **extras):
    """Run a minimization.  This wraps the lnprobfn fixing the ``obs``,
    ``model``, ``noise``, and ``sps`` objects, and then runs a minimization of
    -lnP using scipy.optimize methods.

    :param obs:
        The ``obs`` dictionary containing the data to fit to, which will be
        passed to ``lnprobfn``.

    :param model:
        An instance of the :py:class:`prospect.models.SedModel` class
        containing the model parameterization and parameter state.  It will be
        passed to ``lnprobfn``.

    :param sps:
        An instance of a :py:class:`prospect.sources.SSPBasis` (sub-)class.
        Alternatively, anything with a compatible :py:func:`get_spectrum` can
        be used here. It will be passed to ``lnprobfn``

    :param noise: (optional)
        If given, a tuple of :py:class:`NoiseModel` objects passed to
        ``lnprobfn``.

    :param lnprobfn: (optional, default: lnprobfn)
        A posterior probability function that can take ``obs``, ``model``,
        ``sps``, and ``noise`` as keywords. By default use the
        :py:func:`lnprobfn` defined above.

    :param min_method: (optional, default: 'lm')
        Method to use for minimization
        * 'lm': Levenberg-Marquardt
        * 'powell': Powell line search method

    :param nmin: (optional, default: 1)
        Number of minimizations to do.  Beyond the first, minimizations will be
        started from draws from the prior.

    :param min_opts: (optional, default: {})
        Dictionary of minimization options passed to the scipy.optimize method.
        These include things like 'xtol', 'ftol', etc..

    :param pool: (optional, default: None)
        A pool to use for parallel optimization from multiple initial positions.

    :returns results:
        A list of `scipy.optimize.OptimizeResult` objects.

    :returns tm:
        Wall time used for the minimization, in seconds.

    :returns best:
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
    loss = argfix(lnprobfn, obs=obs, model=model, sps=sps,
                  noise=noise, residuals=residuals, negative=True)
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


def run_emcee(obs, model, sps, noise, lnprobfn=lnprobfn,
              hfile=None, initial_positions=None,
              **kwargs):
    """Run emcee, optionally including burn-in and convergence checking.  Thin
    wrapper on :py:class:`prospect.fitting.ensemble.run_emcee_sampler`

    :param obs:
        The ``obs`` dictionary containing the data to fit to, which will be
        passed to ``lnprobfn``.

    :param model:
        An instance of the :py:class:`prospect.models.SedModel` class
        containing the model parameterization and parameter state.  It will be
        passed to ``lnprobfn``.

    :param sps:
        An instance of a :py:class:`prospect.sources.SSPBasis` (sub-)class.
        Alternatively, anything with a compatible :py:func:`get_spectrum` can
        be used here. It will be passed to ``lnprobfn``

    :param noise:
        A tuple of :py:class:`NoiseModel` objects passed to ``lnprobfn``.

    :param lnprobfn: (optional, default: lnprobfn)
        A posterior probability function that can take ``obs``, ``model``,
        ``sps``, and ``noise`` as keywords. By default use the
        :py:func:`lnprobfn` defined above.

    :param hfile: (optional, default: None)
        A file handle for a :py:class:`h5py.File` object that will be written
        to incremantally during sampling.

    :param initial_positions: (optional, default: None)
        If given, a set of initial positions for the emcee walkers.  Must have
        shape (nwalkers, ndim).  Rounds of burn-in will be skipped if this
        parameter is present.

    Extra Parameters
    --------

    :param nwalkers:
        The number of walkers to use.  If None, use the nearest power of two to
        ``ndim * walker_factor``.

    :param niter:
        Number of iterations for the production run

    :param nburn:
        List of the number of iterations to run in each round of burn-in (for
        removing stuck walkers.) E.g. `nburn=[32, 64]` will run the sampler for
        32 iterations before reinitializing and then run the sampler for
        another 64 iterations before starting the production run.

    :param storechain: (default: True)
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

    :returns sampler:
        An instance of :py:class:`emcee.EnsembleSampler`.

    :returns ts:
        Duration of sampling (including burn-in) in seconds of wall time.
    """
    q = model.theta.copy()

    postkwargs = {}
    for item in ['obs', 'model', 'sps', 'noise']:
        val = eval(item)
        if val is not None:
            postkwargs[item] = val
    
    postkwargs['nested'] = False

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


def run_dynesty(obs, model, sps, noise, lnprobfn=lnprobfn,
                pool=None, nested_target_n_effective=10000, **kwargs):
    """Thin wrapper on :py:class:`prospect.fitting.nested.run_dynesty_sampler`

    :param obs:
        The ``obs`` dictionary containing the data to fit to, which will be
        passed to ``lnprobfn``.

    :param model:
        An instance of the :py:class:`prospect.models.SedModel` class
        containing the model parameterization and parameter state.  It will be
        passed to ``lnprobfn``.

    :param sps:
        An instance of a :py:class:`prospect.sources.SSPBasis` (sub-)class.
        Alternatively, anything with a compatible :py:func:`get_spectrum` can
        be used here. It will be passed to ``lnprobfn``

    :param noise:
        A tuple of :py:class:`prospect.likelihood.NoiseModel` objects passed to
        ``lnprobfn``.

    :param lnprobfn: (optional, default: :py:func:`lnprobfn`)
        A posterior probability function that can take ``obs``, ``model``,
        ``sps``, and ``noise`` as keywords. This function must also take a
        ``nested`` keyword.

    Extra Parameters
    --------
    :param nested_bound: (optional, default: 'multi')

    :param nested_sample: (optional, default: 'unif')

    :param nested_nlive_init: (optional, default: 100)

    :param nested_nlive_batch: (optional, default: 100)

    :param nested_dlogz_init: (optional, default: 0.02)

    :param nested_maxcall: (optional, default: None)

    :param nested_walks: (optional, default: 25)

    Returns
    --------

    :returns result:
        An instance of :py:class:`dynesty.results.Results`.

    :returns ts:
        Duration of sampling in seconds of wall time.
    """
    from dynesty.dynamicsampler import stopping_function, weight_function
    nested_stop_kwargs = {"target_n_effective": nested_target_n_effective}

    lnp = wrap_lnp(lnprobfn, obs, model, sps, noise=noise,
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
