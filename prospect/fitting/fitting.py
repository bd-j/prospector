import time
from functools import partial as argfix

import numpy as np
from scipy.optimize import minimize, least_squares


from .minimizer import minimize_wrapper, minimizer_ball
from .ensemble import run_emcee_sampler
from .nested import run_dynesty_sampler
from ..likelihood import lnlike_spec, lnlike_phot, chi_spec, chi_phot


__all__ = ["lnprobfn", "fit_model",
           "run_minimize", "run_emcee", "run_dynesty"
           ]


def lnprobfn(theta, model=None, obs=None, sps=None, noise=(None, None),
             residuals=False, nested=False, verbose=False):
    """Given a parameter vector and optionally a dictionary of observational
    ata and a model object, return the ln of the posterior. This requires that
    an sps object (and if using spectra and gaussian processes, a GP object) be
    instantiated.

    :param theta:
        Input parameter vector, ndarray of shape (ndim,)

    :param model:
        SedModel model object, with attributes including ``params``, a
        dictionary of model parameters.  It must also have ``prior_product()``,
        and ``mean_model()`` methods defined.

    :param obs:
        A dictionary of observational data.  The keys should be
          *``wavelength``
          *``spectrum``
          *``unc``
          *``maggies``
          *``maggies_unc``
          *``filters``
          * and optional spectroscopic ``mask`` and ``phot_mask``.

    :returns lnp:
        Ln posterior probability.
    """
    # --- Calculate prior probability and exit if not within prior ---
    lnp_prior = model.prior_product(theta, nested=nested)
    if not np.isfinite(lnp_prior):
        return -np.infty

    # --- Generate mean model ---
    try:
        t1 = time.time()
        spec, phot, x = model.mean_model(theta, obs, sps=sps)
        d1 = time.time() - t1
    except(ValueError):
        return -np.infty

    # --- Optionally return chi vectors for least-squares ---
    # note this does not include priors!
    if residuals:
        chispec = chi_spec(spec, obs)
        chiphot = chi_phot(phot, obs)
        return np.concatenate([chispec, chiphot])

    #  --- Update Noise Model ---
    spec_noise, phot_noise = noise
    vectors = {}  # These should probably be copies....
    if spec_noise is not None:
        spec_noise.update(**model.params)
        vectors.update({'spec': spec, "unc": obs['unc']})
        vectors.update({'sed': model._spec, 'cal': model._speccal})
    if phot_noise is not None:
        phot_noise.update(**model.params)
        vectors.update({'phot': phot, 'phot_unc': obs['maggies_unc']})

    # --- Calculate likelihoods ---
    t1 = time.time()
    lnp_spec = lnlike_spec(spec, obs=obs,
                           spec_noise=spec_noise, **vectors)
    lnp_phot = lnlike_phot(phot, obs=obs,
                           phot_noise=phot_noise, **vectors)
    d2 = time.time() - t1
    if verbose:
        write_log(theta, lnp_prior, lnp_spec, lnp_phot, d1, d2)

    return lnp_prior + lnp_phot + lnp_spec


def wrap_lnp(lnpfn, obs, model, sps, **lnp_kwargs):
    return argfix(lnpfn, obs=obs, model=model, sps=sps,
                  **lnp_kwargs)


def fit_model(obs, model, sps, noise, lnprobfn=lnprobfn,
              optimize=False, emcee=False, dynesty=True, **kwargs):
    """
    """

    if emcee & dynesty:
        msg = ("Cannot run both emcee and dynesty fits "
               "in a single call to fit_model")
        raise(ValueError, msg)

    output = {"optimization": (None, 0.),
              "sampling": (None, 0.)}

    if optimize:
        optres, topt, best = run_minimize(obs, model, sps, noise,
                                          lnprobfn=lnprobfn, **kwargs)
        # set to the best
        model.set_parameters(mr[best].x)
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
    """Run a minimization
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
                  noise=noise, residuals=residuals)
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
    """Run emcee.  Thin wrapper on run_emcee_sampler
    """
    q = model.theta.copy()
    
    postkwargs = {"obs": obs,
                  "model": model,
                  "sps": sps,
                  "noise": noise,
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


def run_dynesty(obs, model, sps, noise, lnprobfn=lnprobfn,
                pool=None, **kwargs):
    """Thin wrapper on run_dynesty_sampler
    """

    def prior_transform(u, model=model):
        return model.prior_transform(u)

    lnp = wrap_lnp(lnprobfn, obs, model, sps, noise=noise,
                   nested=True)

    # Need to deal with postkwargs

    t = time.time()
    dynestyout = run_dynesty_sampler(lnp, prior_transform, model.ndim,
                                     pool=pool,  **kwargs)
    ts = time.time() - t

    return dynestyout, ts
