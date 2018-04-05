#!/usr/local/bin/python

import time, sys, os
import numpy as np
np.errstate(invalid='ignore')

from prospect.models import model_setup
from prospect.io import write_results
from prospect import fitting
from prospect.likelihood import lnlike_spec, lnlike_phot, write_log, chi_spec, chi_phot


# --------------
# Read command line arguments
# --------------
sargv = sys.argv
argdict = {'param_file': ''}
clargs = model_setup.parse_args(sargv, argdict=argdict)
run_params = model_setup.get_run_params(argv=sargv, **clargs)

# --------------
# Globals
# --------------
# GP instances as global
spec_noise, phot_noise = model_setup.load_gp(**run_params)
# Model as global
global_model = model_setup.load_model(**run_params)
# Obs as global
global_obs = model_setup.load_obs(**run_params)
# SPS Model instance as global
sps = model_setup.load_sps(**run_params)

# -----------------
# LnP function as global
# ------------------

def lnprobfn(theta, model=None, obs=None, residuals=False,
             verbose=run_params['verbose']):
    """Given a parameter vector and optionally a dictionary of observational
    ata and a model object, return the ln of the posterior. This requires that
    an sps object (and if using spectra and gaussian processes, a GP object) be
    instantiated.

    :param theta:
        Input parameter vector, ndarray of shape (ndim,)

    :param model:
        bsfh.sedmodel model object, with attributes including ``params``, a
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
    if model is None:
        model = global_model
    if obs is None:
        obs = global_obs

    # Calculate prior probability and exit if not within prior
    lnp_prior = model.prior_product(theta)
    if not np.isfinite(lnp_prior):
        return -np.infty

    # Generate mean model
    t1 = time.time()
    try:
        spec, phot, x = model.mean_model(theta, obs, sps=sps)
    except(ValueError):
        return -np.infty
    d1 = time.time() - t1

    # Return chi vectors for least-squares optimization
    if residuals:
        chispec = chi_spec(spec, obs)
        chiphot = chi_phot(phot, obs)
        return np.concatenate([chispec, chiphot])
    
    # Noise modeling
    if spec_noise is not None:
        spec_noise.update(**model.params)
    if phot_noise is not None:
        phot_noise.update(**model.params)
    vectors = {'spec': spec, 'unc': obs['unc'],
               'sed': model._spec, 'cal': model._speccal,
               'phot': phot, 'maggies_unc': obs['maggies_unc']}

    # Calculate likelihoods
    t2 = time.time()
    lnp_spec = lnlike_spec(spec, obs=obs, spec_noise=spec_noise, **vectors)
    lnp_phot = lnlike_phot(phot, obs=obs, phot_noise=phot_noise, **vectors)
    d2 = time.time() - t2
    if verbose:
        write_log(theta, lnp_prior, lnp_spec, lnp_phot, d1, d2)

    return lnp_prior + lnp_phot + lnp_spec


def chisqfn(theta, model, obs):
    """Negative of lnprobfn for minimization, and also handles passing in
    keyword arguments which can only be postional arguments when using scipy
    minimize.
    """
    return -lnprobfn(theta, model=model, obs=obs)


def chivecfn(theta):
    """Return the residuals instead of a posterior probability or negative
    chisq, for use with least-squares optimization methods
    """
    return lnprobfn(theta, residuals=True)


# -----------------
# MPI pool.  This must be done *after* lnprob and
# chi2 are defined since slaves will only see up to
# sys.exit()
# ------------------
try:
    from emcee.utils import MPIPool
    pool = MPIPool(debug=False, loadbalance=True)
    if not pool.is_master():
        # Wait for instructions from the master process.
        pool.wait()
        sys.exit(0)
except(ImportError, ValueError):
    pool = None
    print('Not using MPI')


def halt(message):
    """Exit, closing pool safely.
    """
    print(message)
    try:
        pool.close()
    except:
        pass
    sys.exit(0)

# --------------
# Master branch
# --------------

if __name__ == "__main__":

    # --------------
    # Setup
    # --------------
    rp = run_params
    rp['sys.argv'] = sys.argv
    try:
        rp['sps_libraries'] = sps.ssp.libraries
    except(AttributeError):
        rp['sps_libraries'] = None
    # Use the globals
    model = global_model
    obsdat = global_obs
    chi2args = [None, None]
    postkwargs = {}

    # make zeros into tiny numbers
    initial_theta = model.rectify_theta(model.initial_theta)
    if rp.get('debug', False):
        halt('stopping for debug')

    # Try to set up an HDF5 file and write basic info to it
    outroot = "{0}_{1}".format(rp['outfile'], int(time.time()))
    odir = os.path.dirname(os.path.abspath(outroot))
    if (not os.path.exists(odir)):
        halt('Target output directory {} does not exist, please make it.'.format(odir))
    try:
        import h5py
        hfilename = outroot + '_mcmc.h5'
        hfile = h5py.File(hfilename, "a")
        print("Writing to file {}".format(hfilename))
        write_results.write_h5_header(hfile, run_params, model)
        write_results.write_obs_to_h5(hfile, obsdat)
    except(ImportError):
        hfile = None

    # -----------------------------------------
    # Initial guesses using minimization
    # -----------------------------------------
    if rp['verbose']:
        print('Starting minimization...')

    if not np.isfinite(model.prior_product(model.initial_theta.copy())):
        halt("Halting: initial parameter position has zero prior probability.")

    if bool(rp.get('do_powell', False)):
        ts = time.time()
        powell_opt = {'ftol': rp['ftol'], 'xtol': 1e-6, 'maxfev': rp['maxfev']}
        guesses, pinit = fitting.pminimize(chisqfn, initial_theta,
                                           args=chi2args, model=model,
                                           method='powell', opts=powell_opt,
                                           pool=pool, nthreads=rp.get('nthreads', 1))
        best = np.argmin([p.fun for p in guesses])
        initial_center = fitting.reinitialize(guesses[best].x, model,
                                              edge_trunc=rp.get('edge_trunc', 0.1))
        initial_prob = -guesses[best]['fun']
        pdur = time.time() - ts
        if rp['verbose']:
            print('done Powell in {0}s'.format(pdur))
            print('best Powell guess:{0}'.format(initial_center))

    elif bool(rp.get('do_levenberg', False)):
        from scipy.optimize import least_squares
        nmin = rp.get('nmin', 10)
        ts = time.time()
        pinitial = fitting.minimizer_ball(model.initial_theta.copy(), nmin, model)
        guesses = []
        for i, pinit in enumerate(pinitial):
            res = least_squares(chivecfn, pinit, method='lm', x_scale='jac',
                                xtol=1e-18, ftol=1e-18)
            guesses.append(res)

        chisq = [np.sum(r.fun**2) for r in guesses]
        best = np.argmin(chisq)
        initial_center = fitting.reinitialize(guesses[best].x, model,
                                              edge_trunc=rp.get('edge_trunc', 0.1))
        initial_prob = None
        pdur = time.time() - ts
        if rp['verbose']:
            print('done L-M in {0}s'.format(pdur))
            print('best L-M guess:{0}'.format(initial_center))

    else:
        if rp['verbose']:
            print('No minimization requested.')
        guesses = None
        pdur = 0.0
        initial_center = initial_theta.copy()
        initial_prob = None

    # ---------------------
    # Sampling
    # -----------------------
    if rp['verbose']:
        print('emcee sampling...')
    tstart = time.time()
    out = fitting.run_emcee_sampler(lnprobfn, initial_center, model,
                                    postkwargs=postkwargs, initial_prob=initial_prob,
                                    pool=pool, hdf5=hfile, **rp)
    esampler, burn_p0, burn_prob0 = out
    edur = time.time() - tstart
    if rp['verbose']:
        print('done emcee in {0}s'.format(edur))

    # -------------------------
    # Output HDF5 (and pickles if asked for)
    # -------------------------
    print("Writing to {}".format(outroot))
    if rp.get("output_pickles", False):
        write_results.write_pickles(rp, model, obsdat, esampler, guesses,
                                    outroot=outroot, toptimize=pdur, tsample=edur,
                                    sampling_initial_center=initial_center,
                                    post_burnin_center=burn_p0,
                                    post_burnin_prob=burn_prob0)
    if hfile is None:
        hfile = hfilename
    write_results.write_hdf5(hfile, rp, model, obsdat, esampler, guesses,
                             toptimize=pdur, tsample=edur,
                             sampling_initial_center=initial_center,
                             post_burnin_center=burn_p0,
                             post_burnin_prob=burn_prob0)
    try:
        hfile.close()
    except:
        pass
    halt('Finished')
