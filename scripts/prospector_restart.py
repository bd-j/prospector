#!/usr/local/bin/python

import time, sys, os
import numpy as np
np.errstate(invalid='ignore')

from prospect.models import model_setup
from prospect.io import write_results
from prospect.io import read_results as pr
from prospect import fitting
from prospect.likelihood import lnlike_spec, lnlike_phot, write_log, chi_spec, chi_phot


# --------------
# Read command line arguments
# --------------
sargv = sys.argv
argdict = {'restart_from': '', 'niter': 1024}
clargs = model_setup.parse_args(sargv, argdict=argdict)

# ----------
# Result object and Globals
# ----------
result, global_obs, global_model = pr.results_from(clargs["restart_from"])
is_emcee = (len(result["chain"].shape) == 3) & (result["chain"].shape[0] > 1)
assert is_emcee, "Result file does not have a chain of the proper shape."

# SPS Model instance (with libraries check)
sps = pr.get_sps(result)
run_params = result["run_params"]
run_params.update(clargs)

# Noise model (this should be doable via read_results)
from prospect.models.model_setup import import_module_from_string
param_file = (result['run_params'].get('param_file', ''),
              result.get("paramfile_text", ''))
path, filename = os.path.split(param_file[0])
modname = filename.replace('.py', '')
user_module = import_module_from_string(param_file[1], modname)
spec_noise, phot_noise = user_module.load_gp(**run_params)

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
    postkwargs = {}

    # make zeros into tiny numbers
    initial_theta = model.rectify_theta(model.initial_theta)
    if rp.get('debug', False):
        halt('stopping for debug')

    # Try to set up an HDF5 file and write basic info to it
    outroot = "{}_restart_{}".format(rp['outfile'], int(time.time()))
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
    # Initial guesses from end of last chain
    # -----------------------------------------

    initial_positions = result["chain"][:, -1, :]
    guesses = None
    initial_center = initial_positions.mean(axis=0)

    # ---------------------
    # Sampling
    # -----------------------
    if rp['verbose']:
        print('emcee sampling...')
    tstart = time.time()
    out = fitting.restart_emcee_sampler(lnprobfn, initial_positions,
                                        postkwargs=postkwargs,
                                        pool=pool, hdf5=hfile, **rp)
    esampler = out
    edur = time.time() - tstart
    if rp['verbose']:
        print('done emcee in {0}s'.format(edur))

    # -------------------------
    # Output HDF5 (and pickles if asked for)
    # -------------------------
    print("Writing to {}".format(outroot))
    if rp.get("output_pickles", False):
        write_results.write_pickles(rp, model, obsdat, esampler, guesses,
                                    outroot=outroot, toptimize=0, tsample=edur,
                                    sampling_initial_center=initial_center)
    if hfile is None:
        hfile = hfilename
    write_results.write_hdf5(hfile, rp, model, obsdat, esampler, guesses,
                             toptimize=0, tsample=edur,
                             sampling_initial_center=initial_center)
    try:
        hfile.close()
    except:
        pass
    halt('Finished')
