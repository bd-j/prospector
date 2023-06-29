import time, sys, os
import numpy as np
from scipy.linalg import LinAlgError

__all__ = ["lnlike_spec", "lnlike_phot", "chi_spec", "chi_phot", "write_log"]


def lnlike_spec(spec_mu, obs=None, spec_noise=None, f_outlier_spec=0.0, **vectors):
    """Calculate the likelihood of the spectroscopic data given the
    spectroscopic model.  Allows for the use of a gaussian process
    covariance matrix for multiplicative residuals.

    :param spec_mu:
        The mean model spectrum, in linear or logarithmic units, including
        e.g. calibration and sky emission.

    :param obs: (optional)
        A dictionary of the observational data, including the keys
        *``spectrum`` a numpy array of the observed spectrum, in linear or
         logarithmic units (same as ``spec_mu``).
        *``unc`` the uncertainty of same length as ``spectrum``
        *``mask`` optional boolean array of same length as ``spectrum``
        *``wavelength`` if using a GP, the metric that is used in the
         kernel generation, of same length as ``spectrum`` and typically
         giving the wavelength array.

    :param spec_noise: (optional)
        A NoiseModel object with the methods `compute` and `lnlikelihood`.
        If ``spec_noise`` is supplied, the `wavelength` entry in the obs
        dictionary must exist.

    :param f_outlier_spec: (optional)
        The fraction of spectral pixels which are considered outliers
        by the mixture model

    :param vectors: (optional)
        A dictionary of vectors of same length as ``wavelength`` giving
        possible weghting functions for the kernels

    :returns lnlikelhood:
        The natural logarithm of the likelihood of the data given the mean
        model spectrum.
    """
    if obs['spectrum'] is None:
        return 0.0

    mask = obs.get('mask', slice(None))
    vectors['mask'] = mask
    vectors['wavelength'] = obs['wavelength']

    delta = (obs['spectrum'] - spec_mu)[mask]
    var = (obs['unc'][mask])**2

    if spec_noise is not None:
        try:
            spec_noise.compute(**vectors)
            if (f_outlier_spec == 0.0):
                return spec_noise.lnlikelihood(spec_mu[mask], obs['spectrum'][mask])

            # disallow (correlated noise model + mixture model)
            # and redefine errors
            assert spec_noise.Sigma.ndim == 1
            var = spec_noise.Sigma

        except(LinAlgError):
            return np.nan_to_num(-np.inf)

    lnp = -0.5*( (delta**2/var) + np.log(2*np.pi*var) )
    if (f_outlier_spec == 0.0):
        return lnp.sum()
    else:
        var_bad = var * (vectors["nsigma_outlier_spec"]**2)
        lnp_bad = -0.5*( (delta**2/var_bad) + np.log(2*np.pi*var_bad) )
        lnp_tot = np.logaddexp(lnp + np.log(1-f_outlier_spec), lnp_bad + np.log(f_outlier_spec))

        return lnp_tot.sum()


def lnlike_phot(phot_mu, obs=None, phot_noise=None, f_outlier_phot=0.0, **vectors):
    """Calculate the likelihood of the photometric data given the spectroscopic
    model.  Allows for the use of a gaussian process covariance matrix.

    :param phot_mu:
        The mean model sed, in linear flux units (i.e. maggies).

    :param obs: (optional)
        A dictionary of the observational data, including the keys
          *``maggies`` a numpy array of the observed SED, in linear flux
           units
          *``maggies_unc`` the uncertainty of same length as ``maggies``
          *``phot_mask`` optional boolean array of same length as
           ``maggies``
          *``filters`` optional list of sedpy.observate.Filter objects,
           necessary if using fixed filter groups with different gp
           amplitudes for each group.
       If not supplied then the obs dictionary given at initialization will
       be used.

    :param phot_noise: (optional)
        A ``prospect.likelihood.NoiseModel`` object with the methods
        ``compute()`` and ``lnlikelihood()``.  If not supplied a simple chi^2
        likelihood will be evaluated.

    :param f_outlier_phot: (optional)
        The fraction of photometric bands which are considered outliers
        by the mixture model

    :param vectors:
        A dictionary of possibly relevant vectors of same length as maggies
        that will be passed to the NoiseModel object for constructing weighted
        covariance matrices.

    :returns lnlikelhood:
        The natural logarithm of the likelihood of the data given the mean
        model spectrum.
    """
    if obs['maggies'] is None:
        return 0.0

    mask = obs.get('phot_mask', slice(None))
    delta = (obs['maggies'] - phot_mu)[mask]
    var = (obs['maggies_unc'][mask])**2
    psamples = obs.get('phot_samples', None)

    if phot_noise is not None:
        try:
            filternames = obs['filters'].filternames
        except(AttributeError):
            filternames = [f.name for f in obs['filters']]
        vectors['mask'] = mask
        vectors['filternames'] = np.array(filternames)
        vectors['phot_samples'] = psamples
        try:
            phot_noise.compute(**vectors)
            if (f_outlier_phot == 0.0):
                return phot_noise.lnlikelihood(phot_mu[mask], obs['maggies'][mask])

            # disallow (correlated noise model + mixture model)
            # and redefine errors
            assert phot_noise.Sigma.ndim == 1
            var = phot_noise.Sigma

        except(LinAlgError):
            return np.nan_to_num(-np.inf)

    # simple noise model
    lnp = -0.5*( (delta**2/var) + np.log(2*np.pi*var) )
    if (f_outlier_phot == 0.0):
        return lnp.sum()
    else:
        var_bad = var * (vectors["nsigma_outlier_phot"]**2)
        lnp_bad = -0.5*( (delta**2/var_bad) + np.log(2*np.pi*var_bad) )
        lnp_tot = np.logaddexp(lnp + np.log(1-f_outlier_phot), lnp_bad + np.log(f_outlier_phot))

        return lnp_tot.sum()


def chi_phot(phot_mu, obs, **extras):
    """Return a vector of chi values, for use in non-linear least-squares
    algorithms.

    :param phot_mu:
        Model photometry, same units as the photometry in `obs`.

    :param obs:
        An observational data dictionary, with the keys ``"maggies"`` and
        ``"maggies_unc"``.  If ``"maggies"`` is None then an empty array is
        returned.

    :returns chi:
        An array of noise weighted residuals, same length as the number of
        unmasked phtometric points.
    """
    if obs['maggies'] is None:
        return np.array([])

    mask = obs.get('phot_mask', slice(None))
    delta = (obs['maggies'] - phot_mu)[mask]
    unc = obs['maggies_unc'][mask]
    chi = delta / unc
    return chi


def chi_spec(spec_mu, obs, **extras):
    """Return a vector of chi values, for use in non-linear least-squares
    algorithms.

    :param spec_mu:
        Model spectroscopy, same units as the photometry in `obs`.

    :param obs:
        An observational data dictionary, with the keys ``"spectrum"`` and
        ``"unc"``.  If ``"spectrum"`` is None then an empty array is returned.
        Optinally a ``"mask"`` boolean vector may be supplied that will be used
        to index the residual vector.

    :returns chi:
        An array of noise weighted residuals, same length as the number of
        unmasked spectroscopic points.
    """
    if obs['spectrum'] is None:
        return np.array([])
    mask = obs.get('mask', slice(None))
    delta = (obs['spectrum'] - spec_mu)[mask]
    unc = obs['unc'][mask]
    chi = delta / unc
    return chi


def write_log(theta, lnp_prior, lnp_spec, lnp_phot, d1, d2):
    """Write all sorts of documentary info for debugging.
    """
    print(theta)
    print('model calc = {0}s, lnlike calc = {1}'.format(d1, d2))
    fstring = 'lnp = {0}, lnp_spec = {1}, lnp_phot = {2}'
    values = [lnp_spec + lnp_phot + lnp_prior, lnp_spec, lnp_phot]
    print(fstring.format(*values))
