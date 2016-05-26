import time, sys, os
import numpy as np
from scipy.linalg import LinAlgError

__all__ = ["lnlike_spec", "lnlike_phot", "write_log"]


def lnlike_spec(spec_mu, obs=None, spec_noise=None, **vectors):
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

        if spec_noise is not None:
            try:
                spec_noise.compute(**vectors)
                return spec_noise.lnlikelihood(delta)
            except(LinAlgError):
                return np.nan_to_num(-np.inf)
        else:
            # simple noise model
            var = (obs['unc'][mask])**2
            lnp = -0.5*( (delta**2/var).sum() + np.log(2*np.pi*var).sum() )
            return lnp


def lnlike_phot(phot_mu, obs=None, phot_noise=None, **vectors):
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

    :param gp: (optional)
        A Gaussian process object with the methods ``compute()`` and
        ``lnlikelihood()``.

    :param fractional:
        Treat the GP amplitudes as additional *fractional* uncertainties,
        i.e., multiplicative uncertainties.

    :returns lnlikelhood:
        The natural logarithm of the likelihood of the data given the mean
        model spectrum.
    """
    if obs['maggies'] is None:
        return 0.0

    mask = obs.get('phot_mask', slice(None))

    delta = (obs['maggies'] - phot_mu)[mask]

    if phot_noise is not None:
        filternames = [f.name for f in obs['filters']]
        vectors['mask'] = mask
        vectors['filternames'] = filternames
        try:
            phot_noise.compute(phot_noise, **vectors)
            return phot_noise.lnlikelihood(delta)
        except(LinAlgError):
            return np.nan_to_num(-np.inf)
    else:
        # simple noise model
        var = (obs['maggies_unc'][mask])**2
        lnp = -0.5*( (delta**2/var).sum() + np.log(2*np.pi*var).sum() )
        return lnp


def write_log(theta, lnp_prior, lnp_spec, lnp_phot, d1, d2):
    """Write all sorts of documentary info for debugging.
    """
    print(theta)
    print('model calc = {0}s, lnlike calc = {1}'.format(d1, d2))
    fstring = 'lnp = {0}, lnp_spec = {1}, lnp_phot = {2}'
    values = [lnp_spec + lnp_phot + lnp_prior, lnp_spec, lnp_phot]
    print(fstring.format(*values))
