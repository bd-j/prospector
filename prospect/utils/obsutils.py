#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" obsutils.py - Utilities for manipulating observational data, especially
ensuring the the required keys are present in the `obs` dictionary.
"""

from copy import deepcopy
import numpy as np
import warnings
np.errstate(invalid='ignore')

__all__ = ["fix_obs", "rectify_obs", "norm_spectrum", "logify_data"]


def fix_obs(obs, rescale_spectrum=False, normalize_spectrum=False,
            logify_spectrum=False, grid_filters=False, **kwargs):
    """Make all required changes to the obs dictionary.

    :param obs:
        The `obs` dictionary that will be fit.

    :param rescale_spectrum: (optional, default:False, deprecated)
        Rescale the supplied spectrum to have a median value of 1.  The value
        used to rescale the spectrum is added as the `"rescale"` key in the
        supplied `obs` dictionary.

    :param normalize_spectrum: (optional, default:False, deprecated)
        Renormalize the spectrum to give the supplied magnitude through the
        filter specified by `obs["norm_band_name"]`.  See `norm_spectrum()` for
        details.

    :param logify_spectrum: (optional, default:False, deprecated)
        Take the log of the spectrum and associated uncertainties, for fitting
        in log-space.  Note this distorts errors.

    :param grid_filters: (optional, default:False)
        Switch to place all filter transmission curves on a common grid of
        dlnlambda, to provide small speed gains in the filter projections.  The
        grid is calculated automatically from the supplied filters, and is
        added to the `obs` dictionary as the `"lnwavegrid"` key.

    :returns obs:
        An obs dictionary that has all required keys and that has been modified
        according to the options described above.

    """
    obs = rectify_obs(obs)
    obs['ndof'] = 0
    if obs['spectrum'] is not None:
        obs['ndof'] += int(obs['mask'].sum())
        if (rescale_spectrum):
            sc = np.median(obs['spectrum'][obs['mask']])
            obs['rescale'] = sc
            obs['spectrum'] /= sc
            obs['unc'] /= sc
        if (normalize_spectrum):
            sp_norm, pivot_wave = norm_spectrum(obs, **kwargs)
            obs['normalization_guess'] = sp_norm
            obs['pivot_wave'] = pivot_wave
        if (logify_spectrum):
            s, u, m = logify_data(obs['spectrum'], obs['unc'], obs['mask'])
            obs['spectrum'] = s
            obs['unc'] = u
            obs['mask'] = m
            obs['logify_spectrum'] = True
    else:
        obs['unc'] = None

    if obs['maggies'] is not None:
        obs['ndof'] += int(obs['phot_mask'].sum())
        if grid_filters:
            from sedpy.observate import FilterSet
            fset = FilterSet([f.name for f in obs["filters"]])
            obs["filters"] = fset
            obs['lnwavegrid'] = fset.lnlam.copy()
    else:
        obs['maggies_unc'] = None

    assert obs["ndof"] > 0, "No valid data to fit: check the sign of the masks."

    return obs


def logify_data(data, sigma, mask):
    """Convert data to ln(data) and uncertainty to fractional uncertainty for
    use in additive GP models.  This involves filtering for negative data
    values and replacing them with something else.
    """
    tiny = 0.01 * data[data > 0].min()
    bad = data < tiny
    nbad = bad.sum()
    if nbad == 0:
        return np.log(data), sigma/data, mask
    else:
        message = ("Setting {0} datapoints to {1} to insure "
                   "positivity.".format(nbad, tiny))
        # warnings.warn(message)
        print(message)
        data[bad] = tiny
        sigma[bad] = np.sqrt(sigma[bad]**2 + (data[bad] - tiny)**2)
        return np.log(data), sigma/data, mask


def norm_spectrum(obs, norm_band_name='f475w', **kwargs):
    """Initial guess of spectral normalization using photometry.

    This obtains the multiplicative factor required to reproduce the the
    photometry in one band from the observed spectrum (model.obs['spectrum'])
    using the bsfh unit conventions.  Thus multiplying the observed spectrum by
    this factor gives a spectrum that is approximately erg/s/cm^2/AA at the
    central wavelength of the normalizing band.

    The inverse of the multiplication factor is saved as a fixed parameter to
    be used in producing the mean model.
    """
    from sedpy import observate

    try:
        flist = obs["filters"].filters
    except(AttributeError):
        flist = obs["filters"]

    norm_band = [i for i, f in enumerate(flist)
                 if norm_band_name in f.name][0]

    synphot = observate.getSED(obs['wavelength'], obs['spectrum'], obs['filters'])
    synphot = np.atleast_1d(synphot)
    # Factor by which the observed spectra should be *divided* to give you the
    #  photometry (or the cgs apparent spectrum), using the given filter as
    #  truth.  Alternatively, the factor by which the model spectrum (in cgs
    #  apparent) should be multiplied to give you the observed spectrum.
    norm = 10**(-0.4 * synphot[norm_band]) / obs['maggies'][norm_band]

    # Pivot the calibration polynomial near the filter used for approximate
    # normalization
    pivot_wave = flist[norm_band].wave_effective
    # model.params['pivot_wave'] = 4750.

    return norm, pivot_wave


def rectify_obs(obs):
    """Make sure the passed obs dictionary conforms to code expectations,
    and make simple fixes when possible.
    """
    k = obs.keys()
    if 'maggies' not in k:
        obs['maggies'] = None
        obs['maggies_unc'] = None
    if 'spectrum' not in k:
        obs['spectrum'] = None
        obs['unc'] = None
    if obs['maggies'] is not None:
        assert (len(obs['filters']) == len(obs['maggies']))
        assert ('maggies_unc' in k)
        assert ((len(obs['maggies']) == len(obs['maggies_unc'])) or
                (np.size(obs['maggies_unc'] == 1)))
        m = obs.get('phot_mask', np.ones(len(obs['maggies']), dtype=bool))
        obs['phot_mask'] = (m * np.isfinite(obs['maggies']) *
                            np.isfinite(obs['maggies_unc']) *
                            (obs['maggies_unc'] > 0))
        try:
            obs['filternames'] = obs["filters"].filternames
        except(AttributeError):
            obs['filternames'] = [f.name for f in obs['filters']]
        except:
            pass

    if 'logify_spectrum' not in k:
        obs['logify_spectrum'] = False
    if obs['spectrum'] is not None:
        assert (len(obs['wavelength']) == len(obs['spectrum']))
        assert ('unc' in k)
        np.errstate(invalid='ignore')
        m = obs.get('mask', np.ones(len(obs['wavelength']), dtype=bool))
        obs['mask'] = (m * np.isfinite(obs['spectrum']) *
                       np.isfinite(obs['unc']) * (obs['unc'] > 0))
    return obs


def build_mock(sps, model,
               filterset=None,
               wavelength=None,
               snr_spec=10.0, snr_phot=20., add_noise=False,
               seed=101, **model_params):
    """Make a mock dataset.  Feel free to add more complicated kwargs, and put
    other things in the run_params dictionary to control how the mock is
    generated.
    :param filterset:
        A FilterSet or list of Filters
    :param wavelength:
        A vector
    :param snr_phot:
        The S/N of the phock photometry.  This can also be a vector of same
        lngth as the number of filters, for heteroscedastic noise.
    :param snr_spec:
        The S/N of the phock spectroscopy.  This can also be a vector of same
        lngth as `wavelength`, for heteroscedastic noise.
    :param add_noise: (optional, boolean, default: True)
        If True, add a realization of the noise to the mock photometry.
    :param seed: (optional, int, default: 101)
        If greater than 0, use this seed in the RNG to get a deterministic
        noise for adding to the mock data.
    """
    # We'll put the mock data in this dictionary, just as we would for real
    # data.  But we need to know which filters (and wavelengths if doing
    # spectroscopy) with which to generate mock data.
    mock = {"filters": None, "maggies": None, "wavelength": None, "spectrum": None}
    mock['wavelength'] = wavelength
    if filterset is not None:
        mock['filters'] = filterset

    # Now we get any mock params from the model_params dict
    params = {}
    for p in model.params.keys():
        if p in model_params:
            params[p] = np.atleast_1d(model_params[p])

    # And build the mock
    model.params.update(params)
    spec, phot, mfrac = model.predict(model.theta, mock, sps=sps)

    # Now store some output
    mock['true_spectrum'] = spec.copy()
    mock['true_maggies'] = np.copy(phot)
    mock['mock_params'] = deepcopy(model.params)

    # store the mock photometry
    if filterset is not None:
        pnoise_sigma = phot / snr_phot
        mock['maggies'] = phot.copy()
        mock['maggies_unc'] = pnoise_sigma
        mock['mock_snr_phot'] = snr_phot
        # And add noise
        if add_noise:
            if int(seed) > 0:
                np.random.seed(int(seed))
            pnoise = np.random.normal(0, 1, size=len(phot)) * pnoise_sigma
            mock['maggies'] += pnoise

        try:
            flist = mock["filters"].filters
        except(AttributeError):
            flist = mock["filters"]
        mock['phot_wave'] = np.array([f.wave_effective for f in flist])

    # store the mock spectrum
    if wavelength is not None:
        snoise_sigma = spec / snr_spec
        mock['spectrum'] = spec.copy()
        mock['unc'] = snoise_sigma
        mock['mock_snr_spec'] = snr_spec
        # And add noise
        if add_noise:
            if int(seed) > 0:
                np.random.seed(int(seed))
            snoise = np.random.normal(0, 1, size=len(spec)) * snoise_sigma
            mock['spectrum'] += snoise

    return mock
