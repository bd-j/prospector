#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" obsutils.py - Utilities for manipulating observational data, especially
ensuring the the required keys are present in the `obs` dictionary.
"""

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
            wlo, whi, dlo = [], [], []
            for f in obs['filters']:
                dlnlam = np.gradient(f.wavelength)/f.wavelength
                wlo.append(f.wavelength.min())
                dlo.append(dlnlam.min())
                whi.append(f.wavelength.max())
            wmin = np.min(wlo)
            wmax = np.max(whi)
            dlnlam = np.min(dlo)
            for f in obs['filters']:
                f.gridify_transmission(dlnlam, wmin)
                f.get_properties()
            obs['lnwavegrid'] = np.exp(np.arange(np.log(wmin), np.log(wmax)+dlnlam, dlnlam))
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

    norm_band = [i for i, f in enumerate(obs['filters'])
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
    pivot_wave = obs['filters'][norm_band].wave_effective
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


def generate_mock(model, sps, mock_info):
    """Generate a mock data set given model, mock parameters, wavelength grid,
    and filter set.  Very old and unused.
    """
    # Generate mock spectrum and photometry given mock parameters, and
    # Apply calibration.

    # NEED TO DEAL WITH FILTERNAMES ADDED FOR SPS_BASIS
    obs = {'wavelength': mock_info['wavelength'],
           'filters': mock_info['filters']}
    model.configure(**mock_info['params'])
    mock_theta = model.theta.copy()
    # print('generate_mock: mock_theta={}'.format(mock_theta))
    s, p, x = model.mean_model(mock_theta, obs, sps=sps)
    cal = model.calibration(mock_theta, obs)
    if 'calibration' in mock_info:
        cal = mock_info['calibration']
    s *= cal
    model.configure()
    # Add noise to the mock data
    if mock_info['filters'] is not None:
        p_unc = p / mock_info['phot_snr']
        noisy_p = (p + p_unc * np.random.normal(size=len(p)))
        obs['maggies'] = noisy_p
        obs['maggies_unc'] = p_unc
        obs['phot_mask'] = np.ones(len(obs['filters']), dtype=bool)
    else:
        obs['maggies'] = None
    if mock_info['wavelength'] is not None:
        s_unc = s / mock_info.get('spec_snr', 10.0)
        noisy_s = (s + s_unc * np.random.normal(size=len(s)))
        obs['spectrum'] = noisy_s
        obs['unc'] = s_unc
        obs['mask'] = np.ones(len(obs['wavelength']), dtype=bool)
    else:
        obs['spectrum'] = None
        obs['mask'] = None

    return obs
