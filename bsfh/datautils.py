import numpy as np
import warnings
from .readspec import *

def logify_data(data, sigma, mask):
    """
    Convert data to ln(data) and uncertainty to fractional uncertainty
    for use in additive GP models.  This involves filtering for
    negative data values and replacing them with something else.
    """

    tiny = 0.01 * data[data > 0].min()
    bad = data < tiny
    nbad = bad.sum()
    
    if nbad == 0:
        return np.log(data), sigma/data, mask
    
    else:
        message = "Setting {0} datapoints to {1} to insure positivity.".format(nbad, tiny)
        #warnings.warn(message)
        print(message)
        data[bad] = tiny
        sigma[bad] = np.sqrt(sigma[bad]**2 + (data[bad] - tiny)**2)
        return np.log(data), sigma/data, mask    

def norm_spectrum(obs, norm_band_name='f475w', **kwargs):
    """
    Initial guess of spectral normalization using photometry.

    This obtains the multiplicative factor required to reproduce the
    the photometry in one band from the observed spectrum
    (model.obs['spectrum']) using the bsfh unit conventions.  Thus
    multiplying the observed spectrum by this factor gives a spectrum
    that is approximately erg/s/cm^2/AA at the central wavelength of
    the normalizing band.

    The inverse of the multiplication factor is saved as a fixed
    parameter to be used in producing the mean model.
    """
    from sedpy import observate
    
    norm_band = [i for i,f in enumerate(obs['filters'])
                 if norm_band_name in f.name][0]
    
    synphot = observate.getSED(obs['wavelength'],
                               obs['spectrum'],
                               obs['filters'])

    # Factor by which the observed spectra should be *divided* to give
    #  you the photometry (or the cgs apparent spectrum), using the
    #  given filter as truth.  Alternatively, the factor by which the
    #  model spectrum (in cgs apparent) should be multiplied to give
    #  you the observed spectrum.
    norm = 10**(-0.4 * synphot[norm_band]) / obs['maggies'][norm_band]
       
    # Pivot the calibration polynomial near the filter used for approximate
    # normalization
    pivot_wave =  obs['filters'][norm_band].wave_effective 
    #model.params['pivot_wave'] = 4750.
 
    return norm, pivot_wave

def rectify_obs(obs):
    """
    Make sure the passed obs dictionary conforms to code expectations,
    and make simple fixes when possible.
    """
    k = obs.keys()
    if 'maggies' not in k:
        obs['maggies'] = None
    if 'spectrum' not in k:
        obs['spectrum'] = None
        obs['unc'] = None
    if obs['maggies'] is not None:
        assert (len(obs['filters']) == len(obs['maggies']))
        assert ('maggies_unc' in k)
        assert (( len(obs['maggies']) == len(obs['maggies_unc'])) or
                 (np.size(obs['maggies_unc'] == 1)))
        obs['phot_mask'] = (obs.get('phot_mask',
                                    np.ones(len(obs['maggies']), dtype= bool)) *
                            np.isfinite(obs['maggies']) *
                            np.isfinite(obs['maggies_unc']) *
                            (obs['maggies_unc'] > 0))

    if obs['spectrum'] is not None:
        assert (len(obs['wavelength']) == len(obs['spectrum']))
        assert ('unc' in k)
        obs['mask'] = (obs.get('mask',
                               np.ones(len(obs['wavelength']), dtype= bool)) *
                            np.isfinite(obs['spectrum']) *
                            np.isfinite(obs['unc']) * (obs['unc'] > 0))
    return obs
        
def generate_mock(model, sps, mock_info):
    """
    Generate a mock data set given model, mock parameters, wavelength
    grid, and filter set.
    """

    # Generate mock spectrum and photometry given mock parameters, and
    # Apply calibration.

    #NEED TO DEAL WITH FILTERNAMES ADDED FOR SPS_BASIS 
    obs = {'wavelength': mock_info['wavelength'],
           'filters': mock_info['filters']}
    model.obs = obs
    model.configure(**mock_info['params'])
    mock_theta = model.theta
    #print('generate_mock: mock_theta={}'.format(mock_theta))
    s, p, x = model.mean_model(mock_theta, sps=sps)
    cal = model.calibration(mock_theta)
    if 'calibration' in mock_info:
        cal = mock_info['calibration']
    s *= cal
    
    # Add noise to the mock data
    if mock_info['filters'] is not None:
        p_unc = p / mock_info['phot_snr']
        noisy_p = (p + p_unc * np.random.normal(size = len(p)))
        obs['maggies'] = noisy_p
        obs['maggies_unc'] = p_unc
        obs['phot_mask'] = np.ones(len(obs['filters']), dtype= bool)
    else:
        obs['maggies'] = None
    if mock_info['wavelength'] is not None:
        s_unc = s / mock_info.get('spec_snr', 10.0)
        noisy_s = (s + s_unc * np.random.normal(size = len(s)))
        obs['spectrum'] = noisy_s
        obs['unc'] = s_unc
        obs['mask'] = np.ones(len(obs['wavelength']), dtype= bool)
    else:
        obs['spectrum'] = None
        obs['mask'] = None
    #obs['mock_params'] = model.params
    model.obs  = None
    
    return obs
