import numpy as np
import warnings
from .readspec import *

def logify(data, sigma, mask):
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

def norm_spectrum(model, initial_center, norm_band_name='f475w',
                  **kwargs):
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
    
    norm_band = [i for i,f in enumerate(model.obs['filters'])
                 if norm_band_name in f.name][0]
    
    synphot = observate.getSED(model.obs['wavelength'],
                               model.obs['spectrum'],
                               model.obs['filters'])

    # Factor by which the observed spectra should be *divided* to give
    #  you the photometry (or the cgs apparent spectrum), using the
    #  given filter as truth.  Alternatively, the factor by which the
    #  model spectrum (in cgs apparent) should be multiplied to give
    #  you the observed spectrum.
    norm = 10**(-0.4 * synphot[norm_band]) / model.obs['maggies'][norm_band]
    model.params['normalization_guess'] = norm
       
    # Assume you've got this right to within some factor after
    #  marginalized over everything that changes spectral shape within
    #  the band (polynomial terms, dust, age, etc)
    initial_center[model.theta_desc['spec_norm']['i0']] = 1.0
    fudge = 3.0
    model.theta_desc['spec_norm']['prior_args'] = {'mini':1 / fudge,
                                                   'maxi':fudge }

    # Pivot the calibration polynomial near the filter used for approximate
    # normalization
    model.params['pivot_wave'] =  model.obs['filters'][norm_band].wave_effective 
    #model.params['pivot_wave'] = 4750.
 
    return model, initial_center

def generate_mock(model, sps, mock_info):
    """
    Generate a mock data set given model, mock parameters, wavelength
    grid, and filter set.
    """

    # Generate mock spectrum and photometry given mock parameters, and
    # Apply calibration
    obs = {'wavelength': mock_info['wavelength'],
           'filters': mock_info['filters']}
    model.obs = obs
    for k, v in mock_info['params'].iteritems():
        model.params[k] = np.atleast_1d(v)
    mock_theta = model.theta_from_params()
    #print('generate_mock: mock_theta={}'.format(mock_theta))
    s, p, x = model.mean_model(mock_theta, sps=sps)
    cal = model.calibration(mock_theta)
    if mock_info.get('calibration',None) is not None:
        cal = mock_info['calibration']
    s *= cal
    
    # Add noise to the mock data
    if mock_info['filters'] is not None:
        p_unc = p / mock_info['phot_snr']
        noisy_p = (p + p_unc * np.random.normal(size = len(p)))
        obs['maggies'] = noisy_p
        obs['maggies_unc'] = p_unc
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
