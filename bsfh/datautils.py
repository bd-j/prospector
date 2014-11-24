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
        obs['spec'] = noisy_s
        obs['unc'] = s_unc
    else:
        obs['spec'] = None

    #obs['mock_params'] = model.params
    model.obs  = None
    
    return obs
