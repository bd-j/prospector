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
        warnings.warn("Setting {0} datapoints to \
        {1} to insure positivity.".format(nbad, tiny))
        data[bad] = tiny
        sigma[bad] = np.sqrt(sigma[bad]**2 + (data[bad] - tiny)**2)
        return np.log(data), sigma/data, mask    
    
def generate_mock(model, sps, mock):
    """
    Generate a mock data set given model, mock parameters, wavelength
    grid, and filter set.
    """
    
    obs = {'wavelength': mock['wavelength'], 'filters': mock['filters']}
    model.obs = obs
    for k, v in mock['params'].iteritems():
        model.params[k] = np.atleast_1d(v)
    mock_theta = model.theta_from_params()
    s, p, x = model.mean_model(mock_theta, sps=sps)
    
    if mock['filters'] is not None:
        p_unc = p / mock['phot_snr']
        noisy_p = (p + p_unc * np.random.normal(size = len(p)))
        obs['mags'] = -2.5*np.log10(noisy_p)
        obs['mags_unc'] = 1.086 * p_unc/p
    else:
        obs['mags'] = None
        
    if mock['wavelength'] is not None:
        s_unc = s / mock.get('spec_snr', 1.0)
        noisy_s = (s + s_unc * np.random.normal(size = len(s)))
        obs['spec'] = noisy_s
        obs['unc'] = s_unc
    else:
        obs['spec'] = None
        
    model.obs  = None
    
    return obs
