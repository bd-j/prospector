import numpy as np
import warnings
from bsfh.readspec import *

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

def load_mock(model, sps, parset):

    mock = None
    obs = {'wavelengths': mock.wavelengths, 'filters': mock.filters}
    model.obs = obs
    s, p, x = model.mean_model(mock.theta, sps=sps)
    p_unc = p / mock.phot_snr
    s_unc = s / mock.spec_snr
    noisy_p = (p + p_unc * np.random.normal(size = len(p)))
    noisy_s = (s + s_unc * np.random.normal(size = len(s)))
    obs['mags'] = -2.5*np.log10(noisy_p)
    obs['mags_unc'] = 1.086 * mock.phot_snr
    obs['spec'] = noisy_s
    obs['mags_unc'] = 1.086 * mock.phot_snr
    model.obs = None
    
    return obs
