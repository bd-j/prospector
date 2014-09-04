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
    


        
