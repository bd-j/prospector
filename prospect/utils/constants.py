import numpy as np

__all__ == ['lsun', 'pc', 'lightspeed', 'jansky_mks', 'to_cgs', 'loge']


# Useful constants
lsun = 3.846e33
pc = 3.085677581467192e18  # in cm
lightspeed = 2.998e18  # AA/s
jansky_mks = 1e-26
# value to go from L_sun/AA to erg/s/cm^2/AA at 10pc
to_cgs = lsun/(4.0 * np.pi * (pc*10)**2)

# change base
loge = np.log10(np.e)
