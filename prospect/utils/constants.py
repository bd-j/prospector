import numpy as np

__all__ == ['lsun', 'pc', 'lightspeed', 'ckms',
            'jansky_mks', 'jansky_cgs',
            'to_cgs_at10pc', 'loge']


# Useful constants
lsun = 3.846e33
pc = 3.085677581467192e18  # in cm
lightspeed = 2.998e18  # AA/s
ckms = 2.998e5 # km/s
jansky_mks = 1e-26
jansky_cgs = 1e-23
# value to go from L_sun/AA to erg/s/cm^2/AA at 10pc
to_cgs_at10pc  = lsun / (4.0 * np.pi * (pc*10)**2)

# change base
loge = np.log10(np.e)
