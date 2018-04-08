import numpy as np

try:
    from astropy.cosmology import WMAP9 as cosmo
except(ImportError):
    cosmo = None

__all__ = ['lsun', 'pc', 'lightspeed', 'ckms',
           'jansky_mks', 'jansky_cgs',
           'to_cgs_at_10pc', 'loge',
           'kboltz', 'hplanck',
           'cosmo']

# Useful constants
lsun = 3.846e33  # erg/s
pc = 3.085677581467192e18  # in cm
lightspeed = 2.998e18  # AA/s
ckms = 2.998e5  # km/s
jansky_mks = 1e-26
jansky_cgs = 1e-23
# value to go from L_sun/AA to erg/s/cm^2/AA at 10pc
to_cgs_at_10pc = lsun / (4.0 * np.pi * (pc*10)**2)

# cgs physical constants
kboltz = 1.3806488e-16
hplanck = 6.62606957e-27

# change base
loge = np.log10(np.e)
