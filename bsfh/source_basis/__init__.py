import numpy as np
from .galaxy_basis import *
from .star_basis import *

__all__ = ["StellarPopBasis", "CSPBasis", "StarBasis", "BigStarBasis"]

# Useful constants
lsun = 3.846e33
pc = 3.085677581467192e18  # in cm
lightspeed = 2.998e18  # AA/s
# value to go from L_sun/AA to erg/s/cm^2/AA at 10pc
to_cgs = lsun/(4.0 * np.pi * (pc*10)**2)
