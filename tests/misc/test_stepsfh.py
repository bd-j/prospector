import matplotlib.pyplot as pl
import numpy as np
from bsfh.sources import ssp_basis

sps = ssp_basis.StepSFHBasis(interp_type='logarithmic')

nbin = 7
ages = np.linspace(7, 10, nbin+1)
params = {'agebins': np.array([ages[:-1], ages[1:]]).T,
          'mass': 10**ages[1:] - 10**ages[:-1] }


sps.update(**params)

w = sps.ssp_weights

pl.plot(sps.logage, w)
agebins = sps.params['agebins']
masses = sps.params['mass']
#w = np.zeros(len(self.logage))
for (t1, t2), mass in zip(agebins, masses):
    print(t1, t2, mass)
    pl.axvline(t1, linestyle=':')
    pl.plot(sps.logage, sps.bin_weights(t1, t2)[1:])


spec, phot, x = sps.get_spectrum(**params)
