from itertools import chain
import numpy as np
from ..utils.smoothing import smoothspec
from sedpy.observate import getSED, vac2air, air2vac

try:
    import fsps
except(ImportError):
    pass
try:
    from astropy.cosmology import WMAP9 as cosmo
except(ImportError):
    pass

__all__ = ["SSPBasis"]


class SSPBasis(object):

    def __init__(self, compute_vega_mags=False, zcontinuous=1,
                 interp_type='logarithmic', **kwargs):

        self.interp_type = interp_type
        self.ssp = fsps.StellarPopulation(compute_vega_mags=compute_vega_mags,
                                          zcontinuous=zcontinuous)
        self.params = {}

    def get_spectrum(self, outwave=None, filters=None, **params):
        self.update(**params)
        w, ssp_spectra = self.ssp.get_spectrum(tage=0, peraa=True)
        spectrum = self.ssp_weights[:,None] * self.ssp_spectra).sum(axis=0)

        #distance dimming

        #photometry

        #smoothing
        
    def update(self, **params):
        for k, v in params.items():
            self.params[k] = v
            if k in self.ssp.params.all_params:
                self.ssp.params[k] = v

    def ssp_weights(timebins, masses):
        for (t1, t2), mass in zip(timebins, masses):
            w += mass * self.bin_weights(t1, t2)[1:]
        return w

    def bin_weights(amin, amax):
        """Compute normalizations required to get a piecewise constant
        SFH (SFR=1) within an age bin.  This is super complicated and obscured.
        """
        if self.interp_type == 'linear':
            sspages = np.insert(10**self.logage, 0, 0)
            func = self._linear
            mass = amax - amin
        elif self.interp_type == 'logarithmic':
            sspages = np.insert(self.logage, 0, 0)
            func = self._logarithmic
            mass = 10**amax - 10**amin

        assert amin > 0
        assert amax < sspages.max()

        # below could be done by using two separate dt vectors instead of two age vectors
        ages = np.array([sspages[:-1], sspages[1:]])
        dt = np.diff(ages, axis=0)
        tmin, tmax = np.clip(ages, amin, amax)

        # get contributions from SSP sub-bin to the left and from SSP sub-bin
        # to the right, using linear interpolation in linear time
        left, right = (func(ages, tmax) - func(ages, tmin)) / dt
        # put into full array
        ww = np.zeros(len(sspages))
        ww[:-1] += right # last element has no sub-bin to the right
        ww[1:] += -left # first element has no subbin to the left.  also, need to flip sign

        # normalize to 1 solar mass formed and return
        return ww / mass        

    def _linear(self, ages, t):
        """Linear interpolation of linear times
        """
        return ages * t - t**2 / 2

    def _logarithmic(self, logages, logt):
        """Linear interpolation of logarithmic times
        """
        t = 10**logt
        return logages * t - t * (logt - np.log10(np.e))

    @property
    def logage(self):
        return self.ssp.ssp_ages

