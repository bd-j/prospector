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

__all__ = ["SSPBasis", "StepSFHBasis"]


# Useful constants
lsun = 3.846e33
pc = 3.085677581467192e18  # in cm
lightspeed = 2.998e18  # AA/s
jansky_mks = 1e-26
# value to go from L_sun/AA to erg/s/cm^2/AA at 10pc
to_cgs = lsun/(4.0 * np.pi * (pc*10)**2)


class SSPBasis(object):

    def __init__(self, compute_vega_mags=False, zcontinuous=1,
                 interp_type='logarithmic', **kwargs):

        self.interp_type = interp_type
        self.ssp = fsps.StellarPopulation(compute_vega_mags=compute_vega_mags,
                                          zcontinuous=zcontinuous)
        self.ssp.params['sfh'] = 0
        self.reserved_params = []
        self.params = {}
        self.update(**kwargs)

    def update(self, **params):
        for k, v in params.items():
            self.params[k] = v
            if k in self.reserved_params:
                continue
            if k in self.ssp.params.all_params:
                self.ssp.params[k] = v
        assert self.ssp.params['sfh'] == 0

    def get_spectrum(self, outwave=None, filters=None, **params):
        """
        :returns spec:
            Spectrum in erg/s/AA/cm^2

        :returns phot:
            Photometry in maggies
        """
        self.update(**params)
        wave, ssp_spectra = self.ssp.get_spectrum(tage=0, peraa=True)
        masses = self.ssp_weights
        spectrum = (masses[:, None] * ssp_spectra).sum(axis=0)

        # redshifting
        a = 1 + self.params.get('zred', 0)
        wa, sa = vac2air(wave) * a, spectrum / a
        
        # observed frame photometry after converting to f_lambda
        # (but not dividing by 4* !pi *r^2 or converting from Lsun to cgs)
        if filters is not None:
            mags = getSED(wa, sa, filters)
            phot = np.atleast_1d(10**(-0.4 * mags))
        else:
            phot = 0.0

        # smoothing
        if outwave is None:
            outwave = wa
        if 'sigma_smooth' in self.params:
            smspec = self.smoothspec(wa, sa, self.params['sigma_smooth'],
                                     outwave=outwave, **self.params)
        else:
            smspec = np.interp(outwave, wa, sa, left=0, right=0)

        # distance dimming and unit conversion
        dist10 = self.params.get('lumdist', 1e-5)/1e-5  # d in units of 10pc
        conv = lsun / (4 * np.pi * (dist10*pc*10)**2)
        
        # distance dimming
        return smspec * conv , phot * conv, masses.sum()

    @property
    def ssp_weights(self):
        masses = self.params['mass']
        w = mass * self.single_age_weights(self, ages)

    def weights(self, ages):
        """This is broken
        """
        ind = np.searchsorted(self.logage, ages)
        right =  (ages - self.logage[ind]) / dt[ind]
        w[ind] += right
        w[ind+1] += 1 - right

    def smoothspec(self, wave, spec, sigma, outwave=None, **kwargs):
        outspec = smoothspec(wave, spec, sigma, outwave=outwave, **kwargs)
        return outspec

    @property
    def logage(self):
        return self.ssp.ssp_ages

    @property
    def wavelengths(self):
        return self.ssp.wavelengths

class StepSFHBasis(SSPBasis):

    @property
    def ssp_weights(self):
        ages = self.params['agebins']
        masses = self.params['mass']
        w = np.zeros(len(self.logage))
        # Should cache the bin weights when agebins not changing.  But this is
        # not very time consuming for few enough bins.
        for (t1, t2), mass in zip(ages, masses):
            w += mass * self.bin_weights(t1, t2)[1:]
        return w

    def bin_weights(self, amin, amax):
        """Compute normalizations required to get a piecewise constant SFH
        within an age bin.  This is super complicated and obscured.  The output
        weights are such that one solar mass will have formed during the bin
        (i.e. SFR = 1/(amax-amin))

        This computes weights using \int_tmin^tmax dt (\log t_i - \log t) / (\log
        t_{i+1} - \log t_i) but see sfh.tex for the detailed calculation and
        the linear time interpolation case.
        """
        if self.interp_type == 'linear':
            sspages = np.insert(10**self.logage, 0, 0)
            func = self._linear
            mass = amax - amin
        elif self.interp_type == 'logarithmic':
            sspages = np.insert(self.logage, 0, 0)
            func = self._logarithmic
            mass = 10**amax - 10**amin

        assert amin >= sspages[1]
        assert amax <= sspages.max()

        # below could be done by using two separate dt vectors instead of two age vectors
        ages = np.array([sspages[:-1], sspages[1:]])
        dt = np.diff(ages, axis=0)
        tmin, tmax = np.clip(ages, amin, amax)

        # get contributions from SSP sub-bin to the left and from SSP sub-bin
        # to the right
        left, right = (func(ages, tmax) - func(ages, tmin)) / dt
        # put into full array
        ww = np.zeros(len(sspages))
        ww[:-1] += right # last element has no sub-bin to the right
        ww[1:] += -left # first element has no subbin to the left.  also, need to flip sign

        # normalize to 1 solar mass formed and return
        return ww / mass

    def _linear(self, ages, t):
        """Linear interpolation of linear times, constant SFR.
        """
        return ages * t - t**2 / 2

    def _logarithmic(self, logages, logt):
        """Linear interpolation of logarithmic times, constant SFR.
        """
        t = 10**logt
        return logages * t - t * (logt - np.log10(np.e))
