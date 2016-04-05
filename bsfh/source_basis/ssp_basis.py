from itertools import chain
import numpy as np
from scipy.special import expi, gammainc
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

__all__ = ["SSPBasis", "StepSFHBasis", "CompositeSFH"]


# Useful constants
lsun = 3.846e33
pc = 3.085677581467192e18  # in cm
lightspeed = 2.998e18  # AA/s
jansky_mks = 1e-26
# value to go from L_sun/AA to erg/s/cm^2/AA at 10pc
to_cgs = lsun/(4.0 * np.pi * (pc*10)**2)

# change base
loge = np.log10(np.e)

# minimum time for logarithmic sfhs
mint_log = 0  # log(yr)


class SSPBasis(object):

    def __init__(self, compute_vega_mags=False, zcontinuous=1,
                 interp_type='logarithmic', sfh_type='ssp',
                 flux_interp='linear', mint_log=-3, **kwargs):

        self.interp_type = interp_type
        self.sfh_type = sfh_type
        self.mint_log = mint_log
        self.flux_interp = flux_interp
        self.ssp = fsps.StellarPopulation(compute_vega_mags=compute_vega_mags,
                                          zcontinuous=zcontinuous)
        self.ssp.params['sfh'] = 0
        self.reserved_params = ['sfh']
        self.params = {}
        self.update(**kwargs)

    def update(self, **params):
        for k, v in params.items():
            self.params[k] = v
            if k in self.reserved_params:
                continue
            # If a parameter exists in the FSPS parameter set, pass it in.
            if k in self.ssp.params.all_params:
                self.ssp.params[k] = v

        # We use FSPS for SSPs !!ONLY!!
        assert self.ssp.params['sfh'] == 0

    def get_galaxy_spectrum(self, **params):
        self.update(**params)
        wave, ssp_spectra = self.ssp.get_spectrum(tage=0, peraa=True)
        if self.flux_interp == 'logarithmic':
            ssp_spectra = np.log(ssp_spectra)
        masses = self.all_ssp_weights
        spectrum = (masses[1:, None] * ssp_spectra).sum(axis=0)
        # Add the t_0 spectrum, using the t_1 spectrum
        spectrum += masses[0] * ssp_spectra[0, :]
        if self.flux_interp == 'logarithmic':
            spectrum = np.exp(spectrum)
        return wave, spectrum

    def get_spectrum(self, outwave=None, filters=None, **params):
        """
        :returns spec:
            Spectrum in erg/s/AA/cm^2

        :returns phot:
            Photometry in maggies

        :returns x:
            A generic blob. can be used to return e.g. present day masses,
            total masses, etc.
        """
        wave, spectrum = self.get_galaxy_spectrum(**params)
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
        return smspec * conv, phot * conv, None

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
    def all_ssp_weights(self):
        ages = self.params['agebins']
        masses = self.params['mass']
        w = np.zeros(len(self.logage))
        # Loop over age bins
        # Should cache the bin weights when agebins not changing.  But this is
        # not very time consuming for few enough bins.
        for (t1, t2), mass in zip(ages, masses):
            params = {'tage': t2, 'sf_trunc': t2 - t1}
            #w += mass * self.ssp_weights(self.func, regular_limits, params)
            w += mass * self.bin_weights(t1, t2)[1:]
        return w

    def bin_weights(self, amin, amax):
        """Compute normalizations required to get a piecewise constant SFH
        within an age bin.  This is super complicated and obscured.  The output
        weights are such that one solar mass will have formed during the bin
        (i.e. SFR = 1/(amax-amin))

        This computes weights using \int_tmin^tmax dt (\log t_i - \log t) /
        (\log t_{i+1} - \log t_i) but see sfh.tex for the detailed calculation
        and the linear time interpolation case.
        """
        if self.interp_type == 'linear':
            sspages = np.insert(10**self.logage, 0, 0)
            func = constant_linear
            mass = amax - amin
        elif self.interp_type == 'logarithmic':
            sspages = np.insert(self.logage, 0, 0)
            func = constant_logarithmic
            mass = 10**amax - 10**amin

        assert amin >= sspages[1]
        assert amax <= sspages.max()

        # below could be done by using two separate dt vectors instead of two
        # age vectors
        ages = np.array([sspages[:-1], sspages[1:]])
        dt = np.diff(ages, axis=0)
        tmin, tmax = np.clip(ages, amin, amax)

        # get contributions from SSP sub-bin to the left and from SSP sub-bin
        # to the right
        left, right = (func(ages, tmax) - func(ages, tmin)) / dt
        # put into full array
        ww = np.zeros(len(sspages))
        ww[:-1] += right  # last element has no sub-bin to the right
        ww[1:] += -left  # need to flip sign

        # normalize to 1 solar mass formed and return
        return ww / mass


class CompositeSFH(SSPBasis):

    def configure(self):
        """This reproduces FSPS-like combinations of SFHs.  Note that the
        *same* parameter set is passed to each component in the combination
        """
        sfhs = [self.sfh_type]
        limits = len(sfhs) * ['regular']
        if 'simha' in self.sfh_type:
            sfhs = ['delaytau', 'linear']
            limits = ['regular', 'simha']

        fnames = ['{0}_{1}'.format(f, self.interp_type) for f in sfhs]
        lnames = ['{}_limits'.format(f) for f in limits]
        self.funcs = [globals()[f] for f in fnames]
        self.limits = [globals()[f] for f in lnames]

        if self.interp_type == 'linear':
            sspages = np.insert(10**self.logage, 0, 0)
        elif self.interp_type == 'logarithmic':
            sspages = np.insert(self.logage, 0, self.mint_log)
        self.ages = np.array([sspages[:-1], sspages[1:]])
        self.dt = np.diff(self.ages, axis=0)

    @property
    def _limits(self):
        pass

    @property
    def _funcs(self):
        pass

    @property
    def all_ssp_weights(self):

        # Full output weight array.  We keep separate vectors for each
        # component so we can renormalize after the loop, but for many
        # components it would be better to renormalize and sum within the loop
        ww = np.zeros([len(self.funcs), self.ages.shape[-1] + 1])

        # Loop over components.  Note we are sending the same params to every component
        for i, (limit, func) in enumerate(zip(self.limits, self.funcs)):
            ww[i, :] = self.ssp_weights(func, limit, self.params)

        # renormalize each component to 1 Msun
        ww /= ww.sum(axis=1)[:, None]
        # apply relative normalizations
        ww *= self.normalizations(**self.params)[:, None]
        # And finally add all components together and renormalize again to
        # 1Msun and return
        return ww.sum(axis=0) / ww.sum()

    def ssp_weights(self, integral, limit_function, params, **extras):
        # build full output weight vector
        ww = np.zeros(self.ages.shape[-1] + 1)
        tmin, tmax = limit_function(self.ages, mint_log=self.mint_log,
                                    interp_type=self.interp_type, **params)
        left, right = (integral(self.ages, tmax, **params) -
                       integral(self.ages, tmin, **params)) / self.dt
        # Put into full array, shifting the `right` terms by 1 element
        ww[:-1] += right  # last SSP has no sub-bin to the right
        ww[1:] += -left   # need to flip sign

        # Note that now ww[i,1] = right[1] - left[0], where
        # left[0] is the integral from tmin,0 to tmax,0 of
        # SFR(t) * (sspages[0] - t)/(sspages[1] - sspages[0]) and
        # right[1] is the integral from tmin,1 to tmax,1 of
        # SFR(t) * (sspages[2] - t)/(sspages[2] - sspages[1])
        return ww

    def normalizations(self, tage=0., sf_trunc=0, sf_slope=0, const=0,
                       fburst=0, tau=0., **extras):
        if (sf_trunc <= 0) or (sf_trunc > tage):
            Tmax = tage
        else:
            Tmax = sf_trunc
        # Tau models.  SFH=1 -> power=1; SFH=4,5 -> power=2
        if ('delay' in self.sfh_type) or ('simha' in self.sfh_type):
            power = 2.
        else:
            power = 1.
        mass_tau = tau * gammainc(power, Tmax/tau)

        if 'simha' not in self.sfh_type:
            return np.array([mass_tau])
        # SFR at Tmax
        sfr_q = (Tmax/tau)**(power-1) * np.exp(-Tmax/tau)

        # linear.  integral of (1 - m * (T - Tmax)) from Tmax to Tzero
        Tz = Tmax + 1/np.float64(sf_slope)
        if (Tz < Tmax) or (Tz > tage) or (not np.isfinite(Tz)):
            Tz = tage
        m = sf_slope
        mass_linear = (Tz - Tmax) - m/2.*(Tz**2 + Tmax**2) + m*Tz*Tmax

        # normalize the linear portion relative to the tau portion
        norms = np.array([1, mass_linear * sfr_q / mass_tau])
        norms /= norms.sum()
        # now add in constant and burst
        if (const > 0) or (fburst > 0):
            norms = (1-fburst-const) * norms
            norms.tolist().extend([const, fburst])
        return np.array(norms)


def regular_limits(ages, tage=0., sf_trunc=0., mint_log=-3,
                   interp_type='logarithmic', **extras):
        # get the truncation time in units of lookback time
        if (sf_trunc <= 0) or (sf_trunc > tage):
            tq = 0
        else:
            tq = tage - sf_trunc
        if interp_type == 'logarithmic':
            tq = np.max([np.log10(tq), mint_log])
            tage = np.max([np.log10(tage), mint_log])
        return np.clip(ages, tq, tage)


def simha_limits(ages, tage=0., sf_trunc=0, sf_slope=0., mint_log=-3,
                 interp_type='logarithmic', **extras):
        # get the truncation time in units of lookback time
        if (sf_trunc <= 0) or (sf_trunc > tage):
            tq = 0
        else:
            tq = tage - sf_trunc
        t0 = tq - 1. / np.float64(sf_slope)
        if (t0 > tq) or (t0 <= 0) or (not np.isfinite(t0)):
            t0 = 0.
        if interp_type == 'logarithmic':
            tq = np.max([np.log10(tq), mint_log])
            t0 = np.max([np.log10(t0), mint_log])
        return np.clip(ages, t0, tq)


def constant_linear(ages, t, **extras):
    """Indefinite integral for SFR = 1

    :param ages:
        Linear age(s) of the SSPs.

    :param t:
        Linear time at which to evaluate the indefinite integral
    """
    return ages * t - t**2 / 2


def constant_logarithmic(logages, logt, **extras):
    """SFR = 1
    """
    t = 10**logt
    return t * (logages - logt + loge)


def tau_linear(ages, t, tau=None, **extras):
    """SFR = e^{(tage-t)/\tau}
    """
    return (ages - t + tau) * np.exp(t / tau)


def tau_logarithmic(logages, logt, tau=None, **extras):
    """SFR = e^{(tage-t)/\tau}
    """
    tprime = 10**logt / tau
    return (logages - logt) * np.exp(tprime) + loge * expi(tprime)


def delaytau_linear(ages, t, tau=None, tage=None, **extras):
    """SFR = (tage-t) * e^{(tage-t)/\tau}
    """
    bracket = tage * ages - (tage + ages)*(t - tau) + t**2 - 2*t*tau + 2*tau**2
    return bracket * np.exp(t / tau)


def delaytau_logarithmic(logages, logt, tau=None, tage=None, **extras):
    """SFR = (tage-t) * e^{(tage-t)/\tau}
    """
    t = 10**logt
    tprime = t / tau
    bracket = t * (logt - logages) + tage*logages + tau * (logages - loge)
    term = bracket * np.exp(tprime)
    h = tau * (logt * np.exp(tprime) - loge * expi(tprime))
    return term - (tage / tau + 1) * h


def linear_linear(ages, t, tage=None, sf_trunc=0, sf_slope=0., **extras):
    """SFR = [1 - sf_slope * (tage-t)]
    """
    tq = np.max([0, tage-sf_trunc])
    k = 1 - sf_slope * tq
    return k * ages * t + (sf_slope*ages - k) * t**2 / 2 - sf_slope * t**3 / 3


def linear_logarithmic(logages, logt, tage=None, sf_trunc=0, sf_slope=0., **extras):
    """SFR = [1 - sf_slope * (tage-t)]
    """
    tq = np.max([0, tage-sf_trunc])
    t = 10**logt
    k = 1 - sf_slope * tq
    term1 = k * t * (logages - logt + loge)
    term2 = sf_slope * t**2 / 2 * (logages - logt + loge / 2)
    return term1 + term2


def burst_linear(ages, t, t_burst=None):
    """Burst.  SFR = \delta(t-t_burst)
    """
    # return ages - t_burst
    raise(NotImplementedError)


def burst_logarithmic(logages, logt, t_burst=None):
    """Burst.  SFR = \delta(t-t_burst)
    """
    # return logages - np.log10(t_burst)
    raise(NotImplementedError)
