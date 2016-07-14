from itertools import chain
import numpy as np
from numpy.polynomial.chebyshev import chebval
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

__all__ = ["SSPBasis", "FastSSPBasis", "MultiSSPBasis", "StepSFHBasis", "CompositeSFH"]


# Useful constants
lsun = 3.846e33
pc = 3.085677581467192e18  # in cm
lightspeed = 2.998e18  # AA/s
jansky_mks = 1e-26
# value to go from L_sun/AA to erg/s/cm^2/AA at 10pc
to_cgs = lsun/(4.0 * np.pi * (pc*10)**2)

# change base
loge = np.log10(np.e)


class SSPBasis(object):

    """This is a class that wraps the fsps.StellarPopulation object, which is
    used only for producing SSPs.  The StellarPopulation object is accessed as
    `SSPBasis().ssp`.

    This class allows for the custom calculation of relative SSP weights (by
    overriding ``all_ssp_weights``) to produce spectra from arbitrary composite
    SFHs.  The base implementation here produces an SSP interpolated to the age
    given by `tage`, with initial mass given by ``mass``

    Furthermore, smoothing and filter projections are handled outside of fsps,
    allowing for fast and more flexible algorithms
    """

    def __init__(self, compute_vega_mags=False, zcontinuous=1,
                 interp_type='logarithmic', flux_interp='linear', sfh_type='ssp',
                 mint_log=-3, reserved_params=['sfh', 'tage', 'zred', 'sigma_smooth'],
                 **kwargs):
        """
        :param interp_type: (default: "logarithmic")
            Specify whether to linearly interpolate the SSPs in log(t) or t.
            For the latter, set this to "linear".

        :param flux_interp': (default: "linear")
            Whether to compute the final spectrum as \sum_i w_i f_i or
            e^{\sum_i w_i ln(f_i)}.  Basically you should always do the former,
            which is the default.

        :param mint_log: (default: -3)
            The log of the age (in years) of the youngest SSP.  Note that the
            SSP at this age is assumed to have the same spectrum as the minimum
            age SSP avalibale from fsps.  Typically anything less than 4 or so
            is fine for this parameter, since the integral converges as log(t)
            -> -inf

        :param reserved_params:
            These are parameters which have names like the FSPS parameters but
            will not be passed to the StellarPopulation object because we are
            overriding their functionality using (hopefully more efficient)
            custom algorithms.
        """

        self.interp_type = interp_type
        self.sfh_type = sfh_type
        self.mint_log = mint_log
        self.flux_interp = flux_interp
        self.ssp = fsps.StellarPopulation(compute_vega_mags=compute_vega_mags,
                                          zcontinuous=zcontinuous)
        self.ssp.params['sfh'] = 0
        self.reserved_params = reserved_params
        self.params = {}
        self.update(**kwargs)

    def update(self, **params):
        """Update the parameters, passing through *unreserved* FSPS parameters to
        the fsps.StellarPopulation object.
        """
        for k, v in params.items():
            # try to make parameters scalar
            try:
                if (len(v) == 1) and callable(v[0]):
                    self.params[k] = v[0]
                else:
                    self.params[k] = np.squeeze(v)
            except:
                self.params[k] = v
            # Parameters named like FSPS params but that we reserve for use
            # here.  Do not pass them to FSPS.
            if k in self.reserved_params:
                continue
            # Otherwise if a parameter exists in the FSPS parameter set, pass it in.
            if k in self.ssp.params.all_params:
                self.ssp.params[k] = v

        # We use FSPS for SSPs !!ONLY!!
        assert self.ssp.params['sfh'] == 0

    def get_galaxy_spectrum(self, **params):
        """Update parameters, then multiply SSP weights by SSP spectra and
        stellar masses, and sum.

        :returns wave:
            Wavelength in angstroms.

        :returns spectrum:
            Spectrum in units of Lsun/Hz/solar masses formed.

        :returns mass_fraction:
            Fraction of the formed stellar mass that still exists.
        """
        self.update(**params)

        # Get the SSP spectra and masses (caching the latter), adding an extra
        # mass and spectrum for t=0, using the first SSP spectrum.
        wave, ssp_spectra = self.ssp.get_spectrum(tage=0, peraa=False)
        ssp_spectra = np.vstack([ssp_spectra[0,:], ssp_spectra])
        self.ssp_stellar_masses = np.insert(self.ssp.stellar_mass, 0, 1.0)
        if self.flux_interp == 'logarithmic':
            ssp_spectra = np.log(ssp_spectra)

        # Get weighted sum of spectra, adding the t=0 spectrum using the first SSP.
        weights = self.all_ssp_weights
        spectrum = np.dot(weights, ssp_spectra) / weights.sum()
        if self.flux_interp == 'logarithmic':
            spectrum = np.exp(spectrum)

        # Get the weighted stellar_mass/mformed ratio
        mass_frac = (self.ssp_stellar_masses * weights).sum() / weights.sum()
        return wave, spectrum, mass_frac

    def get_spectrum(self, outwave=None, filters=None, peraa=False, **params):
        """Get a spectrum and SED for the given params.

        :param outwave:
            Desired *vacuum* wavelengths.  Defaults to the values in
            sps.ssp.wavelength.

        :param peraa: (default: False)
            If `True`, return the spectrum in erg/s/cm^2/AA instead of AB
            maggies.

        :returns spec:
            Observed frame spectrum in AB maggies, unless `peraa=True` in which
            case the units are erg/s/cm^2/AA.

        :returns phot:
            Observed frame photometry in AB maggies.

        :returns mass_frac:
            The ratio of the surviving stellar mass to the total mass formed.
        """
        # Spectrum in Lsun/Hz per solar mass formed
        wave, spectrum, mfrac = self.get_galaxy_spectrum(**params)

        # Redshifting + Wavelength solution
        if 'zred' in self.reserved_params:
            # We do it ourselves.
            a = 1 + self.params.get('zred', 0)
            b = 0.0
        else:
            a, b = 1.0, 0.0

        if 'wavecal_coeffs' in self.params:
            x = wave - wave.min()
            x = 2.0 * (x / x.max()) - 1.0
            c = np.insert(self.params['wavecal_coeffs'], 0, 0)
            # assume coeeficients give shifts in km/s
            b = chebval(x, c) / (lightspeed*1e-13)
            
        wa, sa = wave * (a + b), spectrum * a
        if outwave is None:
            outwave = wa

        # Observed frame photometry, as absolute maggies
        if filters is not None:
            mags = getSED(wave, lightspeed/wave**2 * sa * to_cgs, filters)
            phot = np.atleast_1d(10**(-0.4 * mags))
        else:
            phot = 0.0

        # Spectral smoothing.
        do_smooth = (('sigma_smooth' in self.params) and
                     ('sigma_smooth' in self.reserved_params))
        if do_smooth:
            # We do it ourselves.
            smspec = self.smoothspec(wa, sa, self.params['sigma_smooth'],
                                     outwave=outwave, **self.params)
        elif outwave is not wa:
            # Just interpolate
            smspec = np.interp(outwave, wa, sa, left=0, right=0)
        else:
            # no interpolation necessary
            smspec = sa

        # Distance dimming and unit conversion
        if (self.params['zred'] == 0) or ('lumdist' in self.params):
            # Use 10pc for the luminosity distance (or a number
            # provided in the dist key in units of Mpc)
            dfactor = (self.params.get('lumdist', 1e-5) * 1e5)**2
        else:
            lumdist = cosmo.luminosity_distance(self.params['zred']).value
            dfactor = (lumdist * 1e5)**2 / (1 + self.params['zred'])
        if peraa:
            # spectrum will be in erg/s/cm^2/AA
            smspec *= to_cgs / dfactor * lightspeed / outwave**2
        else:
            # Spectrum will be in maggies
            smspec *= to_cgs / dfactor / 1e3 / (3631*jansky_mks)

        # Mass normalization
        mass = np.sum(self.params.get('mass', 1.0))
        if np.all(self.params.get('mass_units', 'mstar') == 'mstar'):
            # Convert from current stellar mass to mass formed
            mass /= mfrac

        return smspec * mass, phot / dfactor * mass, mfrac

    @property
    def all_ssp_weights(self):
        """Weights for a single age population.  This is a slow way to do this!
        """
        if self.interp_type == 'linear':
            sspages = np.insert(10**self.logage, 0, 0)
            tb = self.params['tage'] * 1e9

        elif self.interp_type == 'logarithmic':
            sspages = np.insert(self.logage, 0, self.mint_log)
            tb = np.log10(self.params['tage']) + 9

        ind = np.searchsorted(sspages, tb)  # index of the higher bracketing lookback time
        dt = (sspages[ind] - sspages[ind - 1])
        ww = np.zeros(len(sspages))
        ww[ind - 1] = (sspages[ind] - tb) / dt
        ww[ind] = (tb - sspages[ind-1]) / dt
        return ww

    def smoothspec(self, wave, spec, sigma, outwave=None, **kwargs):
        outspec = smoothspec(wave, spec, sigma, outwave=outwave, **kwargs)
        return outspec

    @property
    def logage(self):
        return self.ssp.ssp_ages

    @property
    def wavelengths(self):
        return self.ssp.wavelengths


class FastSSPBasis(SSPBasis):
    """Let FSPS do the work for a single age.
    """
    def get_galaxy_spectrum(self, **params):
        self.update(**params)
        wave, spec = self.ssp.get_spectrum(tage=float(self.params['tage']), peraa=False)
        return wave, spec, self.ssp.stellar_mass


class MultiSSPBasis(SSPBasis):
    """An array of SSPs with different ages, metallicities, and possibly dust
    attenuations.
    """
    def get_galaxy_spectrum(self):
        raise(NotImplementedError)


class StepSFHBasis(SSPBasis):

    """Subclass of SSPBasis that computes SSP weights for piecewise constant
    SFHs (i.e. a binned SFH).  The parameters for this SFH are:

      * `agebins` - array of shape (nbin, 2) giving the younger and older (in
        lookback time) edges of each bin.  If `interp_type` is `"linear"',
        these are assumed to be in years.  Otherwise they are in log10(years)

      * `mass` - array of shape (nbin,) giving the total surviving stellar mass
        (in solar masses) in each bin, unless the `mass_units` parameter is set
        to something different `"mstar"`, in which case the units are assumed
        to be total stellar mass *formed* in each bin.

    The `agebins` parameter *must not be changed* without also setting
    `self._ages=None`.
    """

    @property
    def all_ssp_weights(self):
        # Cache age bins and relative weights.  This means params['agebins']
        # *must not change* without also setting _ages = None
        if getattr(self, '_ages', None) is None:
            self._ages = self.params['agebins']
            nbin, nssp = len(self._ages), len(self.logage) + 1
            self._bin_weights = np.zeros([nbin, nssp])
            for i, (t1, t2) in enumerate(self._ages):
                # These *should* sum to one (or zero) for each bin
                self._bin_weights[i,:] = self.bin_weights(t1, t2)

        # Now normalize the weights in each bin by the mass parameter, and sum
        # over bins.
        bin_masses = self.params['mass']
        if np.all(self.params.get('mass_units', 'mstar') == 'mstar'):
            # Convert from mstar to mformed for each bin.  We have to do this
            # here as well as in get_spectrum because the *relative*
            # normalization in each bin depends on the units, as well as the
            # overall normalization.
            bin_masses /= self.bin_mass_fraction
        w = (bin_masses[:, None] * self._bin_weights).sum(axis=0)

        return w

    @property
    def bin_mass_fraction(self):
        """Return the ratio m_star(surviving) / m_formed for each bin.
        """
        try:
            mstar = self.ssp_stellar_masses
            w = self._bin_weights
            bin_mfrac = (mstar[None, :] * w).sum(axis=-1) / w.sum(axis=-1)
            return bin_mfrac
        except:
            print('agebin info or ssp masses not chached?')
            return 1.0

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
            sspages = np.insert(self.logage, 0, self.mint_log)
            func = constant_logarithmic
            mass = 10**amax - 10**amin

        assert amin >= sspages[0]
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
        assert np.all(ww >= 0)
        wsum = ww.sum(axis=1)
        # unless truly no SF in the component
        if 0 in wsum:
            wsum[wsum == 0] = 1.0
        ww /= wsum[:, None]
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
        if sf_slope == 0.:
            Tz = tage
        else:
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
            tq = np.log10(np.max([tq, 10**mint_log]))
            tage = np.log10(np.max([tage, 10**mint_log]))
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
            tq = np.log10(np.max([tq, 10**mint_log]))
            t0 = np.log10(np.max([t0, 10**mint_log]))
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
    a = (t - tage - tau) * (logt - logages) - tau * loge
    b = (tage + tau) * loge
    return a * np.exp(tprime) + b * expi(tprime)


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


def burst_linear(ages, t, tburst=None, **extras):
    """Burst.  SFR = \delta(t-t_burst)
    """
    return ages - tburst


def burst_logarithmic(logages, logt, tburst=None, **extras):
    """Burst.  SFR = \delta(t-t_burst)
    """
    return logages - np.log10(tburst)
