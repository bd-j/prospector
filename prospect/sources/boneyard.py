import numpy as np
from scipy.special import expi, gammainc

from .ssp_basis import SSPBasis


__all__ = ["StepSFHBasis", "CompositeSFH"]

# change base
loge = np.log10(np.e)


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
                self._bin_weights[i, :] = self.bin_weights(t1, t2)

        # Now normalize the weights in each bin by the mass parameter, and sum
        # over bins.
        bin_masses = self.params['mass']
        if np.all(self.params.get('mass_units', 'mformed') == 'mstar'):
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
        except(AttributeError):
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
    """Subclass of SSPBasis that computes SSP weights for a parameterized SF.
    The parameters for this SFH are:

      * `sfh_type` - String of "delaytau", "tau", "simha"

      * `tage`, `sf_trunc`,  `sf_slope`, `const`, `fburst`, `tau`

      * `mass` -

    """

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
