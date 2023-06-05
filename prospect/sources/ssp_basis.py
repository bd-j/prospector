from copy import deepcopy
import numpy as np
from numpy.polynomial.chebyshev import chebval

from ..utils.smoothing import smoothspec
from .constants import cosmo, lightspeed, jansky_cgs, to_cgs_at_10pc
from .attenuation import extinction

try:
    import fsps
    from sedpy.observate import getSED
except(ImportError, RuntimeError):
    pass

__all__ = ["SSPBasis", "FastSSPBasis", "FastStepBasis",
           "MultiSSPBasis"]


to_cgs = to_cgs_at_10pc


class SSPBasis(object):

    """This is a class that wraps the fsps.StellarPopulation object, which is
    used for producing SSPs.  The ``fsps.StellarPopulation`` object is accessed
    as ``SSPBasis().ssp``.

    This class allows for the custom calculation of relative SSP weights (by
    overriding ``all_ssp_weights``) to produce spectra from arbitrary composite
    SFHs. Alternatively, the entire ``get_galaxy_spectrum`` method can be
    overridden to produce a galaxy spectrum in some other way, for example
    taking advantage of weight calculations within FSPS for tabular SFHs or for
    parameteric SFHs.

    The base implementation here produces an SSP interpolated to the age given
    by ``tage``, with initial mass given by ``mass``.  However, this is much
    slower than letting FSPS calculate the weights, as implemented in
    :py:class:`FastSSPBasis`.

    Furthermore, smoothing, redshifting, and filter projections are handled
    outside of FSPS, allowing for fast and more flexible algorithms.

    :param reserved_params:
        These are parameters which have names like the FSPS parameters but will
        not be passed to the StellarPopulation object because we are overriding
        their functionality using (hopefully more efficient) custom algorithms.
    """

    def __init__(self, zcontinuous=1, reserved_params=['tage', 'sigma_smooth'],
                 interp_type='logarithmic', flux_interp='linear',
                 mint_log=-3, compute_vega_mags=False,
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
        self.mint_log = mint_log
        self.flux_interp = flux_interp
        self.ssp = fsps.StellarPopulation(compute_vega_mags=compute_vega_mags,
                                          zcontinuous=zcontinuous)
        self.ssp.params['sfh'] = 0
        self.reserved_params = reserved_params
        self.params = {}
        self.update(**kwargs)

    def update(self, **params):
        """Update the parameters, passing the *unreserved* FSPS parameters
        through to the ``fsps.StellarPopulation`` object.

        :param params:
            A parameter dictionary.
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
            # Otherwise if a parameter exists in the FSPS parameter set, pass a
            # copy of it in.
            if k in self.ssp.params.all_params:
                self.ssp.params[k] = deepcopy(v)

        # We use FSPS for SSPs !!ONLY!!
        # except for FastStepBasis.  And CSPSpecBasis. and...
        # assert self.ssp.params['sfh'] == 0

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
        ssp_spectra = np.vstack([ssp_spectra[0, :], ssp_spectra])
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

    def get_galaxy_elines(self):
        """Get the wavelengths and specific emission line luminosity of the nebular emission lines
        predicted by FSPS. These lines are in units of Lsun/solar mass formed.
        This assumes that `get_galaxy_spectrum` has already been called.

        :returns ewave:
            The *restframe* wavelengths of the emission lines, AA

        :returns elum:
            Specific luminosities of the nebular emission lines,
            Lsun/stellar mass formed
        """
        ewave = self.ssp.emline_wavelengths
        # This allows subclasses to set their own specific emission line
        # luminosities within other methods, e.g., get_galaxy_spectrum, by
        # populating the `_specific_line_luminosity` attribute.
        elum = getattr(self, "_line_specific_luminosity", None)

        if elum is None:
            elum = self.ssp.emline_luminosity.copy()
            if elum.ndim > 1:
                elum = elum[0]
            if self.ssp.params["sfh"] == 3:
                # tabular sfh
                mass = np.sum(self.params.get('mass', 1.0))
                elum /= mass

        return ewave, elum

    def get_spectrum(self, outwave=None, filters=None, peraa=False, **params):
        """Get a spectrum and SED for the given params.

        :param outwave: (default: None)
            Desired *vacuum* wavelengths.  Defaults to the values in
            ``sps.wavelength``.

        :param peraa: (default: False)
            If `True`, return the spectrum in erg/s/cm^2/AA instead of AB
            maggies.

        :param filters: (default: None)
            A list of filter objects for which you'd like photometry to be calculated.

        :param params:
            Optional keywords giving parameter values that will be used to
            generate the predicted spectrum.

        :returns spec:
            Observed frame spectrum in AB maggies, unless ``peraa=True`` in which
            case the units are erg/s/cm^2/AA.

        :returns phot:
            Observed frame photometry in AB maggies.

        :returns mass_frac:
            The ratio of the surviving stellar mass to the total mass formed.
        """
        # Spectrum in Lsun/Hz per solar mass formed, restframe
        wave, spectrum, mfrac = self.get_galaxy_spectrum(**params)

        # Redshifting + Wavelength solution
        # We do it ourselves.
        a = 1 + self.params.get('zred', 0)
        af = a
        b = 0.0

        if 'wavecal_coeffs' in self.params:
            x = wave - wave.min()
            x = 2.0 * (x / x.max()) - 1.0
            c = np.insert(self.params['wavecal_coeffs'], 0, 0)
            # assume coeeficients give shifts in km/s
            b = chebval(x, c) / (lightspeed*1e-13)

        wa, sa = wave * (a + b), spectrum * af  # Observed Frame
        if outwave is None:
            outwave = wa

        # Observed frame photometry, as absolute maggies
        if filters is not None:
            flambda = lightspeed/wa**2 * sa * to_cgs
            phot = 10**(-0.4 * np.atleast_1d(getSED(wa, flambda, filters)))
            # TODO: below is faster for sedpy > 0.2.0
            #phot = np.atleast_1d(getSED(wa, lightspeed/wa**2 * sa * to_cgs,
            #                            filters, linear_flux=True))
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
        zred = self.params.get('zred', 0.0)
        if (zred == 0) or ('lumdist' in self.params):
            # Use 10pc for the luminosity distance (or a number
            # provided in the dist key in units of Mpc)
            dfactor = (self.params.get('lumdist', 1e-5) * 1e5)**2
        else:
            lumdist = cosmo.luminosity_distance(zred).value
            dfactor = (lumdist * 1e5)**2
        if peraa:
            # spectrum will be in erg/s/cm^2/AA
            smspec *= to_cgs / dfactor * lightspeed / outwave**2
        else:
            # Spectrum will be in maggies
            smspec *= to_cgs / dfactor / (3631*jansky_cgs)

        # Convert from absolute maggies to apparent maggies
        phot /= dfactor

        # Mass normalization
        mass = np.sum(self.params.get('mass', 1.0))
        if np.all(self.params.get('mass_units', 'mformed') == 'mstar'):
            # Convert input normalization units from current stellar mass to mass formed
            mass /= mfrac

        return smspec * mass, phot * mass, mfrac

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
        return self.ssp.ssp_ages.copy()

    @property
    def wavelengths(self):
        return self.ssp.wavelengths.copy()


class FastSSPBasis(SSPBasis):
    """A subclass of :py:class:`SSPBasis` that is a faster way to do SSP models by letting
    FSPS do the weight calculations.
    """

    def get_galaxy_spectrum(self, **params):
        self.update(**params)
        wave, spec = self.ssp.get_spectrum(tage=float(self.params['tage']), peraa=False)
        return wave, spec, self.ssp.stellar_mass


class FastStepBasis(SSPBasis):
    """Subclass of :py:class:`SSPBasis` that implements a "nonparameteric"
    (i.e. binned) SFH.  This is accomplished by generating a tabular SFH with
    the proper form to be passed to FSPS. The key parameters for this SFH are:

      * ``agebins`` - array of shape ``(nbin, 2)`` giving the younger and older
        (in lookback time) edges of each bin in log10(years)

      * ``mass`` - array of shape ``(nbin,)`` giving the total stellar mass
        (in solar masses) **formed** in each bin.
    """

    def get_galaxy_spectrum(self, **params):
        """Construct the tabular SFH and feed it to the ``ssp``.
        """
        self.update(**params)
        # --- check to make sure agebins have minimum spacing of 1million yrs ---
        #       (this can happen in flex models and will crash FSPS)
        if np.min(np.diff(10**self.params['agebins'])) < 1e6:
            raise ValueError

        mtot = self.params['mass'].sum()
        time, sfr, tmax = self.convert_sfh(self.params['agebins'], self.params['mass'])
        self.ssp.params["sfh"] = 3  # Hack to avoid rewriting the superclass
        self.ssp.set_tabular_sfh(time, sfr)
        wave, spec = self.ssp.get_spectrum(tage=tmax, peraa=False)
        return wave, spec / mtot, self.ssp.stellar_mass / mtot

    def convert_sfh(self, agebins, mformed, epsilon=1e-4, maxage=None):
        """Given arrays of agebins and formed masses with each bin, calculate a
        tabular SFH.  The resulting time vector has time points either side of
        each bin edge with a "closeness" defined by a parameter epsilon.

        :param agebins:
            An array of bin edges, log(yrs).  This method assumes that the
            upper edge of one bin is the same as the lower edge of another bin.
            ndarray of shape ``(nbin, 2)``

        :param mformed:
            The stellar mass formed in each bin.  ndarray of shape ``(nbin,)``

        :param epsilon: (optional, default 1e-4)
            A small number used to define the fraction time separation of
            adjacent points at the bin edges.

        :param maxage: (optional, default: ``None``)
            A maximum age of stars in the population, in yrs.  If ``None`` then the maximum
            value of ``agebins`` is used.  Note that an error will occur if maxage
            < the maximum age in agebins.

        :returns time:
            The output time array for use with sfh=3, in Gyr.  ndarray of shape (2*N)

        :returns sfr:
            The output sfr array for use with sfh=3, in M_sun/yr.  ndarray of shape (2*N)

        :returns maxage:
            The maximum valid age in the returned isochrone.
        """
        #### create time vector
        agebins_yrs = 10**agebins.T
        dt = agebins_yrs[1, :] - agebins_yrs[0, :]
        bin_edges = np.unique(agebins_yrs)
        if maxage is None:
            maxage = agebins_yrs.max()  # can replace maxage with something else, e.g. tuniv
        t = np.concatenate((bin_edges * (1.-epsilon), bin_edges * (1+epsilon)))
        t.sort()
        t = t[1:-1] # remove older than oldest bin, younger than youngest bin
        fsps_time = maxage - t

        #### calculate SFR at each t
        sfr = mformed / dt
        sfrout = np.zeros_like(t)
        sfrout[::2] = sfr
        sfrout[1::2] = sfr  # * (1+epsilon)

        return (fsps_time / 1e9)[::-1], sfrout[::-1], maxage / 1e9

try:
    from cue.continuum import predict as cont_predict
    from cue.line import predict as line_predict
    from cue.constants import *
    from cue.utils import fit_4loglinear, get_4loglinear_spectra, Ltotal, logQ
except:
    pass

def calcQ(lamin0, specin0, mstar=1.0, helium=False, f_nu=True):
    '''
    Claculate the number of lyman ionizing photons for given spectrum
    Input spectrum must be in ergs/s/A!!
    Q = int(Lnu/hnu dnu, nu_0, inf)
    '''
    from scipy.integrate import simps
    lamin = np.asarray(lamin0)
    specin = np.asarray(specin0)
    c = 2.9979e18 #ang/s
    h = 6.626e-27 #erg/s
    if helium:
        lam_0 = 304.0
    else:
        lam_0 = 911.6
    if f_nu:
        nu_0 = c/lam_0
        inds, = np.where(c/lamin >= nu_0)
        hlam, hflu = c/lamin[inds], specin[inds]
        nu = hlam[::-1]
        f_nu = hflu[::-1]
        integrand = f_nu/(h*nu)
        Q = simps(integrand, x=nu)
    else:
        inds, = np.nonzero(lamin <= lam_0)
        lam = lamin[inds]
        spec = specin[inds]
        integrand = lam*spec/(h*c)
        Q = simps(integrand, x=lam)*mstar
    return Q


class NebularLineBasis(FastStepBasis):
    """A subclass of :py:class:`FastStepBasis` that uses 5 nebular parameters and the csp to predict line fluxes.
    """
    
    def get_galaxy_spectrum(self, csp_wav=None, csp_spec=None, **params):
        """Add nebular continum to the ``ssp``. Also construct the tabular SFH and feed it to the ``ssp``.
        :param use_stellar_ionizing_spectrum:
            If true, fit the csp and to get the ionizing spectrum parameters, else read from the model
        :param csp_wav:
            CSP wavelengths, AA
        :param csp_spec:
            CSP fluxes, Lsun/Hz
        :param ionspec_index1, ionspec_index2, ionspec_index3, ionspec_index4, ionspec_logLratio1, ionspec_logLratio2, ionspec_logLratio3:
            ionizing parameters, follow the range 
            ionspec_index1: [1, 42], ionspec_index2: [-0.3, 30],
            ionspec_index3: [-1, 14], ionspec_index4: [-1.7, 8],
            logLratio1: [-1., 10.1], logLratio2: [-0.5, 1.9], logLratio3: [-0.4, 2.2]
        :param gas_logu, gas_logn, gas_logz, gas_logno, gas_logco:
            nebular parameters, follow the range
            gas_logu: [-4, -1], gas_logn: [1, 4], gas_logz: [-2.2, 0.5],
            gas_logno: [-1, log10(5.4)], gas_logco: [-1, log10(5.4)]
        :returns wave:
            The restframe wavelengths of the emission lines, AA
        :returns spec:
            Specific luminosities of the continuum,
            Lsun/stellar mass formed
        :returns mfrac:
            Stellar mass over total mass formed
        """
        self.update(**params)
        # --- check to make sure agebins have minimum spacing of 1million yrs ---
        #       (this can happen in flex models and will crash FSPS)
        if np.min(np.diff(10**self.params['agebins'])) < 1e6:
            raise ValueError
            
        # set SFH
        mtot = self.params['mass'].sum()
        time, sfr, tmax = self.convert_sfh(self.params['agebins'], self.params['mass'])
        self.ssp.params["sfh"] = 3  # Hack to avoid rewriting the superclass
        self.ssp.set_tabular_sfh(time, sfr)

        # use the emulator to predict nebular continuum in units of Lnu
        if self.params["use_stellar_ionizing_spectrum"] == True:
            assert bool(self.ssp.params['add_neb_emission']) is False, "params['add_neb_emission'] is True, set to False to fit the raw ionizing spectrum with no nebular emission"
            assert bool(self.ssp.params['add_neb_continuum']) is False, "params['add_neb_continuum'] is True, set to False to fit the raw ionizing spectrum with no nebular emission"
            if csp_spec == None: 
                # turn off dust and IGM absortion to get the ionizing spectrum
                dust1, dust2 = self.ssp.params['dust1'].copy(), self.ssp.params['dust2'].copy()
                igm_flag = self.ssp.params['add_igm_absorption']
                self.update(dust1=0, dust2=0, add_igm_absorption=False)
                csp_wav, csp_spec = self.ssp.get_spectrum(tage=tmax, peraa=False)
                self.update(dust1=dust1, dust2=dust2, add_igm_absorption=igm_flag)
            popt = fit_4loglinear(csp_wav, csp_spec)
            self._ionparam = popt # cache the parameters of the powerlaw fit to the ionizing spectrum
            self._ionQ = calcQ(csp_wav, csp_spec*3.839E33)
            logLratios = np.diff(np.squeeze(Ltotal(param=popt.reshape(1,4,2))))
            self._logLratios = logLratios
            neb_cont = cont_predict(gammas=popt[:,0],
                                    log_L_ratios=logLratios,
                                    log_QH=logQ(self.ssp.params['gas_logu'], lognH=self.params['gas_logn']),
                                    n_H=10**self.params['gas_logn'],
                                    log_OH_ratio=self.ssp.params['gas_logz'],
                                    log_NO_ratio=self.params['gas_logno'],
                                    log_CO_ratio=self.params['gas_logco'],
                                   ).nn_predict()
            cont_spec = neb_cont[1]/3.839E33/10**logQ(self.ssp.params['gas_logu'], lognH=self.params['gas_logn'])*self._ionQ # convert to the unit in FSPS
            self.update(ionspec_index1=popt[0,0], ionspec_index2=popt[1,0], ionspec_index3=popt[2,0], ionspec_index4=popt[3,0],
                        ionspec_logLratio1=logLratios[0], ionspec_logLratio2=logLratios[1], ionspec_logLratio3=logLratios[2])
        else:
            neb_cont = cont_predict(gammas=[self.params['ionspec_index1'], self.params['ionspec_index2'], 
                                            self.params['ionspec_index3'], self.params['ionspec_index4']],
                                    log_L_ratios=[self.params['ionspec_logLratio1'], self.params['ionspec_logLratio2'],
                                                  self.params['ionspec_logLratio3']],
                                    log_QH=logQ(self.ssp.params['gas_logu'], lognH=self.params['gas_logn']),
                                    n_H=10**self.params['gas_logn'],
                                    log_OH_ratio=self.ssp.params['gas_logz'],
                                    log_NO_ratio=self.params['gas_logno'],
                                    log_CO_ratio=self.params['gas_logco'],
                                   ).nn_predict()
            cont_spec = neb_cont[1]/3.839E33 # divided by Lsun coded in FSPS
            
        from scipy.interpolate import CubicSpline
        neb_cont_cs = CubicSpline(neb_cont[0], cont_spec, extrapolate=True) # interpolate onto the fsps wavelengths

        # calculate stellar continuum; add nebular continuum and attenuate it
        wave, spec = self.ssp.get_spectrum(tage=tmax, peraa=False)
        neb_spec_no_dust = neb_cont_cs(wave)+1e-95 #set a floor of 1E-95 following fsps
        spec += neb_spec_no_dust*extinction(wave,dtype=self.ssp.params['dust_type'],
                                             dust_index=self.ssp.params['dust_index'],dust2=self.ssp.params['dust2'],
                                             dust1_index=self.ssp.params['dust1_index'],dust1=self.ssp.params['dust1'])
        spec = spec / mtot
        
        # add nebular lines to the spectrum
        if self.ssp.params['nebemlineinspec'] == True:
            #define the minimum resolution of the emission lines based on the resolution of the spectral library
            if self.ssp.params["smooth_velocity"] == True:
                dlam = self.ssp.emline_wavelengths*self.ssp.params["sigma_smooth"]/2.9979E18*1E13 #smoothing variable is in km/s
            else:
                dlam = self.ssp.params["sigma_smooth"] #smoothing variable is in AA
            nearest_id = np.searchsorted(wave, self.ssp.emline_wavelengths)
            neb_res_min = wave[nearest_id]-wave[nearest_id-1]
            dlam = np.max([dlam,neb_res_min], axis=0)
            eline_lums = self.get_galaxy_elines()[1]
            gaussnebarr = [1./np.sqrt(2*np.pi)/dlam[i]*np.exp(-(wave-self.ssp.emline_wavelengths[i])**2/2/dlam[i]**2) \
            /2.9979E18*self.ssp.emline_wavelengths[i]**2 for i in range(len(eline_lums))]
            for i in range(len(eline_lums)):
                spec += eline_lums[i]*gaussnebarr[i]

        return wave, spec, self.ssp.stellar_mass / mtot

    def get_galaxy_elines(self):
        """Get the wavelengths and specific emission line luminosity of the nebular emission lines
        predicted by FSPS. These lines are in units of Lsun/solar mass formed.
        This assumes that `get_galaxy_spectrum` has already been called.
        :param use_stellar_ionizing_spectrum:
            If true, fit the csp and to get the ionizing spectrum parameters, else read from the model
        :param ionspec_index1, ionspec_index2, ionspec_index3, ionspec_index4, ionspec_logLratio1, ionspec_logLratio2, ionspec_logLratio3:
            ionizing parameters, follow the range 
            ionspec_index1: [3.2, 42], ionspec_index2: [-0.3, 30],
            ionspec_index3: [-1, 14], ionspec_index4: [-1.7, 8],
            logLratio1: [-2.2, 9.9], logLratio2: [-3.5, 2.1], logLratio3: [-2.6, 3.6]
        :param gas_logu, gas_logn, gas_logz, gas_logno, gas_logco:
            nebular parameters, follow the range
            gas_logu: [-4, -1], gas_logn: [1, 4], gas_logz: [-2.2, 0.5],
            gas_logno: [-1, log10(5.4)], gas_logco: [-1, log10(5.4)]
        :returns ewave:
            The *restframe* wavelengths of the emission lines, AA
        :returns elum:
            Specific luminosities of the nebular emission lines,
            Lsun/stellar mass formed
        """

        ewave = self.ssp.emline_wavelengths
        # This allows subclasses to set their own specific emission line
        # luminosities within other methods, e.g., get_galaxy_spectrum, by
        # populating the `_specific_line_luminosity` attribute.
        elum = getattr(self, "_line_specific_luminosity", None)

        if elum is None:
            neb_flag = self.ssp.params['add_neb_emission']
            self.update(add_neb_emission=True)
            use_grid_lines = self.ssp.emline_luminosity.copy()[[47,71,106]] # use Nell's nebular grid for [NeIV]4720, [ArIV]7330, [SIV]10.5m
            self.update(add_neb_emission=neb_flag)
            if self.params["use_stellar_ionizing_spectrum"] == True:
                assert bool(self.ssp.params['add_neb_emission']) is False, "params['add_neb_emission'] is True, set to False to fit the raw ionizing spectrum with no nebular emission"
                assert bool(self.ssp.params['add_neb_continuum']) is False, "params['add_neb_continuum'] is True, set to False to fit the raw ionizing spectrum with no nebular emission"
                popt = self._ionparam #powerlaw indexes and log normalizations from get_galaxy_spectrum()
                elum = line_predict(gammas=popt[:,0],
                                    log_L_ratios=self._logLratios,
                                    log_QH=logQ(self.ssp.params['gas_logu'], lognH=self.params['gas_logn']),
                                    n_H=10**self.params['gas_logn'],
                                    log_OH_ratio=self.ssp.params['gas_logz'],
                                    log_NO_ratio=self.params['gas_logno'],
                                    log_CO_ratio=self.params['gas_logco'],
                                   ).nn_predict()[1]
                elum = elum/3.839E33/10**logQ(self.ssp.params['gas_logu'], lognH=self.params['gas_logn'])*self._ionQ # convert to the unit in FSPS
            else:
                elum = line_predict(gammas=[self.params['ionspec_index1'], self.params['ionspec_index2'], 
                                            self.params['ionspec_index3'], self.params['ionspec_index4']],
                                    log_L_ratios=[self.params['ionspec_logLratio1'], self.params['ionspec_logLratio2'],
                                                  self.params['ionspec_logLratio3']],
                                    log_QH=logQ(self.ssp.params['gas_logu'], lognH=self.params['gas_logn']),
                                    n_H=10**self.params['gas_logn'],
                                    log_OH_ratio=self.ssp.params['gas_logz'],
                                    log_NO_ratio=self.params['gas_logno'],
                                    log_CO_ratio=self.params['gas_logco'],
                                   ).nn_predict()[1]
                elum = elum/3.839E33 # divided by Lsun coded in FSPS
            # attenuate the line emission
            elum = elum*extinction(ewave,dtype=self.ssp.params['dust_type'],
                                   dust_index=self.ssp.params['dust_index'],dust2=self.ssp.params['dust2'],
                                   dust1_index=self.ssp.params['dust1_index'],dust1=self.ssp.params['dust1'])
            elum[[47,71,106]] = use_grid_lines

            #if elum.ndim > 1:
            #    elum = elum[0]
            #if self.ssp.params["sfh"] == 3:
            # tabular sfh
            mass = np.sum(self.params.get('mass', 1.0))
            elum /= mass

        return ewave, elum

class MultiSSPBasis(SSPBasis):
    """An array of basis spectra with different ages, metallicities, and possibly dust
    attenuations.
    """
    def get_galaxy_spectrum(self):
        raise(NotImplementedError)
