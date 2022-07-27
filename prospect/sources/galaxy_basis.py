from itertools import chain
import numpy as np
from copy import deepcopy

try:
    import fsps
except(ImportError, RuntimeError):
    pass


__all__ = ["SSPBasis", "FastStepBasis",
           "CSPSpecBasis"]


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
        """Update parameters, then get the SSP spectrum

        Returns
        -------
        wave : ndarray
            Restframe avelength in angstroms.

        spectrum : ndarray
            Spectrum in units of Lsun/Hz per solar mass formed.

        mass_fraction : float
            Fraction of the formed stellar mass that still exists.
        """
        self.update(**params)
        wave, spec = self.ssp.get_spectrum(tage=float(self.params['tage']), peraa=False)
        return wave, spec, self.ssp.stellar_mass

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

    @property
    def logage(self):
        return self.ssp.ssp_ages.copy()

    @property
    def wavelengths(self):
        return self.ssp.wavelengths.copy()


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


class CSPSpecBasis(SSPBasis):

    """A subclass of :py:class:`SSPBasis` for combinations of N composite
    stellar populations (including single-age populations). The number of
    composite stellar populations is given by the length of the ``"mass"``
    parameter. Other population properties can also be vectors of the same
    length as ``"mass"`` if they are independent for each component.
    """

    def __init__(self, zcontinuous=1, reserved_params=['sigma_smooth'],
                 vactoair_flag=False, compute_vega_mags=False, **kwargs):

        # This is a StellarPopulation object from fsps
        self.ssp = fsps.StellarPopulation(compute_vega_mags=compute_vega_mags,
                                          zcontinuous=zcontinuous,
                                          vactoair_flag=vactoair_flag)
        self.reserved_params = reserved_params
        self.params = {}
        self.update(**kwargs)

    def update(self, **params):
        """Update the `params` attribute, making parameters scalar if possible.
        """
        for k, v in list(params.items()):
            # try to make parameters scalar
            try:
                if (len(v) == 1) and callable(v[0]):
                    self.params[k] = v[0]
                else:
                    self.params[k] = np.squeeze(v)
            except:
                self.params[k] = v

    def update_component(self, component_index):
        """Pass params that correspond to a single component through to the
        fsps.StellarPopulation object.

        :param component_index:
            The index of the component for which to pull out individual
            parameters that are passed to the fsps.StellarPopulation object.
        """
        for k, v in list(self.params.items()):
            # Parameters named like FSPS params but that we reserve for use
            # here.  Do not pass them to FSPS.
            if k in self.reserved_params:
                continue
            # Otherwise if a parameter exists in the FSPS parameter set, pass a
            # copy of it in.
            if k in self.ssp.params.all_params:
                v = np.atleast_1d(v)
                try:
                    # Try to pull the relevant component.
                    this_v = v[component_index]
                except(IndexError):
                    # Not enogh elements, use the last element.
                    this_v = v[-1]
                except(TypeError):
                    # It was scalar, use that value for all components
                    this_v = v

                self.ssp.params[k] = deepcopy(this_v)

    def get_galaxy_spectrum(self, **params):
        """Update parameters, then loop over each component getting a spectrum
        for each and sum with appropriate weights.

        :param params:
            A parameter dictionary that gets passed to the ``self.update``
            method and will generally include physical parameters that control
            the stellar population and output spectrum or SED.

        :returns wave:
            Wavelength in angstroms.

        :returns spectrum:
            Spectrum in units of Lsun/Hz/solar masses formed.

        :returns mass_fraction:
            Fraction of the formed stellar mass that still exists.
        """
        self.update(**params)
        spectra, linelum = [], []
        mass = np.atleast_1d(self.params['mass']).copy()
        mfrac = np.zeros_like(mass)
        # Loop over mass components
        for i, m in enumerate(mass):
            self.update_component(i)
            wave, spec = self.ssp.get_spectrum(tage=self.ssp.params['tage'],
                                               peraa=False)
            spectra.append(spec)
            mfrac[i] = (self.ssp.stellar_mass)
            linelum.append(self.ssp.emline_luminosity)

        # Convert normalization units from per stellar mass to per mass formed
        if np.all(self.params.get('mass_units', 'mformed') == 'mstar'):
            mass /= mfrac
        spectrum = np.dot(mass, np.array(spectra)) / mass.sum()
        self._line_specific_luminosity = np.dot(mass, np.array(linelum)) / mass.sum()
        mfrac_sum = np.dot(mass, mfrac) / mass.sum()

        return wave, spectrum, mfrac_sum

