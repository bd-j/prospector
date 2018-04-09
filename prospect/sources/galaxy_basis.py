from itertools import chain
import numpy as np
from copy import deepcopy

from .ssp_basis import SSPBasis
from ..utils.smoothing import smoothspec
from .constants import cosmo, lightspeed, jansky_cgs, to_cgs_at_10pc

try:
    import fsps
    from sedpy.observate import getSED, vac2air, air2vac
except(ImportError):
    pass

__all__ = ["CSPSpecBasis", "MultiComponentCSPBasis",
           "to_cgs"]


to_cgs = to_cgs_at_10pc


class CSPSpecBasis(SSPBasis):

    """A subclass of :py:class:`SSPBasis` for combinations of N composite
    stellar populations (including single-age populations). The number of
    composite stellar populations is given by the length of the ``"mass"``
    parameter. Other population properties can also be vectors of the same
    length as ``"mass"`` if they are independent for each component.
    """

    def __init__(self, zcontinuous=1, reserved_params=['zred', 'sigma_smooth'],
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
        spectra = []
        mass = np.atleast_1d(self.params['mass']).copy()
        mfrac = np.zeros_like(mass)
        # Loop over mass components
        for i, m in enumerate(mass):
            self.update_component(i)
            wave, spec = self.ssp.get_spectrum(tage=self.ssp.params['tage'],
                                               peraa=False)
            spectra.append(spec)
            mfrac[i] = (self.ssp.stellar_mass)

        # Convert normalization units from per stellar mass to per mass formed
        if np.all(self.params.get('mass_units', 'mformed') == 'mstar'):
            mass /= mfrac
        spectrum = np.dot(mass, np.array(spectra)) / mass.sum()
        mfrac_sum = np.dot(mass, mfrac) / mass.sum()

        return wave, spectrum, mfrac_sum


class MultiComponentCSPBasis(CSPSpecBasis):

    """Similar to :py:class`CSPSpecBasis`, a class for combinations of N composite stellar
    populations (including single-age populations). The number of composite
    stellar populations is given by the length of the `mass` parameter.

    However, in MultiComponentCSPBasis the SED of the different components are
    tracked, and in get_spectrum() photometry can be drawn from a given
    component or from the sum.
    """
    
    def get_galaxy_spectrum(self, **params):
        """Update parameters, then loop over each component getting a spectrum
        for each.  Return all the component spectra, plus the sum.

        :param params:
            A parameter dictionary that gets passed to the ``self.update``
            method and will generally include physical parameters that control
            the stellar population and output spectrum or SED, some of which
            may be vectors for the different componenets

        :returns wave:
            Wavelength in angstroms.

        :returns spectrum:
            Spectrum in units of Lsun/Hz/solar masses formed.  ndarray of
            shape(ncomponent+1, nwave).  The last element is the sum of the
            previous elements.

        :returns mass_fraction:
            Fraction of the formed stellar mass that still exists, ndarray of
            shape (ncomponent+1,)
        """
        self.update(**params)
        spectra = []
        mass = np.atleast_1d(self.params['mass']).copy()
        mfrac = np.zeros_like(mass)
        # Loop over mass components
        for i, m in enumerate(mass):
            self.update_component(i)
            wave, spec = self.ssp.get_spectrum(tage=self.ssp.params['tage'],
                                               peraa=False)
            spectra.append(spec)
            mfrac[i] = (self.ssp.stellar_mass)

        # Convert normalization units from per stellar mass to per mass formed
        if np.all(self.params.get('mass_units', 'mformed') == 'mstar'):
            mass /= mfrac
        spectrum = np.dot(mass, np.array(spectra)) / mass.sum()
        mfrac_sum = np.dot(mass, mfrac) / mass.sum()

        return wave, np.squeeze(spectra + [spectrum]), np.squeeze(mfrac.tolist() + [mfrac_sum])

    def get_spectrum(self, outwave=None, filters=None, component=-1, **params):
        """Get a spectrum and SED for the given params, choosing from different
        possible components.

        :param outwave: (default: None)
            Desired *vacuum* wavelengths.  Defaults to the values in
            `sps.wavelength`.

        :param peraa: (default: False)
            If `True`, return the spectrum in erg/s/cm^2/AA instead of AB
            maggies.

        :param filters: (default: None)
            A list of filter objects for which you'd like photometry to be
            calculated.

        :param component: (optional, default: -1)
            An optional array where each element gives the index of the
            component from which to choose the magnitude.  scalar or iterable
            of same length as `filters`

        :param **params:
            Optional keywords giving parameter values that will be used to
            generate the predicted spectrum.

        :returns spec:
            Observed frame component spectra in AB maggies, unless `peraa=True` in which
            case the units are erg/s/cm^2/AA.  (ncomp+1, nwave)

        :returns phot:
            Observed frame photometry in AB maggies, ndarray of shape (ncomp+1, nfilters)

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
            # Magic to only do filter projections for unique filters, and get a
            # mapping back into this list of unique filters
            # note that this may scramble order of unique_filters
            fnames = [f.name for f in filters]
            unique_names, uinds, filter_ind = np.unique(fnames, return_index=True, return_inverse=True)
            unique_filters = np.array(filters)[uinds]
            mags = getSED(wa, lightspeed/wa**2 * sa * to_cgs, unique_filters)
            phot = np.atleast_1d(10**(-0.4 * mags))
        else:
            phot = 0.0
            filter_ind = 0

        # Distance dimming and unit conversion
        zred = self.params.get('zred', 0.0)
        if (zred == 0) or ('lumdist' in self.params):
            # Use 10pc for the luminosity distance (or a number
            # provided in the dist key in units of Mpc)
            dfactor = (self.params.get('lumdist', 1e-5) * 1e5)**2
        else:
            lumdist = cosmo.luminosity_distance(zred).value
            dfactor = (lumdist * 1e5)**2

        # Spectrum will be in maggies
        sa *= to_cgs / dfactor / (3631*jansky_cgs)

        # Convert from absolute maggies to apparent maggies
        phot /= dfactor

        # Mass normalization
        mass = np.atleast_1d(self.params['mass'])
        mass = np.squeeze(mass.tolist() + [mass.sum()])

        sa = (sa * mass[:, None])
        phot = (phot * mass[:, None])[component, filter_ind]

        return sa, phot, mfrac


def gauss(x, mu, A, sigma):
    """Lay down mutiple gaussians on the x-axis.
    """
    mu, A, sigma = np.atleast_2d(mu), np.atleast_2d(A), np.atleast_2d(sigma)
    val = (A / (sigma * np.sqrt(np.pi * 2)) *
           np.exp(-(x[:, None] - mu)**2 / (2 * sigma**2)))
    return val.sum(axis=-1)
