from itertools import chain
import numpy as np
from copy import deepcopy

from .ssp_basis import SSPBasis
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

__all__ = ["CSPSpecBasis", "CSPBasis", "to_cgs"]

# Useful constants
lsun = 3.846e33
pc = 3.085677581467192e18  # in cm
lightspeed = 2.998e18  # AA/s
jansky_mks = 1e-26
# value to go from L_sun/AA to erg/s/cm^2/AA at 10pc
to_cgs = lsun/(4.0 * np.pi * (pc*10)**2)


class CSPSpecBasis(SSPBasis):

    """A class for combinations of N composite stellar populations (including
    single-age populations). The number of composite stellar populations is
    given by the length of the `mass` parameter.
    """

    def __init__(self, compute_vega_mags=False, zcontinuous=1,
                 reserved_params=['zred', 'sigma_smooth'], **kwargs):

        # This is a StellarPopulation object from fsps
        self.ssp = fsps.StellarPopulation(compute_vega_mags=compute_vega_mags,
                                          zcontinuous=zcontinuous)
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

        :returns wave:
            Wavelength in angstroms.

        :returns spectrum:
            Spectrum in units of Lsun/Hz/solar masses formed.

        :returns mass_fraction:
            Fraction of the formed stellar mass that still exists.
        """
        self.update(**params)
        spectra = []
        mass = np.atleast_1d(self.params['mass'])
        mfrac = np.zeros_like(mass)
        # Loop over mass components
        for i, m in enumerate(mass):
            self.update_component(i)
            wave, spec = self.ssp.get_spectrum(tage=self.csp.params['tage'],
                                               peraa=False)
            spectra.append(spec)
            mfrac[i] = (self.ssp.stellar_mass)

        # Convert normalization units from per stellar mass to per mass formed
        if np.all(self.params.get('mass_units', 'mformed') == 'mstar'):
            mass /= mfrac
        spectrum = np.dot(mass, np.array(spectra)) / mass.sum()
        mfrac_sum = np.dot(mass, mfrac) / mass.sum()

        return wave, spectrum, mfrac_sum


class CSPBasis(object):
    """
    A class for composite stellar populations, which can be composed from
    multiple versions of parameterized SFHs.  Deprecated, Use CSPSpecBasis instead.
    """
    def __init__(self, compute_vega_mags=False, zcontinuous=1, **kwargs):

        # This is a StellarPopulation object from fsps
        self.csp = fsps.StellarPopulation(compute_vega_mags=compute_vega_mags,
                                          zcontinuous=zcontinuous)
        self.params = {}

    def get_spectrum(self, outwave=None, filters=None, peraa=False, **params):
        """Given a theta vector, generate spectroscopy, photometry and any
        extras (e.g. stellar mass).

        :param theta:
            ndarray of parameter values.

        :param sps:
            A python-fsps StellarPopulation object to be used for
            generating the SED.

        :returns spec:
            The restframe spectrum in units of maggies.

        :returns phot:
            The apparent (redshifted) observed frame maggies in each of the
            filters.

        :returns extras:
            A list of the ratio of existing stellar mass to total mass formed
            for each component, length ncomp.
        """
        self.params.update(**params)
        # Pass the model parameters through to the sps object
        ncomp = len(self.params['mass'])
        for ic in range(ncomp):
            s, p, x = self.one_sed(component_index=ic, filterlist=filters)
            try:
                spec += s
                maggies += p
                extra += [x]
            except(NameError):
                spec, maggies, extra = s, p, [x]
        # `spec` is now in Lsun/Hz, with the wavelength array being the
        # observed frame wavelengths.  Flux array (and maggies) have not been
        # increased by (1+z) due to cosmological redshift

        w = self.csp.wavelengths
        if outwave is not None:
            spec = np.interp(outwave, w, spec)
        else:
            outwave = w
        # Distance dimming and unit conversion
        zred = self.params.get('zred', 0.0)
        if (zred == 0) or ('lumdist' in self.params):
            # Use 10pc for the luminosity distance (or a number provided in the
            # lumdist key in units of Mpc).  Do not apply cosmological (1+z)
            # factor to the flux.
            dfactor = (self.params.get('lumdist', 1e-5) * 1e5)**2
            a = 1.0
        else:
            # Use the comsological luminosity distance implied by this
            # redshift.  Cosmological (1+z) factor on the flux was already done in one_sed
            lumdist = cosmo.luminosity_distance(zred).value
            dfactor = (lumdist * 1e5)**2
        if peraa:
            # spectrum will be in erg/s/cm^2/AA
            spec *= to_cgs / dfactor * lightspeed / outwave**2
        else:
            # Spectrum will be in maggies
            spec *= to_cgs / dfactor / 1e3 / (3631*jansky_mks)

        # Convert from absolute maggies to apparent maggies
        maggies /= dfactor

        return spec, maggies, extra

    def one_sed(self, component_index=0, filterlist=[]):
        """Get the SED of one component for a multicomponent composite SFH.
        Should set this up to work as an iterator.

        :param component_index:
            Integer index of the component to calculate the SED for.

        :param filterlist:
            A list of strings giving the (FSPS) names of the filters onto which
            the spectrum will be projected.

        :returns spec:
            The restframe spectrum in units of Lsun/Hz.

        :returns maggies:
            Broadband fluxes through the filters named in ``filterlist``,
            ndarray.  Units are observed frame absolute maggies: M = -2.5 *
            log_{10}(maggies).

        :returns extra:
            The extra information corresponding to this component.
        """
        # Pass the model parameters through to the sps object, and keep track
        # of the mass of this component
        mass = 1.0
        for k, vs in list(self.params.items()):
            try:
                v = vs[component_index]
            except(IndexError, TypeError):
                v = vs
            if k in self.csp.params.all_params:
                self.csp.params[k] = deepcopy(v)
            if k == 'mass':
                mass = v
        # Now get the spectrum.  The spectrum is in units of
        # Lsun/Hz/per solar mass *formed*, and is restframe
        w, spec = self.csp.get_spectrum(tage=self.csp.params['tage'], peraa=False)
        # redshift and get photometry.  Note we are boosting fnu by (1+z) *here*
        a, b = (1 + self.csp.params['zred']), 0.0
        wa, sa = w * (a + b), spec * a  # Observed Frame
        if filterlist is not None:
            mags = getSED(wa, lightspeed/wa**2 * sa * to_cgs, filterlist)
            phot = np.atleast_1d(10**(-0.4 * mags))
        else:
            phot = 0.0

        # now some mass normalization magic
        mfrac = self.csp.stellar_mass
        if np.all(self.params.get('mass_units', 'mstar') == 'mstar'):
            # Convert input normalization units from per stellar masss to per mass formed
            mass /= mfrac
        # Output correct units
        return mass * sa, mass * phot, mfrac

    @property
    def wavelengths(self):
        return self.csp.wavelengths


def gauss(x, mu, A, sigma):
    """Lay down mutiple gaussians on the x-axis.
    """
    mu, A, sigma = np.atleast_2d(mu), np.atleast_2d(A), np.atleast_2d(sigma)
    val = (A / (sigma * np.sqrt(np.pi * 2)) *
           np.exp(-(x[:, None] - mu)**2 / (2 * sigma**2)))
    return val.sum(axis=-1)
