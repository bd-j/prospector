from itertools import chain
import numpy as np
from copy import deepcopy

from .ssp_basis import SSPBasis
from ..utils.smoothing import smoothspec
from sedpy.observate import getSED, vac2air, air2vac
from .constants import lightspeed, jansky_cgs, to_cgs_at_10pc

try:
    import fsps
except(ImportError):
    pass
try:
    from astropy.cosmology import WMAP9 as cosmo
except(ImportError):
    pass

__all__ = ["CSPSpecBasis", "to_cgs"]


to_cgs = to_cgs_at_10pc


class CSPSpecBasis(SSPBasis):

    """A class for combinations of N composite stellar populations (including
    single-age populations). The number of composite stellar populations is
    given by the length of the `mass` parameter.
    """

    def __init__(self, compute_vega_mags=False, zcontinuous=1, vactoair_flag=False,
                 reserved_params=['zred', 'sigma_smooth'], **kwargs):

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


def gauss(x, mu, A, sigma):
    """Lay down mutiple gaussians on the x-axis.
    """
    mu, A, sigma = np.atleast_2d(mu), np.atleast_2d(A), np.atleast_2d(sigma)
    val = (A / (sigma * np.sqrt(np.pi * 2)) *
           np.exp(-(x[:, None] - mu)**2 / (2 * sigma**2)))
    return val.sum(axis=-1)
