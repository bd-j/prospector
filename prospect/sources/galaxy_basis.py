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

__all__ = ["StellarPopBasis", "CSPBasis", "to_cgs"]

# Useful constants
lsun = 3.846e33
pc = 3.085677581467192e18  # in cm
lightspeed = 2.998e18  # AA/s
jansky_mks = 1e-26
# value to go from L_sun/AA to erg/s/cm^2/AA at 10pc
to_cgs = lsun/(4.0 * np.pi * (pc*10)**2)


class CSPBasis(object):
    """
    A class for composite stellar populations, which can be composed from
    multiple versions of parameterized SFHs.  Should replace CSPModel.
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

        if outwave is not None:
            w = self.csp.wavelengths
            spec = np.interp(outwave, w, spec)
        # Distance dimming and unit conversion
        if (self.params['zred'] == 0) or ('lumdist' in self.params):
            # Use 10pc for the luminosity distance (or a number provided in the
            # lumdist key in units of Mpc).  Do not apply cosmological (1+z)
            # factor to the flux.
            dfactor = (self.params.get('lumdist', 1e-5) * 1e5)**2
            a = 1.0
        else:
            # Use the comsological luminosity distance implied by this
            # redshift.  Incorporate cosmological (1+z) factor on the flux.
            lumdist = cosmo.luminosity_distance(self.params['zred']).value
            dfactor = (lumdist * 1e5)**2
            a = (1 + self.params['zred'])
        if peraa:
            # spectrum will be in erg/s/cm^2/AA
            spec *= to_cgs * a / dfactor * lightspeed / outwave**2
        else:
            # Spectrum will be in maggies
            spec *= to_cgs * a / dfactor / 1e3 / (3631*jansky_mks)

        # Convert from absolute maggies to apparent maggies
        maggies *= a / dfactor
            
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
                if k == 'zmet':
                    vv = np.abs(v - (np.arange(len(self.csp.zlegend)) + 1)).argmin() + 1
                else:
                    vv = v.copy()
                self.csp.params[k] = vv
            if k == 'mass':
                mass = v
        # Now get the magnitudes and spectrum.  The spectrum is in units of
        # Lsun/Hz/per solar mass *formed*
        w, spec = self.csp.get_spectrum(tage=self.csp.params['tage'], peraa=False)
        mags = getSED(w, lightspeed/w**2 * spec * to_cgs, filterlist)
        mfrac = self.csp.stellar_mass
        if np.all(self.params.get('mass_units', 'mstar') == 'mstar'):
            # Convert normalization units from per stellar masss to per mass formed
            mass /= mfrac
        # Output correct units
        return mass * spec, mass * 10**(-0.4*(mags)), mfrac


def gauss(x, mu, A, sigma):
    """Lay down mutiple gaussians on the x-axis.
    """
    mu, A, sigma = np.atleast_2d(mu), np.atleast_2d(A), np.atleast_2d(sigma)
    val = (A / (sigma * np.sqrt(np.pi * 2)) *
           np.exp(-(x[:, None] - mu)**2 / (2 * sigma**2)))
    return val.sum(axis=-1)


def selftest():
    from sedpy.observate import load_filters
    sps = sps_basis.StellarPopBasis(debug=True)
    params = {}
    params['tage'] = np.array([1, 2, 3, 4.])
    params['zmet'] = np.array([-0.5, 0.0])
    ntot = len(params['tage']) * len(params['zmet'])
    params['mass'] = np.random.uniform(0, 1, ntot)
    params['sigma_smooth'] = 100.
    outwave = sps.ssp.wavelengths
    flist = ['sdss_u0', 'sdss_r0']
    filters = load_filters(flist)

    # get a spectrum
    s, p, e = sps.get_spectrum(params, outwave, filters)
    # change parameters that affect neither the basis nor the ssp, and
    # get spectrum again
    params['mass'] = np.random.uniform(0, 1, ntot)
    s, p, e = sps.get_spectrum(params, outwave, filters)
    # lets get the basis components while we're at it
    bs, bp, be = sps.get_components(params, outwave, filters)
    # change something that affects the basis
    params['tage'] += 1.0
    bs, bp, be = sps.get_components(params, outwave, filters)
    # try a single age pop at arbitrary metallicity
    params['tage'] = 1.0
    params['zmet'] = -0.2
    bs, bp, be = sps.get_components(params, outwave, filters)
