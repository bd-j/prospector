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


class StellarPopBasis(object):
    """
    A class that wraps the python-fsps StellarPopulation object in order to
    include more functionality and to allow 'fast' model generation in some
    situations by storing an easily accessible spectral grid.

    :param compute_vega_mags:

    :param zcontinuous:
        Flag to indicate the type of metallicity interpolation.

    :param safe:
        If ``True``, use the get_spectrum() method of of the StellarPopulation
        object to generate your SSPs.  This means that COMPSP in FSPS will do
        all the dust attenuation and emission, all the nebular emission, the
        smoothing, redshifting, etc.  If safe=``False`` then the ztinterp()
        method is used and the dust attenuation, smoothing, and redshifting are
        done by this code, not COMPSP.  There is no dust emission, or varying
        physical nebular parameters.
    """

    def __init__(self, compute_vega_mags=False, zcontinuous=1,
                 debug=False, safe=False, **kwargs):

        self.debug = debug
        self.safe = safe
        # This is a StellarPopulation object from fsps
        self.ssp = fsps.StellarPopulation(compute_vega_mags=compute_vega_mags,
                                          zcontinuous=zcontinuous)

        # This is the main state vector for the model
        self.params = {'outwave': self.ssp.wavelengths.copy(),
                       'dust_tesc': 0.00, 'dust1': 0., 'dust2': 0.,
                       'mass': np.array([1.0]), 'zmet': np.array([0.0])}

        self.ssp_dirty = True

        # These are the parameters whose change will force a regeneration of
        # the basis from the SSPs (but will not force the SSPs to be
        # regenerated)
        if self.safe:
            self.basis_params = ['tage', 'logzsol', 'zmet']
        else:
            self.basis_params = ['tage', 'zmet', 'logzsol']

        self.basis_dirty = True

    def get_spectrum(self, outwave=None, filters=None, nebular=True, **params):
        """Return a spectrum for the given parameters.  If necessary the SSPs
        are updated, and if necessary the component spectra are updated, before
        being combined here.

        :param params:
            A dictionary-like of the model parameters.  Should contain ``mass``
            as a parameter.

        :param outwave:
            The output wavelength points at which model estimates are desired,
            ndarray of shape (nwave,)

        :param filters:
             A list of filters in which synthetic photometry is
             desired.  List of length (nfilt,)

        :param nebular: (Default: True)
            If True, add a nebular spectrum to the total spectrum.  Note that
            this is currently not added to the photometry as well.

        :returns spec:
            The spectrum at the wavelength points given by outwave, ndarray of
            shape (nwave,).  Units are erg/s/cm^2/AA.

        :returns phot:
            The synthetc photometry through the provided filters, ndarray of
            shape (nfilt,).  Note, the units are *apparent maggies*.

        :returns extra:
            Any extra parameters (like stellar mass) that you want to return.
        """
        spec, neb, phot, ex = self.get_components(outwave, filters, **params)
        total_spec = (spec * self.params['mass'][:, None]).sum(axis=0)
        if nebular:
            total_spec += neb
        total_phot = (phot * self.params['mass'][:, None]).sum(axis=0)
        extra = (ex * self.params['mass']).sum()

        return total_spec, total_phot, extra

    def get_components(self, outwave, filters, **params):
        """Return the component spectra for the given parameters, making sure
        to update the components if necessary.

        :param params:
            A dictionary-like of the model parameters.

        :param outwave:
            The output wavelength points at which model estimates are desired,
            ndarray of shape (nwave,)

        :param filters:
             A list of filters in which synthetic photometry is desired.  List
             of length (nfilt,)

        :returns cspec:
            The spectrum at the wavelength points given by outwave, ndarray of
            shape (ncomp,nwave).  Units are erg/s/cm^2/AA/M_sun.

        :returns nebspec:
            The nebular spectrum at the wavelength points given by outwave,
            ndarray of shape (nwave).  Units are erg/s/cm^2/AA

        :returns cphot:
            The synthetc photometry through the provided filters, ndarray of
            shape (ncomp,nfilt).  Units are *apparent maggies*.

        :returns extra:
            Any extra parameters (like stellar mass) that you want to return.
        """
        if outwave is not None:
            params['outwave'] = outwave
        # This will rebuild the basis if relevant parameters changed
        self.update(params)

        # distance dimming and conversion from Lsun/AA to cgs
        dist10 = self.params.get('lumdist', 1e-5)/1e-5  # distance in units of 10s of pcs
        dfactor = to_cgs / dist10**2

        nebspec = self.nebular(params, self.params['outwave']) * dfactor
        cspec = np.empty([self.nbasis, len(outwave)])
        cphot = np.empty([self.nbasis, np.size(filters)])
        for i in range(self.nbasis):
            cspec[i,:], cphot[i,:] = self.process_component(i, outwave, filters)

        return cspec * dfactor, nebspec, cphot * dfactor, self.basis_mass

    def process_component(self, i, outwave, filters):
        """Basically do all the COMPSP stuff for one component.
        """
        cspec = self.basis_spec[i, :].copy()
        cphot = 0
        inwave = self.ssp.wavelengths

        if self.safe:
            cspec = np.interp(self.params['outwave'], vac2air(inwave), cspec/a)
            cphot = 10**(-0.4 * getSED(inwave, cspec/a, filters))
            return cspec, cphot
            
        # Dust attenuation
        tage = self.params['tage'][i]
        tesc = self.params.get('dust_tesc', 0.01)
        dust1 = self.params.get('dust1', 0.0)
        dust2 = self.params['dust2']
        a = (1 + self.params.get('zred', 0.0))
        dust = (tage < tesc) * dust1 + dust2
        att = self.params['dust_curve'][0](inwave, **self.params)
        cspec *= np.exp(-att*dust)

        if filters is not None:
            cphot = 10**(-0.4 * getSED(inwave*a, cspec / a, filters))

        # Wavelength scale.  Broadening and redshifting and placing on output
        # wavelength grid
        if self.params.get('lsf', [None])[0] is not None:
            cspec = smoothspec(vac2air(inwave) * a, cspec / a,
                               self.params['sigma_smooth'], **self.params)
        else:
            sigma = self.params.get('sigma_smooth', 0.0)
            cspec = self.ssp.smoothspec(inwave, cspec, sigma)
            cspec = np.interp(self.params['outwave'], vac2air(inwave * a), cspec/a)

        return cspec, cphot

    def nebular(self, params, outwave):
        """If the emission_rest_wavelengths parameter is present, return a
        nebular emission line spectrum.  Currently uses several approximations
        for the velocity broadening.  Currently does *not* affect photometry.
        Only provides samples of the nebular spectrum at outwave, so will not
        be correct for total power unless outwave densley samples the emission
        dispersion.

        :returns nebspec:
            The nebular emission in the observed frame, at the wavelengths
            specified by the obs['wavelength'].
        """
        if 'emission_rest_wavelengths' not in params:
            return 0.
        
        mu = vac2air(params['emission_rest_wavelengths'])
        # try to get a nebular redshift, otherwise use stellar redshift,
        # otherwise use no redshift
        a1 = params.get('zred_emission', self.params.get('zred', 0.0)) + 1.0
        A = params.get('emission_luminosity', 0.)
        sigma = params.get('emission_disp', 10.)
        if params.get('smooth_velocity', False):
            # This is an approximation to get the dispersion in terms of
            # wavelength at the central line wavelength, but should work much
            # of the time
            sigma = mu * sigma / 2.998e5
        return gauss(outwave, mu * a1, A, sigma * a1)

    def update(self, newparams):
        """Update the parameters, recording whether it was new for the ssp or
        basis parameters.  If either of those changed, regenerate the relevant
        spectral grid(s).
        """
        for k, v in list(newparams.items()):
            if k in self.basis_params:
                # Make sure parameter is in dict, and check if it changed
                if k not in self.params:
                    self.basis_dirty = True
                    self.params[k] = v
                if np.any(v != self.params.get(k)):
                    self.basis_dirty = True
            else:
                try:
                    # here the sps.params.dirtiness should increase to 2 if
                    # there was a change
                    self.ssp.params[k] = v[0]
                except KeyError:
                    pass
            # now update params
            self.params[k] = np.copy(np.atleast_1d(v))
            # if we changed only csp_params but are relying on COMPSP, make
            # sure we remake the basis
            if self.safe and (self.ssp.params.dirtiness == 1):
                self.basis_dirty = True
            # if we changed only csp_params propagate them through but don't
            # force basis remake (unless basis_dirty)
            if self.ssp.params.dirtiness == 1:
                self.ssp._update_params()

        if self.basis_dirty | (self.ssp.params.dirtiness == 2):
            self.build_basis()

    def build_basis(self):
        """Rebuild the component spectra from the SSPs.  The component spectra
        include dust attenuation, redshifting, and spectral regridding.  This
        is basically a proxy for COMPSP from FSPS, with a few small
        differences.  In particular, there is interpolation in metallicity and
        the redshift and the output wavelength grid are taken into account.
        The dust treatment is less sophisticated.

        The assumption is that the basis is a N_z by N_age (by N_wave) array
        where the z values and age values are given by vectors located in
        params['tage'] and params['zmet']

        This method is only called by self.update if necessary.

        :param outwave:
            The output wavelength points at which model estimates are desired,
            ndarray of shape (nwave,)
        """
        if self.debug:
            print('sps_basis: rebuilding basis')
        # Setup the internal component basis arrays
        inwave = self.ssp.wavelengths
        nbasis = len(np.atleast_1d(self.params['mass']))
        self.nbasis = nbasis
        # nbasis = ( len(np.atleast_1d(self.params['zmet'])) *
        #            len(np.atleast_1d(self.params['tage'])) )
        self.basis_spec = np.zeros([nbasis, len(inwave)])
        self.basis_mass = np.zeros(nbasis)

        i = 0
        tesc = self.params['dust_tesc']
        dust1, dust2 = self.params['dust1'], self.params['dust2']
        for j, zmet in enumerate(self.params['zmet']):
            for k, tage in enumerate(self.params['tage']):
                # get the intrinsic spectrum at this metallicity and age
                if self.safe:
                    # do it using compsp
                    if self.ssp._zcontinuous > 0:
                        self.ssp.params['logzsol'] = zmet
                    else:
                        self.ssp.params['zmet'] = zmet
                    w, spec = self.ssp.get_spectrum(tage=tage, peraa=True)
                    mass = self.ssp.stellar_mass
                else:
                    # do it by hand.  Faster but dangerous
                    spec, mass, lbol = self.ssp.ztinterp(zmet, tage, peraa=True)
                self.basis_spec[i, :] = spec
                self.basis_mass[i] = mass
                i += 1
        self.basis_dirty = False

    @property
    def wavelengths(self):
        return self.ssp.wavelengths


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

    def get_spectrum(self, outwave=None, filters=None, **params):
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
            The apparent (redshifted) observed frame maggies in each of the filters.

        :returns extras:
            A list of None type objects, only included for consistency with the
            SedModel class.
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

        if outwave is not None:
            w = self.csp.wavelengths
            spec = np.interp(outwave, w, spec)
        # Modern FSPS does the distance modulus for us in get_mags,
        # !but python-FSPS does not!
        if self.params['zred'] == 0:
            # Use 10pc for the luminosity distance (or a number
            # provided in the dist key in units of Mpc)
            dfactor = (self.params.get('dist', 1e-5) * 1e5)**2
            # spectrum stays in L_sun/Hz
            dfactor_spec = 1.0
        else:
            dfactor = ((cosmo.luminosity_distance(self.csp.params['zred']).value *
                        1e5)**2 / (1 + self.params['zred']))
            # convert to maggies
            dfactor_spec = to_cgs / 1e3 / dfactor / (3631*jansky_mks)
        return (spec * dfactor_spec, maggies / dfactor, extra)

    def one_sed(self, component_index=0, filterlist=[]):
        """Get the SED of one component for a multicomponent composite SFH.
        Should set this up to work as an iterator.

        :param component_index:
            Integer index of the component to calculate the SED for.

        :param filterlist:
            A list of strings giving the (FSPS) names of the filters onto which
            the spectrum will be projected.

        :returns spec:
            The restframe spectrum in units of maggies.

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
        # Now get the magnitudes and spectrum
        w, spec = self.csp.get_spectrum(tage=self.csp.params['tage'], peraa=False)
        mags = getSED(w , lightspeed/w**2 * spec * to_cgs, filterlist)
        #mags = self.csp.get_mags(tage=self.csp.params['tage'], bands=filterlist)
        # Normalize by (current) stellar mass and get correct units
        cmass = self.csp.stellar_mass
        mass_norm = mass / cmass
        return mass_norm * spec, mass_norm * 10**(-0.4*(mags)), cmass


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
