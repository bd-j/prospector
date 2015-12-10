from itertools import chain
import numpy as np
from scipy.spatial import Delaunay
from .smoothing import smoothspec
from sedpy.observate import getSED, vac2air, air2vac

try:
    import fsps
except(ImportError):
    pass

try:
    from sklearn.neighbors import KDTree
except(ImportError):
    from scipy.spatial import  KDTree

# Useful constants
lsun = 3.846e33
pc = 3.085677581467192e18  # in cm
lightspeed = 2.998e18  # AA/s
# value to go from L_sun/AA to erg/s/cm^2/AA at 10pc
to_cgs = lsun/(4.0 * np.pi * (pc*10)**2)


class StellarPopBasis(object):
    """
    A class that wraps the python-fsps StellarPopulation object in
    order to include more functionality and to allow 'fast' model
    generation in some situations by storing an easily accessible
    spectral grid.

    :param compute_vega_mags:

    :param zcontinuous:
        Flag to indicate the type of metallicity interpolation.

    :param safe:
        If ``True``, use the get_spectrum() method of of the
        StellarPopulation object to generate your SSPs.  This means
        that COMPSP in FSPS will do all the dust attenuation and
        emission, all the nebular emission, the smoothing,
        redshifting, etc.  If safe=``False`` then the ztinterp()
        method is used and the dust attenuation, smoothing, and
        redshifting are done by this code, not COMPSP.  There is no
        dust emission, or varying physical nebular parameters.
    """

    def __init__(self, compute_vega_mags=False,
                 zcontinuous=1,
                 debug=False, safe=False, **kwargs):

        self.debug = debug
        self.safe = safe
        # This is a StellarPopulation object from fsps
        self.ssp = fsps.StellarPopulation(compute_vega_mags=compute_vega_mags,
                                          zcontinuous=zcontinuous,
                                          **kwargs)

        # This is the main state vector for the model
        self.params = {'outwave': self.ssp.wavelengths.copy(),
                       'dust_tesc': 0.00, 'dust1': 0., 'dust2': 0.,
                       'mass': np.array([1.0]), 'zmet': np.array([0.0])}

        self.ssp_dirty = True

        # These are the parameters whose change will force a
        # regeneration of the basis from the SSPs (but will not force
        # the SSPs to be regenerated)
        if self.safe:
            self.basis_params = ['tage', 'logzsol', 'zmet']
        #                         'lumdist', 'outwave']
        else:
            self.basis_params = ['tage', 'zmet', 'logzsol']
        #                         'sigma_smooth',
        #                         'dust1', 'dust2', 'dust_tesc', 'dust_curve']
        #                         'lumdist', 'outwave']

        self.basis_dirty = True

    def get_spectrum(self, outwave=None, filters=None, nebular=True, **params):
        """Return a spectrum for the given parameters.  If necessary
        the SSPs are updated, and if necessary the component spectra
        are updated, before being combined here.

        :param params:
            A dictionary-like of the model parameters.  Should contain
            ``mass`` as a parameter.

        :param outwave:
            The output wavelength points at which model estimates are
            desired, ndarray of shape (nwave,)

        :param filters:
             A list of filters in which synthetic photometry is
             desired.  List of length (nfilt,)

        :param nebular: (Default: True)
            If True, add a nebular spectrum to the total spectrum.
            Note that this is currently not added to the photometry as
            well

        :returns spec:
            The spectrum at the wavelength points given by outwave,
            ndarray of shape (nwave,).  Units are erg/s/cm^2/AA

        :returns phot:
            The synthetc photometry through the provided filters,
            ndarray of shape (nfilt,).  Note, the units are *apparent
            maggies*.

        :returns extra:
            Any extra parameters (like stellar mass) that you want to
            return.
        """
        cspec, neb, cphot, cextra = self.get_components(outwave, filters,
                                                        **params)
        spec = (cspec * self.params['mass'][:, None]).sum(axis=0)
        if nebular:
            spec += neb

        phot = (cphot * self.params['mass'][:, None]).sum(axis=0)
        extra = (cextra * self.params['mass']).sum()

        return spec, phot, extra

    def get_components(self, outwave, filters, **params):
        """Return the component spectra for the given parameters,
        making sure to update the components if necessary.

        :param params:
            A dictionary-like of the model parameters.

        :param outwave:
            The output wavelength points at which model estimates are
            desired, ndarray of shape (nwave,)

        :param filters:
             A list of filters in which synthetic photometry is
             desired.  List of length (nfilt,)

        :returns cspec:
            The spectrum at the wavelength points given by outwave,
            ndarray of shape (ncomp,nwave).  Units are
            erg/s/cm^2/AA/M_sun

        :returns nebspec:
            The nebular spectrum at the wavelength points given by outwave,
            ndarray of shape (nwave).  Units are erg/s/cm^2/AA

        :returns cphot:
            The synthetc photometry through the provided filters,
            ndarray of shape (ncomp,nfilt).  Units are
            *apparent maggies*.

        :returns extra:
            Any extra parameters (like stellar mass) that you want to
            return.
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

        if not self.safe:
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

            # Wavelength scale.  Broadening and redshifting and
            # placing on output wavelength grid
            if self.params.get('lsf', [None])[0] is not None:
                cspec = smoothspec(vac2air(inwave) * a, cspec / a,
                                   self.params['sigma_smooth'], **self.params)
            else:
                sigma = self.params.get('sigma_smooth', 0.0)
                cspec = self.ssp.smoothspec(inwave, cspec, sigma)
                cspec = np.interp(self.params['outwave'],
                                  vac2air(inwave * a), cspec/a)
        elif self.safe:
            # Place on output wavelength grid, and get photometry
            cspec = np.interp(self.params['outwave'],
                              vac2air(inwave), cspec/a)
            cphot = 10**(-0.4 * getSED(inwave, cspec/a, filters))

        return cspec, cphot

    def nebular(self, params, outwave):
        """If the emission_rest_wavelengths parameter is present,
        return a nebular emission line spectrum.  Currently uses
        several approximations for the velocity broadening.  Currently
        does *not* affect photometry.  Only provides samples of the
        nebular spectrum at outwave, so will not be correct for total
        power unless outwave densley samples the emission dispersion.

        :returns nebspec:
            The nebular emission in the observed frame, at the wavelengths
            specified by the obs['wavelength'].
        """

        if 'emission_rest_wavelengths' in params:
            mu = vac2air(params['emission_rest_wavelengths'])
            # try to get a nebular redshift, otherwise use stellar
            # redshift, otherwise use no redshift
            a1 = params.get('zred_emission', self.params.get('zred', 0.0)) + 1.0
            A = params.get('emission_luminosity', 0.)
            sigma = params.get('emission_disp', 10.)
            if params.get('smooth_velocity', False):
                # This is an approximation to get the dispersion in
                # terms of wavelength at the central line wavelength,
                # but should work much of the time
                sigma = mu * sigma / 2.998e5
            return gauss(outwave, mu * a1, A, sigma * a1)

        else:
            return 0.

    def update(self, newparams):
        """Update the parameters, recording whether it was new for the
        ssp or basis parameters.  If either of those changed,
        regenerate the relevant spectral grid(s).
        """
        for k, v in newparams.iteritems():
            if k in self.basis_params:
                # make sure parameter is in dict, and check if it changed
                if k not in self.params:
                    self.basis_dirty = True
                    self.params[k] = v
                if np.any(v != self.params.get(k)):
                    self.basis_dirty = True
            else:
                try:
                    # here the sps.params.dirtiness should increase to 2
                    # if there was a change
                    self.ssp.params[k] = v[0]
                except KeyError:
                    pass
            # now update params
            self.params[k] = np.copy(np.atleast_1d(v))
            # if we changed only csp_params but are relying on COMPSP,
            # make sure we remake the basis
            if self.safe and (self.ssp.params.dirtiness == 1):
                self.basis_dirty = True
            # if we changed only csp_params propagate them through but
            # don't force basis remake (unless basis_dirty)
            if self.ssp.params.dirtiness == 1:
                self.ssp._update_params()

        if self.basis_dirty | (self.ssp.params.dirtiness == 2):
            self.build_basis()

    def build_basis(self):
        """Rebuild the component spectra from the SSPs.  The component
        spectra include dust attenuation, redshifting, and spectral
        regridding.  This is basically a proxy for COMPSP from FSPS,
        with a few small differences.  In particular, there is
        interpolation in metallicity and the redshift and the output
        wavelength grid are taken into account.  The dust treatment is
        less sophisticated.

        The assumption is that the basis is a N_z by N_age (by N_wave)
        array where the z values and age values are given by vectors
        located in params['tage'] and params['zmet']

        This method is only called by self.update if necessary.

        :param outwave:
            The output wavelength points at which model estimates are
            desired, ndarray of shape (nwave,)
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


log_rsun_cgs = np.log10(6.955) + 10
log_lsun_cgs = np.log10(3.839) + 33
log_SB_solar = np.log10(5.6704e-5) + 2 * log_rsun_cgs - log_lsun_cgs


class StarBasis(object):

    _spectra = None

    def __init__(self, libname='ckc14_deimos.h5', verbose=False,
                 n_neighbors=0, log_interp=False, logify_Z=True,
                 **kwargs):
        """An object which holds the stellar spectral library, performs
        interpolations of that library, and has methods to return attenuated,
        normalized, smoothed stellar spoectra.

        :param libname:
            Path to the hdf5 file to use for the spectral library. Must have
            "ckc" or "ykc" in the filename (to specify which kind of loader to
            use)

        :param n_neighbors: (default:0)
            Number of nearest neighbors to use when requested parameters are
            outside the convex hull of the library prameters.  If ``0`` then a
            ValueError is raised instead of the nearest spectrum.

        :param verbose:
            If True, print information about the parameters used when a point
            is outside the convex hull
        """
        self.verbose = verbose
        self.logarithmic = log_interp
        self.logify_Z = logify_Z
        self._libname = libname
        self.load_lib(libname)
        # Do some important bookkeeping
        self.stellar_pars = self._libparams.dtype.names
        self.ndim = len(self.stellar_pars)
        self.triangulate()
        try:
            self.build_kdtree()
        except NameError:
            pass
        self.n_neighbors = n_neighbors
        self.params = {}

    def load_lib(self, libname=''):
        """Read a CKC library which has been pre-convolved to be close to your
        resolution.  This library should be stored as an HDF5 file, with the
        datasets ``wavelengths``, ``parameters`` and ``spectra``.  These are
        ndarrays of shape (nwave,), (nmodels,), and (nmodels, nwave)
        respecitvely.  The ``parameters`` array is a structured array.  Spectra
        with no fluxes > 1e-32 are removed from the library
        """
        import h5py
        with h5py.File(libname, "r") as f:
            self._wave = np.array(f['wavelengths'])
            self._libparams = np.array(f['parameters'])
            self._spectra = np.array(f['spectra'])
        # Filter library so that only existing spectra are included
        maxf = np.max(self._spectra, axis=1)
        good = maxf > 1e-32
        self._libparams = self._libparams[good]
        self._spectra = self._spectra[good, :]
        if self.logify_Z:
            self._libparams['Z'] = np.log10(self._libparams['Z'])

    def update(self, **kwargs):
        for k, v in kwargs.iteritems():
            try:
                self.params[k] = np.squeeze(v)
            except:
                pass

    def get_spectrum(self, outwave=None, filters=None, peraa=False, **kwargs):
        """
        :returns spec:
            The spectrum on the outwave grid (assumed in air), in erg/s/Hz.  If
            peraa is True then the spectrum is /AA instead of /Hz. If
            ``lumdist`` is a member of the params dictionary, then the units
            are /cm**2 as well

        :returns phot:
            Observed frame photometry in units of AB maggies.  If ``lumdist``
            is not present in the parameters then these are absolute maggies,
            otherwise they are apparent.

        :returns x:
            A blob of extra quantities (e.g. mass, uncertainty)
        """
        self.update(**kwargs)

        # star spectrum
        wave, spec, unc = self.get_star_spectrum(**self.params)
        spec *= self.normalize()

        # dust
        if 'dust_curve' in self.params:
            att = self.params['dust_curve'](self._wave, **self.params)
            spec *= np.exp(-att)

        # distance dimming
        if 'lumdist' in self.params:
            dist10 = self.params['lumdist']/1e-5  # d in units of 10pc
            spec /= 4 * np.pi * (dist10*pc*10)**2
            conv = 1
        else:
            conv = 4 * np.pi * (pc*10)**2

        # Broadening, redshifting, and interpolation onto observed
        # wavelengths.  The redshift treatment needs to be checked
        a = (1 + self.params.get('zred', 0.0))
        wa, sa = vac2air(wave) * a, spec * a
        if outwave is None:
            outwave = wa
        if 'sigma_smooth' in self.params:
            smspec = self.smoothspec(wa, sa, self.params['sigma_smooth'],
                                     outwave=outwave, **self.params)
        else:
            smspec = np.interp(outwave, wa, sa, left=0, right=0)

        # Photometry (observed frame)
        mags = getSED(wa, sa * lightspeed / wa**2 / conv, filters)
        phot = np.atleast_1d(10**(-0.4 * mags))

        # conversion from /Hz to /AA
        if peraa:
            smspec *= lightspeed / outwave**2

        return smspec / conv, phot, None

    def get_star_spectrum(self, **kwargs):
        """Given stellar parameters, obtain an interpolated spectrum at those
        parameters.

        :param **kwargs:
            Keyword arguments must include values for the ``stellar_pars``
            parameters that are stored in ``_libparams``.

        :returns wave:
            The wavelengths at which the spectrum is defined.

        :returns spec:
            The spectrum interpolated to the requested parameters

        :returns unc:
            The uncertainty spectrum, where the uncertainty is due to
            interpolation error.  Curently unimplemented (i.e. it is a None
            type object)
        """
        inparams = [kwargs[p] for p in self.stellar_pars]
        inparams = np.squeeze(np.array(inparams))
        inds, wghts = self.weights(inparams, **kwargs)
        if self.logarithmic:
            spec = np.exp(np.dot(wghts, np.log(self._spectra[inds, :])))
        else:
            spec = np.dot(wghts, self._spectra[inds, :])
        spec_unc = None
        return self._wave, spec, spec_unc

    def smoothspec(self, wave, spec, sigma, outwave=None, **kwargs):
        outspec = smoothspec(wave, spec, sigma, outwave=outwave, **kwargs)
        return outspec

    def normalize(self):
        """Use either logr or logl to normalize the spectrum.  Both should be
        in solar units.  logr is checked first.

        :returns norm:
            Factor by which the CKC spectrum should be multiplied to get units
            of erg/s/Hz
        """
        if 'logr' in self.params:
            logr = self.params['logr']
        elif 'logl' in self.params:
            logr = (self.params['logl']/2.0 - 2*self.params['logt'] -
                    log_SB_solar / 2 - np.log10(4 * np.pi) / 2)
        else:
            logr = -log_rsun_cgs - np.log10(4 * np.pi)
        logr += log_rsun_cgs
        norm = 4 * np.pi * 10**(2 * logr)
        return norm * 4 * np.pi

    def weights(self, inparams, **extras):
        """Delauynay weighting.  Return indices of the models forming the
        enclosing simplex, as well as the barycentric coordinates of the point
        within this simplex to use as weights.
        """
        triangle_ind = self._dtri.find_simplex(inparams)
        if triangle_ind == -1:
            if self.n_neighbors == 0:
                pstring = ', '.join(self.ndim * ['{}={}'])
                pstring = pstring.format(*chain(*zip(self.stellar_pars, inparams)))
                raise ValueError("Requested spectrum ({}) outside convex hull,"
                                 " and nearest neighbor interpolation turned "
                                 "off.".format(*pstring))
            ind, wght = self.weights_kNN(inparams, k=self.n_neighbors)
            if self.verbose:
                print("Parameters {0} outside model convex hull. "
                      "Using model index {1} instead. ".format(inparams, ind))
            return ind, wght

        inds = self._dtri.simplices[triangle_ind, :]
        transform = self._dtri.transform[triangle_ind, :, :]
        Tinv = transform[:self.ndim, :]
        x_r = inparams - transform[self.ndim, :]
        bary = np.dot(Tinv, x_r)
        last = 1.0 - bary.sum()
        wghts = np.append(bary, last)
        oo = inds.argsort()
        return inds[oo], wghts[oo]

    def triangulate(self):
        """Build the Delauynay Triangulation of the model library.
        """
        # slow.  should use a view based method
        model_points = np.array([list(d) for d in self._libparams])
        self._dtri = Delaunay(model_points)

    def build_kdtree(self):
        """Build the kdtree of the model points.
        """
        # slow.  should use a view based method
        model_points = np.array([list(d) for d in self._libparams])
        self._kdt = KDTree(model_points)

    def weights_kNN(self, target_points, k=1):
        """The interpolation weights are determined from the inverse distance
        to the k nearest neighbors.

        :param target_points: ndarray, shape(ntarg,npar)
            The coordinates to which you wish to interpolate.

        :param k:
            The number of nearest neighbors to use.

        :returns inds: ndarray, shape(ntarg,npar+1)
             The model indices of the interpolates.

        :returns weights: narray, shape (ntarg,npar+1)
             The weights of each model given by ind in the interpolates.
        """
        try:
            dists, inds = self._kdt.query(target_points, k=k,
                                          return_distance=True)
        except:
            return [0], [0]
        inds = np.atleast_1d(np.squeeze(inds))
        if k == 1:
            return inds, np.ones(inds.shape)
        weights = 1 / dists
        # weights[np.isinf(weights)] = large_number
        weights = weights/weights.sum(axis=-1)
        return inds, np.atleast_1d(np.squeeze(weights))

    def param_vector(self, **kwargs):
        """Take a dictionary of parameters and return the stellar library
        parameter vector corresponding to these parameters as an ndarray.
        Raises a KeyError if the dictionary does not contain *all* of the
        required stellar parameters.
        """
        pvec = [kwargs[n] for n in self.stellar_pars]
        return np.array(pvec)

    @property
    def wavelengths(self):
        return self._wave


class BigStarBasis(StarBasis):

    def __init__(self, libname='', verbose=False, log_interp=True,
                 n_neighbors=0,  driver=None, **kwargs):
        """An object which holds the stellar spectral library, performs
        interpolations of that library, and has methods to return attenuated,
        normalized, smoothed stellar spoectra.

        This object is set up to work with large grids, so the models file is
        kept open for acces from disk.  scikits-learn kd-trees are required for
        model access.  Ideally the grid should be regular (though the spacings
        need not be equal along a given dimension).

        :param libname:
            Path to the hdf5 file to use for the spectral library. Must have
            "ckc" or "ykc" in the filename (to specify which kind of loader to
            use)

        :param n_neighbors: (default:0)
            Number of nearest neighbors to use when requested parameters are
            outside the convex hull of the library prameters.  If ``0`` then a
            ValueError is raised instead of the nearest spectrum.

        :param verbose:
            If True, print information about the parameters used when a point
            is outside the convex hull
        """
        self.verbose = verbose
        self.logarithmic = log_interp
        self._libname = libname
        self.load_lib(libname, driver=driver)
        # Do some important bookkeeping
        self.stellar_pars = self._libparams.dtype.names
        self.ndim = len(self.stellar_pars)
        self.lib_as_grid()
        self.n_neighbors = n_neighbors
        self.params = {}

    def load_lib(self, libname='', driver=None):
        """Read a ykc library which has been preconvolved to be close to your
        data resolution. This library should be stored as an HDF5 file, with
        the datasets ``wavelengths``, ``parameters`` and ``spectra``.  These
        are ndarrays of shape (nwave,), (nmodels,), and (nmodels, nwave)
        respecitvely.  The ``parameters`` array is a structured array.  The h5
        file object is left open so that spectra can be accessed from disk.
        """
        import h5py
        f = h5py.File(libname, "r", driver=driver)
        self._wave = np.array(f['wavelengths'])
        self._libparams = np.array(f['parameters'])
        self._spectra = f['spectra']

    def get_star_spectrum(self, **kwargs):
        """Given stellar parameters, obtain an interpolated spectrum at those
        parameters.

        :param **kwargs:
            Keyword arguments must include values for the ``stellar_pars``
            parameters that are stored in ``_libparams``.

        :returns wave:
            The wavelengths at which the spectrum is defined.

        :returns spec:
            The spectrum interpolated to the requested parameters

        :returns unc:
            The uncertainty spectrum, where the uncertainty is due to
            interpolation error.  Curently unimplemented (i.e. it is a None
            type object)
        """
        inds = self.knearest_inds(**kwargs)
        wghts = self.linear_weights(inds, **kwargs)
        # if wghts.sum() < 1.0:
        #     raise ValueError("Something is wrong with the weights")
        good = wghts > 0
        # if good.sum() < 2**self.ndim:
        #     raise ValueError("Did not find all vertices of the hypercube, "
        #                      "or there is no enclosing hypercube in the library.")
        inds = inds[good]
        wghts = wghts[good]
        wghts /= wghts.sum()

        if self.logarithmic:
            spec = np.exp(np.dot(wghts, np.log(self._spectra[inds, :])))
        else:
            spec = np.dot(wghts, self._spectra[inds, :])
        spec_unc = None
        return self._wave, spec, spec_unc

    def lib_as_grid(self):
        """Convert the library parameters to pixel indices in each dimension,
        and build and store a KDTree for the pixel coordinates.
        """
        # Get the unique gridpoints in each param
        self.gridpoints = {}
        for p in self.stellar_pars:
            self.gridpoints[p] = np.unique(self._libparams[p])
        # Digitize the library parameters
        X = np.array([np.digitize(self._libparams[p], bins=self.gridpoints[p],
                                  right=True) for p in self.stellar_pars])
        self.X = X.T
        # Build the KDTree
        self._kdt = KDTree(self.X)#, metric='euclidean')

    def params_to_grid(self, **targ):
        """Convert a set of parameters to grid pixel coordinates.

        :param targ:
            The target parameter location, as keyword arguments.  The elements
            of ``stellar_pars`` must be present as keywords.

        :returns x:
            The target parameter location in pixel coordinates.
        """
        # bin index
        inds = np.array([np.digitize([targ[p]], bins=self.gridpoints[p], right=False) - 1
                         for p in self.stellar_pars])
        inds = np.squeeze(inds)
        # fractional index.  Could use stored denominator to be slightly faster
        try:
            find = [(targ[p] - self.gridpoints[p][i]) /
                    (self.gridpoints[p][i+1] - self.gridpoints[p][i])
                    for i, p in zip(inds, self.stellar_pars)]
        except(IndexError):
            s = ["{0}: min={2} max={3} targ={1}\n".format(p, targ[p], *self.gridpoints[p][[0,-1]])
                 for p in self.stellar_pars]
            raise ValueError("At least one parameter outside grid.\n{}".format(' '.join(s)))
        return inds + np.squeeze(find)

    def knearest_inds(self, **params):
        """Find all parameter ``vertices`` within a sphere of radius
        sqrt(ndim).  The parameter values are converted to pixel coordinates
        before a search of the KDTree.

        :param params:
             Keyword arguments which must include keys corresponding to
             ``stellar_pars``, the parameters of the grid.

        :returns inds:
             The sorted indices of all vertices within sqrt(ndim) of the pixel
             coordinates, corresponding to **params.
        """
        # Convert from physical space to grid index space
        xtarg = self.params_to_grid(**params)
        # Query the tree within radius sqrt(ndim)
        try:
            inds = self._kdt.query_radius(xtarg.reshape(1, -1),
                                          r=np.sqrt(self.ndim))
            inds = inds[0]
        except(AttributeError):
            inds = self._kdt.query_ball_point(xtarg.reshape(1, -1),
                                              np.sqrt(self.ndim))
        return np.sort(inds)

    def linear_weights(self, knearest, **params):
        """Use ND-linear interpolation over the knearest neighbors.

        :param knearest:
            The indices of the ``vertices`` for which to calculate weights.

        :param params:
            The target parameter location, as keyword arguments.

        :returns wght:
            The weight for each vertex, computed as the volume of the hypercube
            formed by the target parameter and each vertex.  Vertices more than
            1 away from the target in any dimension are given a weight of zero.
        """
        xtarg = self.params_to_grid(**params)
        x = self.X[knearest, :]
        dx = xtarg - x
        # Fractional pixel weights
        wght = ((1 - dx) * (dx >= 0) + (1 + dx) * (dx < 0))
        # set weights to zero if model is more than a pixel away
        wght *= (dx > -1) * (dx < 1)
        # compute hyperarea for each model and return
        return wght.prod(axis=-1)

    def triangle_weights(self, knearest, **params):
        """Triangulate the k-nearest models, then use the barycenter of the
        enclosing simplex to interpolate.
        """
        inparams = np.array([params[p] for p in self.stellar_pars])
        dtri = Delaunay(self.model_points[knearest, :])
        triangle_ind = dtri.find_simplex(inparams)
        inds = dtri.simplices[triangle_ind, :]
        transform = dtri.transform[triangle_ind, :, :]
        Tinv = transform[:self.ndim, :]
        x_r = inparams - transform[self.ndim, :]
        bary = np.dot(Tinv, x_r)
        last = 1.0 - bary.sum()
        wghts = np.append(bary, last)
        oo = inds.argsort()
        return inds[oo], wghts[oo]


def gauss(x, mu, A, sigma):
    """
    Lay down mutiple gaussians on the x-axis.
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
