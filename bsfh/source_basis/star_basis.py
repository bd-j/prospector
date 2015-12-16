from itertools import chain
import numpy as np
from scipy.spatial import Delaunay
from ..utils.smoothing import smoothspec
from sedpy.observate import getSED, vac2air, air2vac

try:
    from sklearn.neighbors import KDTree
except(ImportError):
    from scipy.spatial import KDTree

__all__ = ["StarBasis", "BigStarBasis"]


# Useful constants
lsun = 3.846e33
pc = 3.085677581467192e18  # in cm
lightspeed = 2.998e18  # AA/s
# value to go from L_sun/AA to erg/s/cm^2/AA at 10pc
to_cgs = lsun/(4.0 * np.pi * (pc*10)**2)
# for converting Kurucz spectral units
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
            Path to the hdf5 file to use for the spectral library.

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
        """Return an attenuated, smoothed, distance dimmed stellar spectrum and SED.
        
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
        inds, wghts = self.weights(**kwargs)
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

    def weights(self, **kwargs):
        """Delauynay weighting.  Return indices of the models forming the
        enclosing simplex, as well as the barycentric coordinates of the point
        within this simplex to use as weights.  If point is outside the convex
        hull then fallback to nearest neighbor unless ``n_neighbors`` is 0.
        """
        inparams = np.squeeze(self.param_vector(**kwargs))
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
        inds, wghts = self.weights(**kwargs)
        if self.logarithmic:
            spec = np.exp(np.dot(wghts, np.log(self._spectra[inds, :])))
        else:
            spec = np.dot(wghts, self._spectra[inds, :])
        spec_unc = None
        return self._wave, spec, spec_unc

    def weights(self, **params):
        inds = self.knearest_inds(**params)
        wghts = self.linear_weights(inds, **params)
        # if wghts.sum() < 1.0:
        #     raise ValueError("Something is wrong with the weights")
        good = wghts > 0
        # if good.sum() < 2**self.ndim:
        #     raise ValueError("Did not find all vertices of the hypercube, "
        #                      "or there is no enclosing hypercube in the library.")
        inds = inds[good]
        wghts = wghts[good]
        wghts /= wghts.sum()
        return inds, wghts

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
        self._kdt = KDTree(self.X)  # , metric='euclidean')

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
            pstring = "{0}: min={2} max={3} targ={1}\n"
            s = [pstring.format(p, targ[p], *self.gridpoints[p][[0, -1]])
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
        except(AttributeError):
            inds = self._kdt.query_ball_point(xtarg.reshape(1, -1),
                                              np.sqrt(self.ndim))
        return np.sort(inds[0])

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
