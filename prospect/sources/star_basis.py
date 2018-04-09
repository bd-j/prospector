from itertools import chain
import numpy as np
from numpy.polynomial.chebyshev import chebval
from scipy.spatial import Delaunay

from ..utils.smoothing import smoothspec
from .constants import lightspeed, lsun, jansky_cgs, to_cgs_at_10pc

try:
    from sklearn.neighbors import KDTree
except(ImportError):
    from scipy.spatial import cKDTree as KDTree

try:
    from sedpy.observate import getSED, vac2air, air2vac
except(ImportError):
    pass

    
__all__ = ["StarBasis", "BigStarBasis"]


# Useful constants
# value to go from L_sun to erg/s/cm^2 at 10pc
to_cgs = to_cgs_at_10pc

# for converting Kurucz spectral units
log4pi = np.log10(4 * np.pi)
log_rsun_cgs = np.log10(6.955) + 10
log_lsun_cgs = np.log10(lsun)
log_SB_cgs = np.log10(5.6704e-5)
log_SB_solar = log_SB_cgs + 2 * log_rsun_cgs - log_lsun_cgs


class StarBasis(object):

    _spectra = None

    def __init__(self, libname='ckc14_deimos.h5', verbose=False,
                 n_neighbors=0, log_interp=True, logify_Z=False,
                 use_params=None, rescale_libparams=False, in_memory=True,
                 **kwargs):
        """An object which holds the stellar spectral library, performs
        interpolations of that library, and has methods to return attenuated,
        normalized, smoothed stellar spectra.  The interpolations are performed
        using barycenter coordinates of the enclosing simplex found from the
        Delauynay triangulation.  This is not tractable for large dimension
        (see BigStarBasis for that case).

        :param libname:
            Path to the hdf5 file to use for the spectral library.

        :param verbose:
            If True, print information about the parameters used when a point
            is outside the convex hull.

        :param n_neighbors: (default:0)
            Number of nearest neighbors to use when requested parameters are
            outside the convex hull of the library prameters.  If ``0`` then a
            ValueError is raised instead of the nearest spectrum.  If greater
            than 1 then the neighbors are combined using inverse distance
            weights.

        :param log_interp: (default:True)
            Switch to interpolate in log(flux) instead of linear flux.

        :param use_params:
            Sequence of strings. If given, only use the listed parameters
            (which must be present in the `_libparams` structure) to build the
            grid and construct spectra.  Otherwise all fields of `_libparams`
            will be used.

        :param rescale_libparams: (default: False)
            If True, rescale the parameters to the unit cube before generating
            the triangulation (and kd-tree).  Note that the `param_vector`
            method will also rescale the input parameters in this case.  This
            can help for nearest neighbor lookup and in the triangulation based
            weights when your variables have very different scales, assuming
            that the ranges give a reasonable relative distance metric.

        :param in_memory: (default: True)
            Switch to keep the spectral library in memory or access it through
            the h5py File object.  Note if the latter, then zeroed spectra are
            *not* filtered out.
        """
        # Cache initialization variables
        self.verbose = verbose
        self.logarithmic = log_interp
        self.logify_Z = logify_Z
        self._in_memory = in_memory
        self._libname = libname
        self.n_neighbors = n_neighbors
        self._rescale = rescale_libparams

        # Load the library
        self.load_lib(libname)

        # Do some important bookkeeping
        if use_params is None:
            self.stellar_pars = self._libparams.dtype.names
        else:
            self.stellar_pars = tuple(use_params)
        self.ndim = len(self.stellar_pars)

        # Build the triangulation and kdtree (after rescaling)
        if self._rescale:
            ranges = [[self._libparams[d].min(), self._libparams[d].max()]
                      for d in self.stellar_pars]
            self.parameter_range = np.array(ranges).T
        self.triangulate()
        try:
            self.build_kdtree()
        except NameError:
            pass

        self.params = {}

    def load_lib(self, libname='', driver=None):
        """Read a CKC library which has been pre-convolved to be close to your
        resolution.  This library should be stored as an HDF5 file, with the
        datasets ``wavelengths``, ``parameters`` and ``spectra``.  These are
        ndarrays of shape (nwave,), (nmodels,), and (nmodels, nwave)
        respecitvely.  The ``parameters`` array is a structured array.  Spectra
        with no fluxes > 1e-32 are removed from the library if the librarty is
        kept in memory.
        """
        import h5py
        f = h5py.File(libname, "r", driver=driver)
        self._wave = np.array(f['wavelengths'])
        self._libparams = np.array(f['parameters'])

        if self._in_memory:
            self._spectra = np.array(f['spectra'])
            f.close()
            # Filter library so that only existing spectra are included
            maxf = np.max(self._spectra, axis=1)
            good = maxf > 1e-32
            self._libparams = self._libparams[good]
            self._spectra = self._spectra[good, :]
        else:
            self._spectra = f['spectra']

        if self.logify_Z:
            from numpy.lib import recfunctions as rfn
            self._libparams['Z'] = np.log10(self._libparams['Z'])
            rfn.rename_fields(self._libparams, {'Z': 'logZ'})

    def update(self, **kwargs):
        """Update the `params` dictionary, turning length 1 arrays into scalars
        and pull out functions from length one arrays
        """
        for k, val in list(kwargs.items()):
            v = np.atleast_1d(val)
            try:
                if (len(v) == 1) and callable(v[0]):
                    self.params[k] = v[0]
                else:
                    self.params[k] = np.squeeze(v)
            except(KeyError):
                pass

    def get_spectrum(self, outwave=None, filters=None, peraa=False, **kwargs):
        """Return an attenuated, smoothed, distance dimmed stellar spectrum and SED.

        :returns spec:
            The spectrum on the outwave grid (assumed in air), in AB maggies.
            If peraa is True then the spectrum is erg/s/cm^2/AA.

        :returns phot:
            Observed frame photometry in units of AB maggies.  If ``lumdist``
            is not present in the parameters then these are absolute maggies,
            otherwise they are apparent.

        :returns x:
            A blob of extra quantities (e.g. mass, uncertainty)
        """
        self.update(**kwargs)

        # star spectrum (in Lsun/Hz)
        wave, spec, unc = self.get_star_spectrum(**self.params)
        spec *= self.normalize()

        # dust
        if 'dust_curve' in self.params:
            att = self.params['dust_curve'](self._wave, **self.params)
            spec *= np.exp(-att)

        # Redshifting + Wavelength solution.  We also convert to in-air.
        a = 1 + self.params.get('zred', 0)
        b = 0.0

        if 'wavecal_coeffs' in self.params:
            x = wave - wave.min()
            x = 2.0 * (x / x.max()) - 1.0
            c = np.insert(self.params['wavecal_coeffs'], 0, 0)
            # assume coeeficients give shifts in km/s
            b = chebval(x, c) / (lightspeed*1e-13)

        wa, sa = vac2air(wave) * (a + b), spec * a
        if outwave is None:
            outwave = wa

        # Broadening, interpolation onto output wavelength grid
        if 'sigma_smooth' in self.params:
            smspec = self.smoothspec(wa, sa, self.params['sigma_smooth'],
                                     outwave=outwave, **self.params)
        elif outwave is not wa:
            smspec = np.interp(outwave, wa, sa, left=0, right=0)
        else:
            smspec = sa

        # Photometry (observed frame absolute maggies)
        if filters is not None:
            mags = getSED(wa, sa * lightspeed / wa**2 * to_cgs, filters)
            phot = np.atleast_1d(10**(-0.4 * mags))
        else:
            phot = 0.0

        # Distance dimming.  Default to 10pc distance (i.e. absolute)
        dfactor = (self.params.get('lumdist', 1e-5) * 1e5)**2
        if peraa:
            # spectrum will be in erg/s/cm^2/AA
            smspec *= to_cgs / dfactor * lightspeed / outwave**2
        else:
            # Spectrum will be in maggies
            smspec *= to_cgs / dfactor / (3631*jansky_cgs)

        # Convert from absolute maggies to apparent maggies
        phot /= dfactor

        return smspec, phot, None

    def get_star_spectrum(self, **kwargs):
        """Given stellar parameters, obtain an interpolated spectrum at those
        parameters.

        :param **kwargs:
            Keyword arguments must include values for the parameters listed in
            ``stellar_pars``.

        :returns wave:
            The wavelengths at which the spectrum is defined.

        :returns spec:
            The spectrum interpolated to the requested parameters.  This has
            the same units as the supplied library spectra.

        :returns unc:
            The uncertainty spectrum, where the uncertainty is due to
            interpolation error.  Curently unimplemented (i.e. it is a None
            type object).
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
        """Use either `logr` or `logl` to normalize the spectrum.  Both should
        be in solar units.  `logr` is checked first.  If neither is present
        then 1.0 is returned.

        :returns norm:
            Factor by which the CKC spectrum should be multiplied to get units
            of L_sun/Hz.  This assumes the native library spectrum is in units
            of erg/s/cm^2/Hz/sr.
        """
        if 'logr' in self.params:
            twologr = 2. * (self.params['logr'] + log_rsun_cgs)
        elif 'logl' in self.params:
            twologr = ((self.params['logl'] + log_lsun_cgs) -
                       4 * self.params['logt'] - log_SB_cgs - log4pi)
        else:
            return 1.0

        norm = 10**(twologr + 2 * log4pi - log_lsun_cgs)
        return norm

    def weights(self, **kwargs):
        """Delauynay weighting.  Return indices of the models forming the
        enclosing simplex, as well as the barycentric coordinates of the point
        within this simplex to use as weights.  If point is outside the convex
        hull then fallback to nearest neighbor unless ``n_neighbors`` is 0.
        """
        inparams = np.squeeze(self.param_vector(**kwargs))
        triangle_ind = self._dtri.find_simplex(inparams)
        if triangle_ind == -1:
            self.edge_flag = True
            if self.n_neighbors == 0:
                pstring = ', '.join(self.ndim * ['{}={}'])
                pstring = pstring.format(*chain(*zip(self.stellar_pars, inparams)))
                raise ValueError("Requested spectrum ({}) outside convex hull,"
                                 " and nearest neighbor interpolation turned "
                                 "off.".format(*pstring))
            ind, wght = self.weights_knn(inparams, k=self.n_neighbors)
            if self.verbose:
                print("Parameters {0} outside model convex hull. "
                      "Using model index {1} instead. ".format(inparams, ind))
            return ind, wght

        inds = self._dtri.simplices[triangle_ind, :]
        transform = self._dtri.transform[triangle_ind, :, :]
        Tinv = transform[:self.ndim, :]
        x_r = inparams - transform[self.ndim, :]
        bary = np.dot(Tinv, x_r)
        last = np.clip(1.0 - bary.sum(), 0.0, 1.0)
        wghts = np.append(bary, last)
        oo = inds.argsort()
        return inds[oo], wghts[oo]

    def rescale_params(self, points):
        """Rescale the given parameters to the unit cube, if the ``_rescale`` attribute is ``True``

        :param points:
            An array of parameter values, of shape (npoint, ndim)

        :returns x:
            An array of parameter values rescaled to the unit cube, ndarray of
            shape (npoint, ndim)
        """
        if self._rescale:
            x = np.atleast_2d(points)
            x = (x - self.parameter_range[0, :]) / np.diff(self.parameter_range, axis=0)
            return np.squeeze(x)
        else:
            return points

    def triangulate(self):
        """Build the Delauynay Triangulation of the model library.
        """
        # slow.  should use a view based method
        model_points = np.array([list(self._libparams[d])
                                 for d in self.stellar_pars]).T
        self._dtri = Delaunay(self.rescale_params(model_points))

    def build_kdtree(self):
        """Build the kdtree of the model points.
        """
        # slow.  should use a view based method
        model_points = np.array([list(self._libparams[d])
                                 for d in self.stellar_pars])
        self._kdt = KDTree(self.rescale_params(model_points.T))

    def weights_knn(self, target_points, k=1):
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
            dists, inds = self._kdt.query(np.atleast_2d(target_points), k=k,
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
        return self.rescale_params(np.array(pvec))

    @property
    def wavelengths(self):
        return self._wave


class BigStarBasis(StarBasis):

    def __init__(self, libname='', verbose=False, log_interp=True,
                 n_neighbors=0,  driver=None, in_memory=False,
                 use_params=None, strictness=0.0, **kwargs):
        """An object which holds the stellar spectral library, performs linear
        interpolations of that library, and has methods to return attenuated,
        normalized, smoothed stellar spoectra.

        This object is set up to work with large grids, so the models file is
        kept open for access from disk.  scikits-learn or scipy kd-trees are
        required for model access.  Ideally the grid should be regular (though
        the spacings need not be equal along a given dimension).

        :param libname:
            Path to the hdf5 file to use for the spectral library.

        :param n_neighbors: (default:0)
            Number of nearest neighbors to use when requested parameters are
            outside the convex hull of the library prameters.  If ``0`` then a
            ValueError is raised instead of the nearest spectrum.  Does not
            work, currently.

        :param verbose:
            If True, print information about the parameters used when a point
            is outside the convex hull

        :param log_interp: (default: True)
            Interpolate in log(flux) instead of flux.

        :param in_memory: (default: False)
            Switch to determine whether the grid is loaded in memory or read
            from disk each time a model is constructed (like you'd want for
            very large grids).

        :param use_params:
            Sequence of strings. If given, only use the listed parameters
            (which must be present in the `_libparams` structure) to build the
            grid and construct spectra.  Otherwise all fields of `_libparams`
            will be used.

        :param strictness: (default: 0.0)
            Float from 0.0 to 1.0 that gives the fraction of a unit hypercube
            that is required for a parameter position to be accepted.  That is,
            if the weights of the enclosing vertices sum to less than this
            number, raise an error.
        """
        self.verbose = verbose
        self.logarithmic = log_interp
        self._libname = libname
        self.n_neighbors = n_neighbors
        self._in_memory = in_memory
        self._strictness = strictness

        self.load_lib(libname, driver=driver)
        # Do some important bookkeeping
        if use_params is None:
            self.stellar_pars = self._libparams.dtype.names
        else:
            self.stellar_pars = tuple(use_params)
        self.ndim = len(self.stellar_pars)
        self.lib_as_grid()
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
        if self._in_memory:
            self._spectra = np.array(f['spectra'])
            f.close()
        else:
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
        if wghts.sum() <= self._strictness:
            raise ValueError("Something is wrong with the weights")
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
