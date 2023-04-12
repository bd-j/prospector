# -*- coding: utf-8 -*-

import json
import numpy as np

from sedpy.observate import FilterSet
from sedpy.smoothing import smoothspec, smooth_fft

from ..likelihood.noise_model import NoiseModel


__all__ = ["Observation", "Spectrum", "Photometry", "Lines"
           "from_oldstyle", "from_serial", "obstypes"]


CKMS = 2.998e5

class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, type):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


class Observation:

    """Data to be predicted (and fit)

    Attributes
    ----------
    flux :
    uncertainty :
    mask :
    noise :
    """

    logify_spectrum = False
    alias = {}
    _meta = ["kind", "name"]
    _data = ["wavelength", "flux", "uncertainty", "mask"]

    def __init__(self,
                 flux=None,
                 uncertainty=None,
                 mask=slice(None),
                 noise=NoiseModel(),
                 name="ObsA",
                 **kwargs
                 ):

        self.flux = np.array(flux)
        self.uncertainty = np.array(uncertainty)
        self.mask = mask
        self.noise = noise
        self.name = name
        self.from_oldstyle(**kwargs)

    def __getitem__(self, item):
        """Dict-like interface for backwards compatibility
        """
        k = self.alias.get(item, item)
        return getattr(self, k)

    def get(self, item, default):
        try:
            return self[item]
        except(AttributeError):
            return default

    def from_oldstyle(self, **kwargs):
        """Take an old-style obs dict and use it to populate the relevant
        attributes.
        """
        for k, v in self.alias.items():
            if k in kwargs:
                setattr(self, v, kwargs[k])

    def rectify(self):
        """Make sure required attributes for fitting are present and have the
        appropriate sizes.  Also auto-masks non-finite data or negative
        uncertainties.
        """
        if self.flux is None:
            print(f"{self.__repr__} has no data")
            return

        assert self.wavelength.ndim == 1, "`wavelength` is not 1-d array"
        assert self.flux.ndim == 1, "flux is not a 1d array"
        assert self.uncertainty.ndim == 1, "uncertainty is not a 1d array"
        assert self.ndata > 0, "no wavelength points supplied!"
        assert self.uncertainty is not None, "No uncertainties."
        assert len(self.wavelength) == len(self.flux), "Flux array not same shape as wavelength."
        assert len(self.wavelength) == len(self.uncertainty), "Uncertainty array not same shape as wavelength."

        self._automask()

        assert self.ndof > 0, f"{self.__repr__()} has no valid data to fit: check the sign of the masks."
        assert hasattr(self, "noise")

    def _automask(self):
        # make mask array with automatic filters
        marr = np.zeros(self.ndata, dtype=bool)
        marr[self.mask] = True
        if self.flux is not None:
            self.mask = (marr &
                         (np.isfinite(self.flux)) &
                         (np.isfinite(self.uncertainty)) &
                         (self.uncertainty > 0))
        else:
            self.mask = marr

    def render(self, wavelength, spectrum):
        raise(NotImplementedError)

    @property
    def ndof(self):
        # TODO: cache this?
        return int(np.sum(np.ones(self.ndata)[self.mask]))

    @property
    def ndata(self):
        # TODO: cache this?
        if self.wavelength is None:
            return 0
        else:
            return len(self.wavelength)

    @property
    def wave_min(self):
        return np.min(self.wavelength)

    @property
    def wave_max(self):
        return np.max(self.wavelength)

    @property
    def metadata(self):
        meta = {m: getattr(self, m) for m in self._meta}
        if "filternames" in meta:
            meta["filters"] = ",".join(meta["filternames"])
        return meta

    def to_struct(self, data_dtype=np.float32):
        """Convert data to a structured array
        """
        self._automask()
        dtype = np.dtype([(c, data_dtype) for c in self._data])
        struct = np.zeros(self.ndata, dtype=dtype)
        for c in self._data:
            data = getattr(self, c)
            try:
                struct[c] = data
            except(ValueError):
                pass
        return struct

    def to_fits(self, filename=""):
        from astropy.io import fits
        hdus = fits.HDUList([fits.PrimaryHDU(),
                             fits.BinTableHDU(self.to_struct())])
        for hdu in hdus:
            hdu.header.update(self.metadata)
        if filename:
            hdus.writeto(filename, overwrite=True)
            hdus.close()

    def to_h5_dataset(self, handle):
        dset = handle.create_dataset(self.name, data=self.to_struct())
        dset.attrs.update(self.metadata)

    def to_json(self):
        obs = {m: getattr(self, m) for m in self._meta + self._data}
        serial = json.dumps(obs, cls=NumpyEncoder)
        return serial

    @property
    def to_nJy(self):
        return 1e9 * 3631


class Photometry(Observation):

    kind = "photometry"
    alias = dict(maggies="flux",
                 maggies_unc="uncertainty",
                 filters="filters",
                 phot_mask="mask")
    _meta = ["kind", "name", "filternames"]

    def __init__(self, filters=[], name="PhotA", **kwargs):
        """On Observation object that holds photometric data

        Parameters
        ----------
        filters : list of strings or list of `sedpy.observate.Filter` instances
            The names or instances of Filters to use

        flux : iterable of floats
            The flux through the filters, in units of maggies

        uncertainty : iterable of floats
            The uncertainty on the flux

        name : string, optional
            The name for this set of data
        """
        self.set_filters(filters)
        super(Photometry, self).__init__(name=name, **kwargs)

    def set_filters(self, filters):
        if (len(filters) == 0) or (filters is None):
            self.filters = filters
            self.filternames = []
            self.filterset = None
            return

        try:
            self.filternames = [f.name for f in filters]
        except(AttributeError):
            self.filternames = filters
        #if type(filters[0]) is str:
        #    self.filternames = filters
        #else:
        #    self.filternames = [f.name for f in filters]

        self.filterset = FilterSet(self.filternames)
        # filters on the gridded resolution
        self.filters = [f for f in self.filterset.filters]

    @property
    def wavelength(self):
        return np.array([f.wave_effective for f in self.filters])

    def to_oldstyle(self):
        obs = {}
        obs.update(vars(self))
        for k, v in self.alias.items():
            obs[k] = self[v]
            _ = obs.pop(v)
        #obs.update({k: self[v] for k, v in self.alias.items()})
        #_ = [obs.pop(k) for k in ["flux", "uncertainty", "mask"]]
        obs["phot_wave"] = self.wavelength
        return obs


class Spectrum(Observation):

    kind = "spectrum"
    alias = dict(spectrum="flux",
                 unc="uncertainty",
                 wavelength="wavelength",
                 mask="mask")

    data = ["wavelength", "flux", "uncertainty", "mask",
            "resolution", "calibration"]

    def __init__(self,
                 wavelength=None,
                 resolution=None,
                 calibration=None,
                 name="SpecA",
                 **kwargs):

        """
        Parameters
        ----------
        wavelength : iterable of floats
            The wavelength of each flux measurement, in vacuum AA

        flux : iterable of floats
            The flux at each wavelength, in units of maggies, same length as ``wavelength``

        uncertainty : iterable of floats
            The uncertainty on the flux

        resolution : (optional, default: None)
            Instrumental resolution at each wavelength point in units of km/s
            dispersion (:math:`= c \, \sigma_\lambda / \lambda = c \, \FWHM_\lambda / 2.355 / \lambda = c / (2.355 \, R_\lambda)`
            where :math:`c=2.998e5 {\rm km}/{\rm s}`

        :param calibration:
            not sure yet ....
        """
        super(Spectrum, self).__init__(name=name, **kwargs)
        self.wavelength = wavelength
        self.resolution = resolution
        self.calibration = calibration
        self.instrument_smoothing_parameters = dict(smoothtype="vel", fftsmooth=True)
        assert np.all(np.diff(self.wavelength) > 0)
        self.pad_wavelength_array()

    def pad_wavelength_array(self, lambda_pad=100):
        #wave_min = self.wave_min * (1 - np.arange(npad, 0, -1) * Kdelta[0] / ckms)
        low_pad = np.arange(lambda_pad, 1, (self.wavelength[0]-self.wavelength[1]))
        hi_pad = np.arange(1, lambda_pad, (self.wavelength[-1]-self.wavelength[-2]))
        wave_min = self.wave_min - low_pad
        wave_max = self.wave_max + hi_pad
        self.padded_wavelength = np.concatenate([wave_min, self.wavelength, wave_max])
        self.padded_resolution = np.interp(self.padded_wavelength, self.wavelength, self.resolution)
        self._unpadded_inds = slice(len(low_pad), -len(hi_pad))

    def smooth_lsf_fft(self, inwave, influx, outwave, sigma):
        dw = np.gradient(outwave)
        sigma_per_pixel = (dw / sigma)
        cdf = np.cumsum(sigma_per_pixel)
        cdf /= cdf.max()
        # check: do we need this?
        x_per_pixel = np.gradient(cdf)
        x_per_sigma = np.nanmedian(x_per_pixel / sigma_per_pixel)
        pix_per_sigma = 1
        N = pix_per_sigma / x_per_sigma
        nx = int(2**np.ceil(np.log2(N)))
        # now evenly sample in the x coordinate
        x = np.linspace(0, 1, nx)
        dx = 1.0 / nx
        lam = np.interp(x, cdf, outwave)
        newflux = np.interp(lam, inwave, influx)
        flux_conv = smooth_fft(dx, newflux, x_per_sigma)
        outflux = np.interp(outwave, lam, flux_conv)
        return outflux

    def instrumental_smoothing(self, wave_obs, influx, zred=0, libres=0):
        """Smooth a spectrum by the instrumental resolution, optionally
        accounting (in quadrature) the intrinsic library resolution.

        Parameters
        ----------
        obswave : ndarray of shape (N_pix_model,)
            Observed frame wavelengths, in units of AA for the model

        influx : ndarray of shape (N_pix_model,)
            Flux array

        libres : float or ndarray of shape (N_pix_model,)
            Library resolution in units of km/s (dispersion) to be subtracted from the smoothing kernel.
            This should be in the observed frame and *on the same wavelength grid as obs.wavelength*

        Returns
        -------
        outflux : ndarray of shape (ndata,)
            If instrument resolution is not None, this is the smoothed flux on
            the observed ``wavelength`` grid.  If resolution is None, this just
            passes ``influx`` right back again.
        """
        # interpolate library resolution onto the instrumental wavelength grid
        Klib = np.interp(self.padded_wavelength, wave_obs, libres)
        # quadrature difference of instrumental and library reolution
        Kdelta = np.sqrt(self.padded_resolution**2 - Klib**2)
        Kdelta_lambda = Kdelta / CKMS * self.padded_wavelength

        outspec_padded = self.smooth_lsf_fft(wave_obs,
                                             influx,
                                             self.padded_wavelength,
                                             Kdelta_lambda)
        if False:
            warr = [wave_min]
            while warr[-1] < wave_max:
                w = warr[-1]
                dv = np.interp(w, self.wavelength, Kdelta)
                warr.append((1 + dv / ckms) * w)
            warr = np.array(warr)
            flux_resampled = np.interp(warr, wave_obs, influx)
            np.convolve(flux_resampled, )

        return outspec_padded[self._unpadded_inds]

    def to_oldstyle(self):
        obs = {}
        obs.update(vars(self))
        for k, v in self.alias.items():
            obs[k] = self[v]
            _ = obs.pop(v)
        return obs


class Lines(Spectrum):

    kind = "lines"
    alias = dict(spectrum="flux",
                 unc="uncertainty",
                 wavelength="wavelength",
                 mask="mask",
                 line_inds="line_ind")

    data = ["wavelength", "flux", "uncertainty", "mask",
            "resolution", "calibration", "line_ind"]

    def __init__(self,
                 line_ind=None,
                 name="SpecA",
                 **kwargs):

        """
        Parameters
        ----------
        line_ind : iterable of int
            The index of the lines in the FSPS spectral line array.

        wavelength : iterable of floats
            The wavelength of each flux measurement, in vacuum AA

        flux : iterable of floats
            The flux at each wavelength, in units of maggies, same length as ``wavelength``

        uncertainty : iterable of floats
            The uncertainty on the flux

        resolution : (optional, default: None)
            Instrumental resolution at each wavelength point in units of km/s
            dispersion (:math:`= c \, \sigma_\lambda / \lambda = c \, \FWHM_\lambda / 2.355 / \lambda = c / (2.355 \, R_\lambda)`
            where :math:`c=2.998e5 {\rm km}/{\rm s}`

        :param calibration:
            not sure yet ....
        """
        super(Lines, self).__init__(name=name, **kwargs)
        assert (line_ind is not None), "You must identify the lines by their index in the FSPS emission line array"
        self.line_ind = np.array(line_ind).as_type(int)


obstypes = dict(photometry=Photometry,
                spectrum=Spectrum,
                lines=Lines)


def from_oldstyle(obs, **kwargs):
    """Convert from an oldstyle dictionary to a list of observations
    """
    spec, phot = Spectrum(**obs), Photometry(**obs)
    #phot.set_filters(phot.filters)
    #[o.rectify() for o in obslist]

    return [spec, phot]


def from_serial(arr, meta):
    adict = {a:arr[a] for a in arr.dtype.names}
    adict["name"] = meta.get("name", "")
    if 'filters' in meta:
        adict["filters"] = meta["filters"].split(",")
    obs = obstypes[meta["kind"]](**adict)
    #[setattr(obs, m, v) for m, v in meta.items()]
    return obs

