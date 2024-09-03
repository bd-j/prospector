# -*- coding: utf-8 -*-

import json
import numpy as np

from sedpy.observate import FilterSet
from sedpy.smoothing import smooth_fft
from sedpy.observate import rebin

from ..likelihood.noise_model import NoiseModel


__all__ = ["Observation",
           "Spectrum", "Photometry", "Lines",
           "UndersampledSpectrum", "IntrinsicSpectrum",
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

    _kind = "observation"
    logify_spectrum = False
    alias = {}
    _meta = ("kind", "name")
    _data = ("wavelength", "flux", "uncertainty", "mask")

    def __init__(self,
                 flux=None,
                 uncertainty=None,
                 mask=slice(None),
                 noise=NoiseModel(),
                 name=None,
                 **kwargs
                 ):

        self.flux = np.array(flux)
        self.uncertainty = np.array(uncertainty)
        self.mask = mask
        self.noise = noise
        self.from_oldstyle(**kwargs)
        if name is None:
            addr = f"{id(self):04x}"
            self.name = f"{self.kind[:5]}-{addr[:6]}"
        else:
            self.name = name

    def __str__(self):
        return f"{self.kind} ({self.name})"

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
        n = self.__repr__
        if self.flux is None:
            print(f"{n} has no data")
            return

        assert self.flux.ndim == 1, f"{n}: flux is not a 1d array"
        assert self.uncertainty.ndim == 1, f"{n}: uncertainty is not a 1d array"
        assert self.ndata > 0, f"{n} no data supplied!"
        assert self.uncertainty is not None, f"{n} No uncertainties."
        assert len(self.flux) == len(self.uncertainty), f"{n}: flux and uncertainty of different length"
        if self.wavelength is not None:
            assert self.wavelength.ndim == 1, f"{n}: `wavelength` is not 1-d array"
            assert len(self.wavelength) == len(self.flux), f"{n}: Wavelength array not same shape as flux array"

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
    def kind(self):
        # make 'kind' private
        return self._kind

    @property
    def ndof(self):
        # TODO: cache this?
        return int(np.sum(np.ones(self.ndata)[self.mask]))

    @property
    def ndata(self):
        # TODO: cache this?
        if self.flux is None:
            return 0
        else:
            return len(self.flux)

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
        cols = []
        for c in self._data:
            dat = getattr(self, c)
            if (dat is None):
                continue
            if (len(dat) != self.ndata):
                continue
                #raise ValueError(f"The {c} attribute of the {self.name} observation has the wrong length ({len(dat)} instead of {self.ndata})")
            cols += [(c, dat.dtype)]
        dtype = np.dtype(cols)
        struct = np.zeros(self.ndata, dtype=dtype)
        for c in dtype.names:
            data = getattr(self, c)
            if c is not None:
                struct[c] = data
            #except(ValueError):
            #    pass
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

    def to_oldstyle(self):
        obs = {}
        obs.update(vars(self))
        for k, v in self.alias.items():
            obs[k] = self[v]
            _ = obs.pop(v)
        return obs

    @property
    def maggies_to_nJy(self):
        return 1e9 * 3631


class Photometry(Observation):

    _kind = "photometry"
    alias = dict(maggies="flux",
                 maggies_unc="uncertainty",
                 filters="filters",
                 phot_mask="mask")
    _meta =  ("kind", "name", "filternames")

    def __init__(self, filters=[],
                 name=None,
                 **kwargs):
        """On Observation object that holds photometric data

        Parameters
        ----------
        filters : list of strings or list of `sedpy.observate.Filter` instances
            The names or instances of Filters to use

        flux : iterable of floats
            The flux through the filters, in units of maggies.

        uncertainty : iterable of floats
            The uncertainty on the flux

        name : string, optional
            The name for this set of data
        """
        self.set_filters(filters)
        super(Photometry, self).__init__(name=name, **kwargs)

    def set_filters(self, filters):

        # TODO: Make this less convoluted
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
        obs = super(Photometry, self).to_oldstyle()
        obs["phot_wave"] = self.wavelength
        return obs


class Spectrum(Observation):

    _kind = "spectrum"
    alias = dict(spectrum="flux",
                 unc="uncertainty",
                 wavelength="wavelength",
                 mask="mask")

    _meta = ("kind", "name", "lambda_pad")
    _data = ("wavelength", "flux", "uncertainty", "mask",
             "resolution", "response")

    def __init__(self,
                 wavelength=None,
                 resolution=None,
                 response=None,
                 name=None,
                 lambda_pad=100,
                 **kwargs):

        """
        Parameters
        ----------
        wavelength : iterable of floats
            The wavelength of each flux measurement, in vacuum AA

        flux : iterable of floats
            The flux at each wavelength, in units of maggies, same length as
            ``wavelength``

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
        self.lambda_pad = lambda_pad
        self.resolution = resolution
        self.response = response
        self.instrument_smoothing_parameters = dict(smoothtype="vel", fftsmooth=True)
        self.wavelength = np.atleast_1d(wavelength)

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, wave):
        self._wavelength = wave
        if self._wavelength is not None:
            assert np.all(np.diff(self._wavelength) > 0)
            self.pad_wavelength_array()

    def pad_wavelength_array(self):
        """Pad the wavelength and, if present, resolution arrays so that FFTs
        can be used on the models without edge effects.
        """
        if self.wavelength is None:
            return

        low_pad = np.arange(self.lambda_pad, 1, (self.wavelength[0]-self.wavelength[1]))
        hi_pad = np.arange(1, self.lambda_pad, (self.wavelength[-1]-self.wavelength[-2]))
        wave_min = self.wave_min - low_pad
        wave_max = self.wave_max + hi_pad
        self.padded_wavelength = np.concatenate([wave_min, self.wavelength, wave_max])
        self._unpadded_inds = slice(len(low_pad), -len(hi_pad))
        if self.resolution is not None:
            self.padded_resolution = np.interp(self.padded_wavelength, self.wavelength, self.resolution)

    def _smooth_lsf_fft(self, inwave, influx, outwave, sigma):

        # construct cdf of 'resolution elements' as a fn of wavelength
        dw = np.gradient(outwave)
        sigma_per_pixel = (dw / sigma)
        cdf = np.cumsum(sigma_per_pixel)
        cdf /= cdf.max()

        # Get number of resolution elements: is this the best way to do this?
        # Can't we just use unnormalized cdf above
        x_per_pixel = np.gradient(cdf)
        x_per_sigma = np.nanmedian(x_per_pixel / sigma_per_pixel)
        pix_per_sigma = 1
        N = pix_per_sigma / x_per_sigma
        nx = int(2**np.ceil(np.log2(N)))

        # now evenly sample in the x coordinate
        x = np.linspace(cdf[0], 1, nx)
        dx = (1.0 - cdf[0]) / nx
        # convert x back to wavelength, and get model at these wavelengths
        lam = np.interp(x, cdf, outwave)
        newflux = np.interp(lam, inwave, influx)

        # smooth flux sampled at constant resolution
        # could replace this with np.conv()
        flux_conv = smooth_fft(dx, newflux, x_per_sigma)

        # sample at wavelengths of pixels
        outflux = self._pixelize(outwave, lam, flux_conv)
        return outflux

    def _pixelize(self, outwave, inwave, influx):
        # could do this with an FFT in pixel space using a sinc
        return np.interp(outwave, inwave, influx)

    def instrumental_smoothing(self, model_wave_obsframe, model_flux, zred=0, libres=0):
        """Smooth a spectrum by the instrumental resolution, optionally
        accounting (in quadrature) the intrinsic library resolution.

        Parameters
        ----------
        model_wave_obsframe : ndarray of shape (N_pix_model,)
            Observed frame wavelengths, in units of AA for the *model*

        model_flux : ndarray of shape (N_pix_model,)
            Flux array corresponding to the observed frame wavelengths

        libres : float or ndarray of shape (N_pix_model,)
            Library resolution in units of km/s (dispersion) to be subtracted
            from the smoothing kernel. This should be in the observed frame and
            on the same wavelength grid as obswave

        Returns
        -------
        outflux : ndarray of shape (ndata,)
            If instrument resolution is not None, this is the smoothed flux on
            the observed ``wavelength`` grid.  If wavelength is None, this just
            passes ``influx`` right back again.  If ``resolution`` is None then
            ``influx`` is simply interpolated onto the wavelength grid
        """
        if self.wavelength is None:
            return model_flux
        if self.resolution is None:
            return np.interp(self.wavelength, model_wave_obsframe, model_flux)

        # interpolate library resolution onto the instrumental wavelength grid
        Klib = np.interp(self.padded_wavelength, model_wave_obsframe, libres)
        assert np.all(self.padded_resolution >= Klib), "data higher resolution than library"

        # quadrature difference of instrumental and library resolution
        Kdelta = np.sqrt(self.padded_resolution**2 - Klib**2)
        Kdelta_lambda = Kdelta / CKMS * self.padded_wavelength

        # Smooth by the difference kernel
        outspec_padded = self._smooth_lsf_fft(model_wave_obsframe,
                                              model_flux,
                                              self.padded_wavelength,
                                              Kdelta_lambda)

        return outspec_padded[self._unpadded_inds]

    def compute_response(self, **extras):
        if self.response is not None:
            return self.response
        else:
            return 1.0


class Lines(Observation):

    _kind = "lines"
    alias = dict(spectrum="flux",
                 unc="uncertainty",
                 wavelength="wavelength",
                 mask="mask",
                 line_inds="line_ind")

    _meta = ("name", "kind")
    _data = ("wavelength", "flux", "uncertainty", "mask",
             "line_ind")

    def __init__(self,
                 line_ind=None,
                 line_names=None,
                 wavelength=None,
                 name=None,
                 **kwargs):

        """
        Parameters
        ----------
        line_ind : iterable of int
            The index of the lines in the FSPS spectral line array.

        wavelength : iterable of floats
            The wavelength of each flux measurement, in vacuum AA

        flux : iterable of floats
            The flux at each wavelength, in units of erg/s/cm^2, same length as
            ``wavelength``

        uncertainty : iterable of floats
            The uncertainty on the flux

        resolution : (optional, default: None)
            Instrumental resolution at each wavelength point in units of km/s
            dispersion (:math:`= c \, \sigma_\lambda / \lambda = c \, \FWHM_\lambda / 2.355 / \lambda = c / (2.355 \, R_\lambda)`
            where :math:`c=2.998e5 {\rm km}/{\rm s}`

        line_ind : iterable of string, optional
            Names of the lines.

        :param calibration:
            not sure yet ....
        """
        super(Lines, self).__init__(name=name, **kwargs)
        assert (line_ind is not None), "You must identify the lines by their index in the FSPS emission line array"
        self.line_ind = np.array(line_ind).astype(int)
        self.line_names = line_names

        if wavelength is not None:
            self._wavelength = np.atleast_1d(wavelength)
        else:
            self._wavelength = None

    @property
    def wavelength(self):
        return self._wavelength


class UndersampledSpectrum(Spectrum):
    """
    As for Spectrum, but account for pixelization effects when pixels
    undersample the instrumental LSF.
    """
    #TODO: Implement as a convolution with a square kernel (or sinc in frequency space)

    def _pixelize(self, outwave, inwave, influx):
        return rebin(outwave, inwave, influx)


class IntrinsicSpectrum(Spectrum):

    """
    As for Spectrum, but predictions for this object type will not include
    polynomial fitting or fitting of the emission line strengths (previously fit
    and cached emission luminosities will be used.)
    """

    _kind = "intrinsic"


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
    # TODO: This does not account for composite or special classes, or include
    # noise models
    kind = obstypes[meta.pop("kind")]

    adict = {a:arr[a] for a in arr.dtype.names}
    adict["name"] = meta.pop("name", "")
    if 'filters' in meta:
        adict["filters"] = meta.pop("filters").split(",")

    obs = kind(**adict)

    # set other metadata as attributes? No, needs to be during instantiation
    #for k, v in meta.items():
    #    if k in kind._meta:
    #        setattr(obs, k, v)

    return obs


def wave_to_x(wavelength=None, mask=slice(None), **extras):
    """Map unmasked wavelengths to the interval -1, 1
            masked wavelengths may have x>1, x<-1
    """
    x = wavelength - (wavelength[mask]).min()
    x = 2.0 * (x / (x[mask]).max()) - 1.0
    return x