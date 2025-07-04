# -*- coding: utf-8 -*-

import json
import numpy as np

from numpy.polynomial.chebyshev import chebval, chebvander
from scipy.interpolate import splrep, BSpline
from scipy.signal import medfilt

from sedpy.observate import FilterSet
from sedpy.smoothing import smooth_fft
from sedpy.observate import rebin

from ..likelihood.noise_model import NoiseModel


__all__ = ["Observation",
           "Spectrum", "Photometry", "Lines",
           "UndersampledSpectrum", "IntrinsicSpectrum",
           "SplineOptCal", "PolyOptCal",
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

        # TODO: clean up this logic
        # _flux_array = False
        # try:
        #     len(self.flux)
        #     _flux_array = True
        # except:
        #     _flux_array = False

        # if _flux_array:
        if self.flux is None:
            print(f"{n} has no data")
            return
        # else:
        #     if self.flux == None:
        #         print(f"{n} has no data")
        #         self.flux = None
        #         self.wavelength = None
        #         return

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
        if self.mask is not None:
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
        if wavelength is not None:
            self.wavelength = np.atleast_1d(wavelength)
        else:
            self.wavelength = None

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


class PolyOptCal:

    """A mixin class that allows for optimization of a Chebyshev response
    function given a model spectrum.
    """

    def __init__(self, *args,
                 polynomial_order=0,
                 polynomial_regularization=0,
                 median_polynomial=0,
                 **kwargs):
        super(PolyOptCal, self).__init__(*args, **kwargs)
        self.polynomial_order = polynomial_order
        self.polynomial_regularization = polynomial_regularization
        self.median_polynomial = median_polynomial

    def _available_parameters(self):
        # These should both be attached to the Observation instance as attributes
        pars = [("polynomial_order", "order of the polynomial to fit"),
                ("polynomial_regularization", "vector of length `polyorder` providing regularization for each polynomial term"),
                ("median_polynomial", "if > 0, median smooth with a kernel of width order/range/median_polynomial before fitting")
                ]

        return pars

    def compute_response(self, spec=None, extra_mask=True, **kwargs):
        """Implements a Chebyshev polynomial response function model. This uses
        least-squares to find the maximum-likelihood Chebyshev polynomial of a
        certain order describing the ratio of the observed spectrum to the model
        spectrum. If emission lines are being marginalized out, they should be
        excluded from the least-squares fit using the ``extra_mask`` keyword

        Parameters
        ----------
        spec : ndarray of shape (Spectrum().ndata,)
            The model spectrum on the observed wavelength grid

        extra_mask : ndarray of Booleans of shape (Spectrum().ndata,)
            The extra mask to be applied.  True=use data, False=mask data

        Returns
        -------
        response : ndarray of shape (nwave,)
           A polynomial given by :math:`\sum_{m=0}^M a_{m} * T_m(x)`.
        """

        order = self.polynomial_order
        reg = self.polynomial_regularization
        mask = self.mask & extra_mask
        assert (self.mask.sum() > order), f"Not enough points to constrain polynomial of order {order}"

        polyopt = (order > 0)
        if (not polyopt):
            print("no polynomial")
            self.response = np.ones_like(self.wavelength)
            return self.response

        x = wave_to_x(self.wavelength, mask)
        y = (self.flux / spec - 1.0)[mask]
        yerr = (self.uncertainty / spec)[mask]
        yvar = yerr**2

        if self.median_polynomial > 0:
            kernel_factor = self.median_polynomial
            knl = int((x.max() - x.min()) / order / kernel_factor)
            knl += int((knl % 2) == 0)
            y = medfilt(y, knl)

        Afull = chebvander(x, order)
        A = Afull[mask, :]
        ATA = np.dot(A.T, A / yvar[:, None])
        if np.any(reg > 0):
            ATA += reg**2 * np.eye(order+1)
        c = np.linalg.solve(ATA, np.dot(A.T, y / yvar))

        poly = np.dot(Afull, c)

        self._chebyshev_coefficients = c
        self.response = poly + 1.0
        return self.response


class SplineOptCal:

    """A mixin class that allows for optimization of a Chebyshev response
    function given a model spectrum.
    """


    def __init__(self, *args,
                 spline_knot_wave=None,
                 spline_knot_spacing=None,
                 spline_knot_n=None,
                 **kwargs):
        super(SplineOptCal, self).__init__(*args, **kwargs)

        self.params = {}
        if spline_knot_wave is not None:
            self.params["spline_knot_wave"] = spline_knot_wave
        elif spline_knot_spacing is not None:
            self.params["spline_knot_spacing"] = spline_knot_spacing
        elif spline_knot_n is not None:
            self.params["spline_knot_n"] = spline_knot_n

        # build and cache the knots
        w = self.wavelength[self.mask]
        (wlo, whi) = w.min(), w.min()
        self.wave_x = 2.0 * (self.wavelength - wlo) / (whi - wlo) - 1.0
        self.knots_x = self.make_knots(wlo, whi, as_x=True, **self.params)


    def _available_parameters(self):
        pars = [("spline_knot_wave", "vector of wavelengths for the location of the spline knots"),
                ("spline_knot_spacing", "spacing between knots, in units of wavelength"),
                ("spline_knot_n", "number of interior knots between minimum and maximum unmasked wavelength")
                ]

        return pars

    def compute_response(self, spec=None, extra_mask=True, **extras):
        """Implements a spline response function model.  This fits a cubic
        spline with determined knot locations to the ratio of the observed
        spectrum to the model spectrum.  If emission lines are being
        marginalized out, they are excluded from the least-squares fit.

        The knot locations must be specified as model parameters, either
        explicitly or via a number of knots or knot spacing (in angstroms)
        """
        mask = self.mask & extra_mask


        splineopt = True
        if ~splineopt:
            self.response = np.ones_like(self.wavelength)
            return self.response

        y = (self.flux / spec - 1.0)[mask]
        yerr = (self.uncertainty / spec)[mask] # HACK
        tck = splrep(self.wave_x[mask], y[mask], w=1/yerr[mask], k=3, task=-1, t=self.knots_x)
        self._spline_coeffs = tck
        spl = BSpline(*tck)
        spline = spl(self.wave_x)

        self.response = (1.0 + spline)
        return self.response

    def make_knots(self, wlo, whi, as_x=True, **params):
        """Can we move this to instantiation?
        """
        if "spline_knot_wave" in params:
            knots = np.squeeze(params["spline_knot_wave"])
        elif "spline_knot_spacing" in params:
            s = np.squeeze(params["spline_knot_spacing"])
            # we drop the start so knots are interior
            knots = np.arange(wlo, whi, s)[1:]
        elif "spline_knot_n" in params:
            n = np.squeeze(params["spline_knot_n"])
            # we need to drop the endpoints so knots are interior
            knots = np.linspace(wlo, whi, n)[1:-1]
        else:
            raise KeyError("No valid spline knot specification in self.params")

        if as_x:
            knots = 2.0 * (knots - wlo) / (whi - wlo) - 1.0

        return knots


class PolyFitCal:

    """This is a mixin class that generates the
    multiplicative response vector as a Chebyshev polynomial described by the
    ``poly_param_name`` parameter of the model, which may be free (fittable)
    """

    def __init__(self, *args, poly_param_name=None, **kwargs):
        super(SplineOptCal, self).__init(*args, **kwargs)
        self.poly_param_name = poly_param_name

    def _available_parameters(self):
        pars = [(self.poly_param_name, "vector of polynomial chabyshev coefficients")]

        return pars

    def compute_response(self, **kwargs):
        """Implements a Chebyshev polynomial calibration model.  This only
        occurs if ``"poly_coeffs"`` is present in the :py:attr:`params`
        dictionary, otherwise the value of ``params["spec_norm"]`` is returned.

        :param theta: (optional)
            If given, set :py:attr:`params` using this vector before
            calculating the calibration polynomial. ndarray of shape
            ``(ndim,)``

        :param obs:
            A dictionary of observational data, must contain the key
            ``"wavelength"``

        :returns cal:
           If ``params["cal_type"]`` is ``"poly"``, a polynomial given by
           :math:`\times (\Sum_{m=0}^M```'poly_coeffs'[m]``:math:` \times T_n(x))`.
        """

        if self.poly_param_name in kwargs:
            mask = self.get('mask', slice(None))
            # map unmasked wavelengths to the interval -1, 1
            # masked wavelengths may have x>1, x<-1
            x = wave_to_x(self.wavelength, mask)
            # get coefficients.
            c = kwargs[self.poly_param_name]
            poly = chebval(x, c)
        else:
            poly = 1.0

        self.response = poly
        return self.response


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