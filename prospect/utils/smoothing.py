# Spectral smoothing functionality
# To do:
# 3) add extra zero-padding for FFT algorithms so they don't go funky at the
#    edges?

import numpy as np
from numpy.fft import fft, ifft, fftfreq, rfftfreq

__all__ = ["smoothspec", "smooth_wave", "smooth_vel", "smooth_lsf",
           "smooth_wave_fft", "smooth_vel_fft", "smooth_fft", "smooth_lsf_fft",
           "mask_wave", "resample_wave"]

ckms = 2.998e5
sigma_to_fwhm = 2.355


def smoothspec(wave, spec, resolution=None, outwave=None,
               smoothtype="vel", fftsmooth=True,
               min_wave_smooth=0, max_wave_smooth=np.inf, **kwargs):
    """
    :param wave:
        The wavelength vector of the input spectrum, ndarray.  Assumed
        angstroms.

    :param spec:
        The flux vector of the input spectrum, ndarray

    :param resolution:
        The smoothing parameter.  Units depend on ``smoothtype``.

    :param outwave:
        The output wavelength vector.  If ``None`` then the input wavelength
        vector will be assumed, though if ``min_wave_smooth`` or
        ``max_wave_smooth`` are also specified, then the output spectrum may
        have different length than ``spec`` or ``wave``, or the convolution may
        be strange outside of ``min_wave_smooth`` and ``max_wave_smooth``.
        Basically, always set ``outwave`` to be safe.

    :param smoothtype: (optional default: "vel")
        The type of smoothing to do.  One of:

        * "vel" - velocity smoothing, ``resolution`` units are in km/s
          (dispersion not FWHM)
        * "R" - resolution smoothing, ``resolution`` is in units of \lambda/
          \sigma(\lambda) (where \sigma(\lambda) is dispersion, not FWHM)
        * "lambda" - wavelength smoothing.  ``resolution`` is in units of \AA
        * "lsf" - line-spread function.  Use an aribitrary line spread
          function, which can be given as a vector the same length as ``wave``
          that gives the dispersion (in AA) at each wavelength.  Alternatively,
          if ``resolution`` is ``None`` then a line-spread function must be
          present as an additional ``lsf`` keyword.  In this case all
          additional keywords as well as the ``wave`` vector will be passed to
          this ``lsf`` function.

    :param fftsmooth: (optional, default: True)
        Switch to use FFTs to do the smoothing, usually resulting in massive
        speedups of all algorithms.

    :param min_wave_smooth: (optional default: 0)
        The minimum wavelength of the input vector to consider when smoothing
        the spectrum.  If ``None`` then it is determined from the output
        wavelength vector and padded by some multiple of the desired
        resolution.

    :param max_wave_smooth: (optional default: Inf)
        The maximum wavelength of the input vector to consider when smoothing
        the spectrum.  If None then it is determined from the output wavelength
        vector and padded by some multiple of the desired resolution.

    :param inres: (optional)
        If given, this parameter specifies the resolution of the input.  This
        resolution is subtracted in quadrature from the target output
        resolution before the kernel is formed.

        In certain cases this can be used to properly switch from resolution
        that is constant in velocity to one that is constant in wavelength,
        taking into account the wavelength dependence of the input resolution
        when defined in terms of lambda.  This is possible iff:
        * ``fftsmooth`` is False
        * ``smoothtype`` is ``"lambda"``
        * The optional ``in_vel`` parameter is supplied and True.

        The units of ``inres`` should be the same as the units of
        ``resolution``, except in the case of switching from velocity to
        wavelength resolution, in which case the units of ``inres`` should be
        in units of lambda/sigma_lambda.

    :param in_vel: (optional)
        If supplied and True, the ``inres`` parameter is assumed to be in units
        of lambda/sigma_lambda. This parameter is ignored **unless** the
        ``smoothtype`` is ``"lambda"`` and ``fftsmooth`` is False.

    :returns flux:
        The smoothed spectrum on the `outwave` grid, ndarray.
    """
    if smoothtype == 'vel':
        linear = False
        units = 'km/s'
        sigma = resolution
        fwhm = sigma * sigma_to_fwhm
        Rsigma = ckms / sigma
        R = ckms / fwhm
        width = Rsigma
        assert np.size(sigma) == 1, "`resolution` must be scalar for `smoothtype`='vel'"

    elif smoothtype == 'R':
        linear = False
        units = 'km/s'
        Rsigma = resolution
        sigma = ckms / Rsigma
        fwhm = sigma * sigma_to_fwhm
        R = ckms / fwhm
        width = Rsigma
        assert np.size(sigma) == 1, "`resolution` must be scalar for `smoothtype`='R'"
        # convert inres from Rsigma to sigma (km/s)
        try:
            kwargs['inres'] = ckms / kwargs['inres']
        except(KeyError):
            pass

    elif smoothtype == 'lambda':
        linear = True
        units = 'AA'
        sigma = resolution
        fwhm = sigma * sigma_to_fwhm
        Rsigma = None
        R = None
        width = sigma
        assert np.size(sigma) == 1, "`resolution` must be scalar for `smoothtype`='lambda'"

    elif smoothtype == 'lsf':
        linear = True
        width = 100
        sigma = resolution

    else:
        raise ValueError("smoothtype {} is not valid".format(smoothtype))

    # Mask the input spectrum depending on outwave or the wave_smooth kwargs
    mask = mask_wave(wave, width=width, outwave=outwave, linear=linear,
                     wlo=min_wave_smooth, whi=max_wave_smooth, **kwargs)
    w = wave[mask]
    s = spec[mask]
    if outwave is None:
        outwave = wave

    # Choose the smoothing method
    if smoothtype == 'lsf':
        if fftsmooth:
            smooth_method = smooth_lsf_fft
            if sigma is not None:
                # mask the resolution vector
                sigma = resolution[mask]
        else:
            smooth_method = smoooth_lsf
            if sigma is not None:
                # convert to resolution on the output wavelength grid
                sigma = np.interp(outwave, wave, resolution)
    elif linear:
        if fftsmooth:
            smooth_method = smooth_wave_fft
        else:
            smooth_method = smooth_wave
    else:
        if fftsmooth:
            smooth_method = smooth_vel_fft
        else:
            smooth_method = smooth_vel

    # Actually do the smoothing and return
    return smooth_method(w, s, outwave, sigma, **kwargs)


def smooth_vel(wave, spec, outwave, sigma, nsigma=10, inres=0, **extras):
    """Smooth a spectrum in velocity space.  This is insanely slow, but general
    and correct.

    :param wave:
        Wavelength vector of the input spectrum.

    :param spec:
        Flux vector of the input spectrum.

    :param outwave:
        Desired output wavelength vector.

    :param sigma:
        Desired velocity resolution (km/s), *not* FWHM.

    :param nsigma:
        Number of sigma away from the output wavelength to consider in the
        integral.  If less than zero, all wavelengths are used.  Setting this
        to some positive number decreses the scaling constant in the O(N_out *
        N_in) algorithm used here.

    :param inres:
        The velocity resolution of the input spectrum (km/s), *not* FWHM.
    """
    sigma_eff_sq = sigma**2 - inres**2
    if np.any(sigma_eff_sq) < 0.0:
        raise ValueError("Desired velocity resolution smaller than the value"
                         "possible for this input spectrum.".format(inres))
    # sigma_eff is in units of sigma_lambda / lambda
    sigma_eff = np.sqrt(sigma_eff_sq) / ckms

    lnwave = np.log(wave)
    flux = np.zeros(len(outwave))
    for i, w in enumerate(outwave):
        x = (np.log(w) - lnwave) / sigma_eff
        if nsigma > 0:
            good = np.abs(x) < nsigma
            x = x[good]
            _spec = spec[good]
        else:
            _spec = spec
        f = np.exp(-0.5 * x**2)
        flux[i] = np.trapz(f * _spec, x) / np.trapz(f, x)
    return flux


def smooth_vel_fft(wavelength, spectrum, outwave, sigma_out, inres=0.0,
                   **extras):
    """Smooth a spectrum in velocity space, using FFTs. This is fast, but makes
    some assumptions about the form of the input spectrum and can have some
    issues at the ends of the spectrum depending on how it is padded.

    :param wavelength:
        Wavelength vector of the input spectrum. An assertion error will result
        if this is not a regular grid in wavelength.

    :param spectrum:
        Flux vector of the input spectrum.

    :param outwave:
        Desired output wavelength vector.

    :param sigma_out:
        Desired velocity resolution (km/s), *not* FWHM.  Scalar or length 1 array.

    :param inres:
        The velocity resolution of the input spectrum (km/s), dispersion *not*
        FWHM.
    """
    # The kernel width for the convolution.
    sigma = np.sqrt(sigma_out**2 - inres**2)
    if sigma <= 0:
        return np.interp(outwave, wavelength, spectrum)

    # make length of spectrum a power of 2 by resampling
    wave, spec = resample_wave(wavelength, spectrum)

    # get grid resolution (*not* the resolution of the input spectrum) and make
    # sure it's nearly constant.  It should be, by design (see resample_wave)
    invRgrid = np.diff(np.log(wave))
    assert invRgrid.max() / invRgrid.min() < 1.05
    dv = ckms * np.median(invRgrid)

    # Do the convolution
    spec_conv = smooth_fft(dv, spec, sigma)
    # interpolate onto output grid
    if outwave is not None:
        spec_conv = np.interp(outwave, wave, spec_conv)

    return spec_conv


def smooth_wave(wave, spec, outwave, sigma, nsigma=10, inres=0, in_vel=False,
                **extras):
    """Smooth a spectrum in wavelength space.  This is insanely slow, but
    general and correct (except for the treatment of the input resolution if it
    is velocity)

    :param wave:
        Wavelength vector of the input spectrum.

    :param spec:
        Flux vector of the input spectrum.

    :param outwave:
        Desired output wavelength vector.

    :param sigma:
        Desired resolution (*not* FWHM) in wavelength units.  This can be a
        vector of same length as ``wave``, in which case a wavelength dependent
        broadening is calculated

    :param nsigma: (optional, default=10)
        Number of sigma away from the output wavelength to consider in the
        integral.  If less than zero, all wavelengths are used.  Setting this
        to some positive number decreses the scaling constant in the O(N_out *
        N_in) algorithm used here.

    :param inres: (optional, default: 0.0)
        Resolution of the input, in either wavelength units or
        lambda/dlambda (c/v).  Ignored if <= 0.

    :param in_vel: (optional, default: False)
        If True, the input spectrum has been smoothed in velocity
        space, and ``inres`` is assumed to be in lambda/dlambda.

    :returns flux:
        The output smoothed flux vector, same length as ``outwave``.
    """
    # sigma_eff is in angstroms
    if inres <= 0:
        sigma_eff_sq = sigma**2
    elif in_vel:
        # Make an approximate correction for the intrinsic wavelength
        # dependent dispersion.  This sort of maybe works.
        sigma_eff_sq = sigma**2 - (wave / inres)**2
    else:
        sigma_eff_sq = sigma**2 - inres**2
    if np.any(sigma_eff_sq < 0):
        raise ValueError("Desired wavelength sigma is lower than the value "
                         "possible for this input spectrum.")

    sigma_eff = np.sqrt(sigma_eff_sq)
    flux = np.zeros(len(outwave))
    for i, w in enumerate(outwave):
        x = (wave - w) / sigma_eff
        if nsigma > 0:
            good = np.abs(x) < nsigma
            x = x[good]
            _spec = spec[good]
        else:
            _spec = spec
        f = np.exp(-0.5 * x**2)
        flux[i] = np.trapz(f * _spec, x) / np.trapz(f, x)
    return flux


def smooth_wave_fft(wavelength, spectrum, outwave, sigma_out=1.0,
                    inres=0.0, **extras):
    """Smooth a spectrum in wavelength space, using FFTs.  This is fast, but
    makes some assumptions about the input spectrum, and can have some
    issues at the ends of the spectrum depending on how it is padded.

    :param wavelength:
        Wavelength vector of the input spectrum.

    :param spectrum:
        Flux vector of the input spectrum.

    :param outwave:
        Desired output wavelength vector.

    :param sigma:
        Desired resolution (*not* FWHM) in wavelength units.

    :param inres:
        Resolution of the input, in wavelength units (dispersion not FWHM).

    :returns flux:
        The output smoothed flux vector, same length as ``outwave``.
    """
    # restrict wavelength range (for speed)
    # should also make nearest power of 2
    wave, spec = resample_wave(wavelength, spectrum, linear=True)

    # The kernel width for the convolution.
    sigma = np.sqrt(sigma_out**2 - inres**2)
    if sigma < 0:
        return np.interp(wave, outwave, flux)

    # get grid resolution (*not* the resolution of the input spectrum) and make
    # sure it's nearly constant.  Should be by design (see resample_wave)
    Rgrid = np.diff(wave)
    assert Rgrid.max() / Rgrid.min() < 1.05
    dw = np.median(Rgrid)

    # Do the convolution
    spec_conv = smooth_fft(dw, spec, sigma)
    # interpolate onto output grid
    if outwave is not None:
        spec_conv = np.interp(outwave, wave, spec_conv)
    return spec_conv


def smooth_lsf(wave, spec, outwave, sigma=None, lsf=None, return_kernel=False,
               **kwargs):
    """Broaden a spectrum using a wavelength dependent line spread function.
    This function is only approximate because it doesn't actually do the
    integration over pixels, so for sparsely sampled points you'll have
    problems.  This function needs to be checked and possibly rewritten.

    :param wave:
        Input wavelengths.  ndarray of shape (nin,)

    :param spec:
        Input spectrum.  ndarray of same shape as ``wave``.

    :param outwave:
        Output wavelengths, ndarray of shape (nout,)

    :param sigma: (optional, default: None)
        The dispersion (not FWHM) as a function of wavelength that you want to
        apply to the input spectrum.  ``None`` or ndarray of same length as
        ``outwave``.  If ``None`` then the wavelength dependent dispersion will be
        calculated from the function supplied with the ``lsf`` keyward.

    :param lsf:
        A function that returns the gaussian dispersion at each wavelength.
        This is assumed to be in sigma, not FWHM.

    :param kwargs:
        Passed to the function supplied in the ``lsf`` keyword.

    :param return_kernel: (optional, default: False)
        If True, return the kernel used to broaden the spectrum as ndarray of
        shape (nout, nin).

    :returns newspec:
        The broadened spectrum, same length as ``outwave``.
    """
    if (lsf is None) and (sigma is None):
        return np.interp(outwave, wave, spec)
    dw = np.gradient(wave)
    if sigma is None:
        sigma = lsf(outwave, **kwargs)
    kernel = outwave[:, None] - wave[None, :]
    kernel = (1 / (sigma * np.sqrt(np.pi * 2))[:, None] *
              np.exp(-kernel**2 / (2 * sigma[:, None]**2)) *
              dw[None, :])
    # should this be axis=0 or axis=1?
    kernel = kernel / kernel.sum(axis=1)[:, None]
    newspec = np.dot(kernel, spec)
    # kernel /= np.trapz(kernel, wave, axis=1)[:, None]
    # newspec = np.trapz(kernel * spec[None, :], wave, axis=1)
    if return_kernel:
        return newspec, kernel
    return newspec


def smooth_lsf_fft(wave, spec, outwave, sigma=None, lsf=None, pix_per_sigma=2,
                   eps=0.25, preserve_all_input_frequencies=False, **kwargs):
    """Smooth a spectrum by a wavelength dependent line-spread function, using
    FFTs.

    :param wave:
        Wavelength vector of the input spectrum.

    :param spectrum:
        Flux vector of the input spectrum.

    :param outwave:
        Desired output wavelength vector.

    :param sigma: (optional)
        Dispersion (in same units as ``wave``) as a function `wave`.  ndarray
        of same length as ``wave``.  If not given, sigma will be computed from
        the function provided by the ``lsf`` keyword.

    :param lsf: (optional)
        Function used to calculate the dispersion as a function of wavelength.
        Must be able to take as an argument the ``wave`` vector and any extra
        keyword arguments and return the dispersion (in the same units as the
        input wavelength vector) at every value of ``wave``.  If not provided
        then ``sigma`` must be specified.

    :param pix_per_sigma: (optional, default: 2)
        Number of pixels per sigma of the smoothed spectrum to use in
        intermediate interpolation and FFT steps. Increasing this number will
        increase the accuracy of the output (to a point), and the run-time, by
        preserving all high-frequency information in the input spectrum.

    :param preserve_all_input_frequencies: (default: False)
        This is a switch to use a very dense sampling of the input spectrum
        that preserves all input frequencies.  It can significantly increase
        the call time for often modest gains...

    :param eps: (optional)
        Deprecated.

    :param **kwargs:
        All additional keywords are passed to the function supplied to the
        ``lsf`` keyword, if present.

    :returns flux:
        The input spectrum smoothed by the wavelength dependent line-spread
        function.  Same length as ``outwave``.
    """
    # This is sigma vs lambda
    if sigma is None:
        sigma = lsf(wave, **kwargs)

    # Now we need the CDF of 1/sigma, which provides the relationship between x and lambda
    # does dw go in numerator or denominator?
    # I think numerator but should be tested
    dw = np.gradient(wave)
    cdf = np.cumsum(dw / sigma)
    cdf /= cdf.max()

    # Now we create an evenly sampled grid in the x coordinate on the interval [0,1]
    # and convert that to lambda using the cdf.
    # This should result in some power of two x points, for FFT efficiency

    # Furthermore, the number of points should be high enough that the
    # resolution is critically sampled.  And we want to know what the
    # resolution is in this new coordinate.
    # There are two possible ways to do this

    # 1) Choose a point ~halfway in the spectrum
    # half = len(wave) / 2
    # Now get the x coordinates of a point eps*sigma redder and bluer
    # wave_eps = eps * np.array([-1, 1]) * sigma[halpha]
    # x_h_eps = np.interp(wave[half] + wave_eps, wave, cdf)
    # Take the differences to get dx and dsigma and ratio to get x per sigma
    # x_per_sigma = np.diff(x_h_eps) / (2.0 * eps) #x_h_epsilon - x_h

    # 2) Get for all points (slower?):
    sigma_per_pixel = (dw / sigma)
    x_per_pixel = np.gradient(cdf)
    x_per_sigma = np.nanmedian(x_per_pixel / sigma_per_pixel)
    N = pix_per_sigma / x_per_sigma

    # Alternatively, just use the smallest dx of the input, divided by two for safety
    # Assumes the input spectrum is critically sampled.
    # And does not actually give x_per_sigma, so that has to be determined anyway
    if preserve_all_input_frequencies:
        # preserve more information in the input spectrum, even when way higher
        # frequency than the resolution of the output.  Leads to slightly more
        # accurate output, but with a substantial time hit
        N = max(N, 1.0 / np.nanmin(x_per_pixel))

    # Now find the smallest power of two that divides the interval (0, 1) into
    # segments that are smaller than dx
    nx = int(2**np.ceil(np.log2(N)))

    # now evenly sample in the x coordinate
    x = np.linspace(0, 1, nx)
    dx = 1.0 / nx

    # And now we get the spectrum at the lambda coordinates of the even grid in x
    lam = np.interp(x, cdf, wave)
    newspec = np.interp(lam, wave, spec)

    # And now we convolve.
    # If we did not know sigma in terms of x we could estimate it here
    # from the resulting sigma(lamda(x)) / dlambda(x):
    # dlam = np.gradient(lam)
    # sigma_x = np.median(lsf(lam, **kwargs) / dlam)
    # But the following just uses the fact that we know x_per_sigma (duh).
    spec_conv = smooth_fft(dx, newspec, x_per_sigma)

    # and interpolate back to the output wavelength grid.
    return np.interp(outwave, lam, spec_conv)


def smooth_fft(dx, spec, sigma):
    """Basic math for FFT convolution with a gaussian kernel.

    :param dx:
        The wavelength or velocity spacing, same units as sigma

    :param sigma:
        The width of the gaussian kernel, same units as dx

    :param spec:
        The spectrum flux vector
    """
    # The Fourier coordinate
    ss = rfftfreq(len(spec), d=dx)
    # Make the fourier space taper; just the analytical fft of a gaussian
    taper = np.exp(-2 * (np.pi ** 2) * (sigma ** 2) * (ss ** 2))
    ss[0] = 0.01  # hack
    # Fourier transform the spectrum
    spec_ff = np.fft.rfft(spec)
    # Multiply in fourier space
    ff_tapered = spec_ff * taper
    # Fourier transform back
    spec_conv = np.fft.irfft(ff_tapered)
    return spec_conv


def mask_wave(wavelength, width=1, wlo=0, whi=np.inf, outwave=None,
              nsigma_pad=20.0, linear=False, **extras):
    """Restrict wavelength range (for speed) but include some padding based on
    the desired resolution.
    """
    # Base wavelength limits
    if outwave is not None:
        wlim = np.array([outwave.min(), outwave.max()])
    else:
        wlim = np.squeeze(np.array([wlo, whi]))
    # Pad by nsigma * sigma_wave
    if linear:
        wlim += nsigma_pad * width * np.array([-1, 1])
    else:
        wlim *= (1 + nsigma_pad / width * np.array([-1, 1]))
    mask = (wavelength > wlim[0]) & (wavelength < wlim[1])
    return mask


def resample_wave(wavelength, spectrum, linear=False):
    """Resample spectrum, so that the number of elements is the next highest
    power of two.  This uses np.interp.  Note that if the input wavelength grid
    did not critically sample the spectrum then there is no gaurantee the
    output wavelength grid will.
    """
    wmin, wmax = wavelength.min(), wavelength.max()
    nw = len(wavelength)
    nnew = 2.0**(np.ceil(np.log2(nw)))
    if linear:
        Rgrid = np.diff(wavelength)  # in same units as ``wavelength``
        w = np.linspace(wmin, wmax, nnew)
    else:
        Rgrid = np.diff(np.log(wavelength))  # actually 1/R
        lnlam = np.linspace(np.log(wmin), np.log(wmax), nnew)
        w = np.exp(lnlam)
    # Make sure the resolution really is nearly constant
    #assert Rgrid.max() / Rgrid.min() < 1.05
    s = np.interp(w, wavelength, spectrum)
    return w, s


def subtract_input_resolution(res_in, res_target, smoothtype_in, smoothtype_target, wave=None):
    """Subtract the input resolution (in quadrature) from a target output
    resolution to get the width of the kernel that will convolve the input to
    the output.  Assumes all convolutions are with gaussians.
    """
    if smoothtype_in == "R":
        width_in = 1.0 / res_in
    else:
        width_in = res_in
    if smoothtype_target == "R":
        width_target = 1.0 / res_target
    else:
        width_target = res_target

    if smoothtype_in == smoothtype_target:
        dwidth_sq = width_target**2 - width_in**2

    elif (smoothtype_in == "vel") & (smoothype_target == "lambda"):
        dwidth_sq = width_target**2 - (wave * width_in / ckms)**2

    elif (smoothtype_in == "R") & (smoothype_target == "lambda"):
        dwidth_sq = width_target**2 - (wave * width_in)**2

    elif (smoothtype_in == "lambda") & (smoothtype_target == "vel"):
        dwidth_sq = width_target**2 - (ckms * width_in / wave)**2

    elif (smoothtype_in == "lambda") & (smoothtype_target == "R"):
        dwidth_sq = width_target**2 - (width_in / wave)**2

    elif (smoothtype_in == "R") & (smoothtype_target == "vel"):
        print("srsly?")
        return None
    elif (smoothtype_in == "vel") & (smoothtype_target == "R"):
        print("srsly?")
        return None

    if np.any(dwidth_sq <= 0):
        print("Warning:  Desired resolution is better than input resolution")
        dwidth_sq = np.clip(dwidth_sq, 0, np.inf)

    if smoothtype_target == "R":
        return 1.0 / np.sqrt(dwidth_sq)
    else:
        return np.sqrt(dwidth_sq)

    return delta_width
