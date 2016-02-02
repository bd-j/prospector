# Spectral smoothing functionality
# To do:
# 3) add extra zero-padding for FFT algorithms so they don't go funky at the
#    edges?

import numpy as np
from numpy.fft import fft, ifft, fftfreq, rfftfreq

__all__ = ["smoothspec", "smooth_wave", "smooth_vel", "smooth_lsf",
           "smooth_wave_fft", "smooth_vel_fft", "smooth_fft", "mask_wave",
           "resample_wave"]

ckms = 2.998e5
sigma_to_fwhm = 2.355


def smoothspec(wave, spec, resolution, outwave=None,
               smoothtype='vel', fftsmooth=False,
               min_wave_smooth=0, max_wave_smooth=np.inf, **kwargs):
    """
    :param wave:
        The wavelength vector of the input spectrum, ndarray.  Assumed
        angstroms.

    :param spec:
        The flux vector of the input spectrum, ndarray

    :param resolution:
        The smoothing parameter.  Units depend on ``smoothtype``

    :param outwave:
        The output wavelength vector.  If None then the input wavelength vector
        will be assumed, though if min_wave_smooth or max_wave_smooth are also
        specified, then the output spectrum may have different length than spec
        or wave, or the convulution may be strange outside of min_wave_smooth
        and max_wave_smooth.  Basically, always set outwave to be safe.

    :param smoothtype:
        The type of smoothing to do.  One of:

        * `vel` - velocity smoothing, ``sigma`` units are in km/s
          (dispersion not FWHM)
        * `R` - resolution smoothing, ``sigma`` is in units of \lambda/
          \sigma(\lambda) (where \sigma(\lambda) is dispersion, not FWHM)
        * `lambda` - wavelength smoothing.  ``sigma`` is in units of \AA
        * `lsf` - line-spread function.  Use an aribitrary line spread
          function, which must be present as the ``lsf`` keyword.  In
          this case ``sigma`` is ignored, but all additional keywords
          will be passed to the `lsf` function.

    :param min_wave_smooth:
        The minimum wavelength of the input vector to consider when smoothing
        the spectrum.  If None then it is determined from the output wavelength
        vector and padded by some multiple of the desired resolution.

    :param max_wave_smooth:
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

    elif smoothtype == 'R':
        linear = False
        units = 'km/s'
        Rsigma = resolution
        sigma = ckms / Rsigma
        fwhm = sigma * sigma_to_fwhm
        R = ckms / fwhm
        width = Rsigma
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

    elif smoothtype == 'lsf':
        linear = True
        width = 100

    # Mask the input spectrum depending on outwave or the wave_smooth kwargs
    mask = mask_wave(wave, width=width, outwave=outwave, linear=linear,
                     wlo=min_wave_smooth, whi=max_wave_smooth, **kwargs)
    w = wave[mask]
    s = spec[mask]
    if outwave is None:
        outwave = wave

    # Actually do the smoothing
    if linear:
        if smoothtype == 'lsf':
            return smooth_lsf(w, s, outwave, **kwargs)
        elif fftsmooth:
            return smooth_wave_fft(w, s, outwave, sigma, **kwargs)
        else:
            return smooth_wave(w, s, outwave, sigma, **kwargs)
    else:
        if fftsmooth:
            return smooth_vel_fft(w, s, outwave, sigma, **kwargs)
        else:
            return smooth_vel(w, s, outwave, sigma, **kwargs)


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
    """Smooth a spectrum in wavelength space, using FFTs. This is fast, but makes
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
        Desired velocity resolution (km/s), *not* FWHM.

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
    # sure it's nearly constant.  Should be by design (see resample_wave)
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

    :param inres:
        Resolution of the input, in either wavelength units or
        lambda/dlambda (c/v).

    :param in_vel:
        If True, the input spectrum has been smoothed in velocity
        space, and ``inres`` is in lambda/dlambda.

    :param nsigma: (default=10)
        Number of sigma away from the output wavelength to consider in the
        integral.  If less than zero, all wavelengths are used.  Setting this
        to some positive number decreses the scaling constant in the O(N_out *
        N_in) algorithm used here.

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


def smooth_lsf(wave, spec, outwave, lsf=None, return_kernel=False,
               **kwargs):
    """Broaden a spectrum using a wavelength dependent line spread function.
    This function is only approximate because it doesn't actually do the
    integration over pixels, so for sparsely sampled points you'll have
    problems.

    :param wave:
        Input wavelengths.

    :param lsf:
        A function that returns the gaussian dispersion at each wavelength.
        This is assumed to be in sigma, not FWHM.

    :param outwave:
        Output wavelengths

    :param kwargs:
        Passed to lsf()

    :returns newspec:
        The broadened spectrum
    """
    if lsf is None:
        return np.interp(outwave, wave, spec)
    dw = np.gradient(wave)
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


def downsample_onespec(wave, spec, outwave, outres,
                       smoothtype='r', **kwargs):
    """
    """
    outspec = []
    # loop over the output segments
    for owave, ores in zip(outwave, outres):
        wmin, wmax = owave.min(), owave.max()
        if smoothtype == 'r':
            sigma = 2.998e5 / ores  # in km/s
            smin = wmin - 5 * wmin/ores
            smax = wmax + 5 * wmax/ores
        elif smoothype == 'lambda':
            sigma = ores  # in AA
            smin = wmin - 5 * sigma
            smax = wmax + 5 * sigma
        imin = np.argmin(np.abs(smin - wave))
        imax = np.argmin(np.abs(smax - wave))
        ospec = smoothspec(wave[imin:imax], spec[imin:imax], sigma,
                           smoothtype=smoothtype, outwave=owave, **kwargs)
        outspec += [ospec]
    return outspec
