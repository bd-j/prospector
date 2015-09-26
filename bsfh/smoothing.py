# Spectral smoothing functionality
# To do:
# 1) Clean up smooth_wave
# 2) Deal properly with vectorized sigma input to smooth_wave, smooth_vel
# 3) sort out how to deal with input spectral resolution.  Is sigma
# the output sigma accounting for input, or just the amount of extra
# broadening to apply?
# 4) speed up smooth_lsf

import numpy as np

def smoothspec(wave, spec, sigma, outwave=None, smoothtype='vel',
               min_wave_smooth=None, max_wave_smooth=None,
               **kwargs):
    """
    :param wave:
        The wavelength vector of the input spectrum, ndarray.

    :param spec:
        The flux vector of the input spectrum, ndarray

    :param sigma:
        The smoothing parameter.  Units depend on ``smoothtype``
        
    :param outwave:
        The output wavelength vector.  If None then the input
        wavelength vector will be assumed.  If min_wave_smooth or
        max_wave_smooth are also specified, then the output spectrum
        may have differnt dimensions than spec or inwave.

    :param smoothtype:
        The type of smoothing to do.  One of
        
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
        The minimum wavelength of the input vector to consider when
        smoothing the spectrum.  If None then it is determined from
        the minimum of the output wavelength vector, minus 50.0.

    :param max_wave_smooth:
        The maximum wavelength of the input vector to consider when
        smoothing the spectrum.  If None then it is determined from
        the minimum of the output wavelength vector, plus 50.0
    """
    if outwave is None:
        outwave = wave
    # The smoothing limits need to depend on sigma.... and be used to
    # subscript wave and spec
    if min_wave_smooth is None:
        min_wave_smooth = [outwave.min() - 50.0]
    if max_wave_smooth is None:
        max_wave_smooth = [outwave.max() + 50.0]
    smask = (wave > min_wave_smooth[0]) & (wave < max_wave_smooth[0])
    
    if smoothtype == 'vel':
        return smooth_vel(wave, spec, outwave, sigma, **kwargs)
    elif smoothtype == 'R':
        sigma_vel = 2.998e5 / sigma
        return smooth_vel(wave, spec, outwave, sigma_vel, **kwargs)
    elif smoothtype == 'lambda':
        return smooth_wave(wave, spec, outwave, sigma, **kwargs)
    elif smoothtype == 'lsf':
        return smooth_lsf(wave, spec, outwave, **kwargs)


def smooth_vel(wave, spec, outwave, sigma, nsigma=10,
                inres=0, **extras):
    """Smooth a spectrum in velocity space.  This is insanely slow,
    but general and correct.

    :param sigma:
        Desired velocity resolution (km/s), *not* FWHM.

    :param nsigma:
        Number of sigma away from the output wavelength to consider in
        the integral.  If less than zero, all wavelengths are used.
        Setting this to some positive number decreses the scaling
        constant in the O(N_out * N_in) algorithm used here.

    :param inres:
        The velocity resolution of the input spectrum (km/s)
    """
    sigma_eff = np.sqrt(sigma**2 - inres**2)/2.998e5
    if sigma_eff <= 0.0:
        return np.interp(wave, outwave, flux)

    lnwave = np.log(wave)
    flux = np.zeros(len(outwave))
    maxdiff = nsigma * sigma

    for i, w in enumerate(outwave):
        x = np.log(w) - lnwave
        if nsigma > 0:
            good = (x > -maxdiff) & (x < maxdiff)
            x = x[good]
            _spec = spec[good]
        else:
            _spec = spec
        f = np.exp(-0.5 * (x / sigma_eff)**2)
        flux[i] = np.trapz(f * _spec, x) / np.trapz(f, x)
    return flux


def smooth_wave(wave, spec, outwave, sigma, nsigma=10, 
                input_res=0, in_vel=False, **extras):
    """Smooth a spectrum in wavelength space.  This is insanely slow,
    but general and correct (except for the treatment of the input
    resolution if it is velocity)

    :param sigma:
        Desired resolution (*not* FWHM) in wavelength units.

    :param input_res:
        Resolution of the input, in either wavelength units or
        lambda/dlambda (c/v).

    :param in_vel:
        If True, the input spectrum has been smoothed in velocity
        space, and ``inres`` is in dlambda/lambda.

    :param nsigma: (default=10)
        Number of sigma away from the output wavelength to consider in
        the integral.  If less than zero, all wavelengths are used.
        Setting this to some positive number decreses the scaling
        constant in the O(N_out * N_in) algorithm used here.
    """
    if input_res <= 0:
        sigma_eff = sigma
    elif in_vel:
        sigma_min = np.max(outwave) / input_res
        if sigma < sigma_min:
            raise ValueError("Desired wavelength sigma is lower "
                             "than the value possible for this input "
                             "spectrum ({0}).".format(sigma_min))
        # Make an approximate correction for the intrinsic wavelength
        # dependent dispersion.  This doesn't really work.
        sigma_eff = np.sqrt(sigma**2 - (wave / input_res)**2)
    else:
        if sigma < inres:
            raise ValueError("Desired wavelength sigma is lower "
                             "than the value possible for this input "
                             "spectrum ({0}).".format(sigma_min))
        sigma_eff = np.sqrt(sigma**2 - input_res**2)

    flux = np.zeros(len(outwave))
    for i, w in enumerate(outwave):
        x = (wave - w) / sigma_eff
        if nsigma > 0:
            good = np.abs(x) < nsigma
            x = x[good]
            _spec = spec[good]
            _wave = wave[good]
        else:
            _spec = spec
            _wave = wave
        f = np.exp(-0.5 * x**2)
        flux[i] = np.trapz(f * _spec, _wave) / np.trapz(f, _wave)
    return flux


def smooth_lsf(wave, spec, outwave, lsf=None, return_kernel=False,
               **kwargs):
    """Broaden a spectrum using a wavelength dependent line spread
    function.  This function is only approximate because it doesn't
    actually do the integration over pixels, so for sparsely sampled
    points you'll have problems.

    :param wave:
        Input wavelengths.

    :param lsf:
        A function that returns the gaussian dispersion at each
        wavelength.  This is assumed to be in sigma unless ``fwhm`` is
        ``True``

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
    kernel = kernel / kernel.sum(axis=1)[:,None]
    newspec = np.dot(kernel, spec)
    # kernel /= np.trapz(kernel, wave, axis=1)[:, None]
    # newspec = np.trapz(kernel * spec[None, :], wave, axis=1)
    if return_kernel:
        return newspec, kernel
    return newspec


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


