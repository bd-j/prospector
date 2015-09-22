# Spectral smoothing functionality
# To do:
# 1) Clean up smooth_wave
# 2) Deal properly with vectorized sigma input to smooth_wave, smooth_vel
# 3) sort out how to deal with input spectral resolution.  Is sigma
# the output sigma accounting for input, or just the amount of extra
# broadening to apply?
# 4) speed up smooth_lsf

def smoothspec(wave, spec, sigma, smoothtype='vel', **kwargs):
    """
    """
    if smoothtype == 'vel':
        return smooth_vel(wave, spec, sigma, **kwargs)
    elif smoothtype == 'r':
        sigma_vel = 2.998e5 / sigma
        return smooth_vel(wave, spec, sigma_vel, **kwargs)
    elif smoothtype == 'lambda':
        return smooth_wave(wave, spec, sigma, **kwargs)
    elif smoothtype == 'lsf':
        return smooth_lsf(wave, spec, **kwargs)


def smooth_vel(wave, spec, sigma, outwave=None, inres=0,
               nsigma=10, **extras):
    """Smooth a spectrum in velocity space.  This is insanely slow,
    but general and correct.

    :param sigma:
        desired velocity resolution (km/s)

    :param nsigma:
        Number of sigma away from the output wavelength to consider in
        the integral.  If less than zero, all wavelengths are used.
        Setting this to some positive number decreses the scaling
        constant in the O(N_out * N_in) algorithm used here.

    :param inres:
        The velocity resolution of the input spectrum.
    """
    sigma_eff = np.sqrt(sigma**2 - inres**2)/2.998e5
    if outwave is None:
        outwave = wave
    if sigma <= 0.0:
        return np.interp(wave, outwave, flux)

    lnwave = np.log(wave)
    flux = np.zeros(len(outwave))
    # norm = 1/np.sqrt(2 * np.pi)/sigma
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


def smooth_wave(wave, spec, sigma, outwave=None,
                input_res=0, in_vel=False, nsigma=10,
                **extras):
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
    if outwave is None:
        outwave = wave

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


def smooth_lsf(wave, spec, lsf=None, outwave=None,
                return_kernel=False, fwhm=False, **kwargs):
    """Broaden a spectrum using a wavelength dependent line spread
    function.  This function is only approximate because it doesn't
    actually do the integration over pixels, so for sparsely sampled
    points you'll have problems.

    :param wave:
        input wavelengths
    :param lsf:
        A function that returns the gaussian dispersion at each
        wavelength.  This is assumed to be in sigma unless ``fwhm`` is
        ``True``
    :param outwave:
        Optional output wavelengths

    :param kwargs:
        Passed to lsf()

    :returns newspec:
        The broadened spectrum
    """
    if outwave is None:
        outwave = wave
    if lsf is None:
        return np.interp(outwave, wave, spec)
    dw = np.gradient(wave)
    sigma = lsf(outwave, **kwargs)
    if fwhm:
        sigma = sigma / 2.35
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


