import numpy as np

# Useful constants
from ..utils.constants import to_cgs_at10pc, lightspeed, ckms, jansky_cgs

from astropy.cosmology import WMAP9


__all__ = ["get_photometry", "smooth_galaxy", "smooth_instrument",
           "calculate_linespec", "gauss",
           "calculate_zobs", "distance_dimming"]


class sbasis(object):


    def get_spectrum_josh(self, outwave=None, filters=None, peraa=False, **params):
        raise(NotImplementedError)

    @property
    def nebline_wavelengths(self):
        return self.ssp.emline_wavelengths

    @property
    def nebline_luminosity(self):
        """Emission line luminosities in units of Lsun per solar mass formed
        """
        return self.ssp.emline_luminosity / self.params['mass'].sum()

    def get_spectrum(self, outwave=None, filters=None, **params):

        """Things I don't like:
        1) The lines are being added with not quite the right shape
        2) The line spectrum is calculated twice
        3) The lines for the spectrum are added after the instrumental
        broadening *and wavelength calibration*
        4) The result of smooth galaxy is actually velocity broadening plus library broadening.
        Library resolution is only removed at the instrumental step
        5) We are doing two full convolutions, and gains might be possible if
        we fit in restframe and apply instrument minus library kernel at the SSP
        level: this is not straightforward or clear in the current setup.
        6) Unclear whether instrument_lsf should take wavelengths that include
        the wavelength calibration perturbation or not
        7) params.get() is not the cleanest or clearest pattern....
        """
        # --- Spectrum in Lsun/Hz per solar mass formed, restframe ---
        wave, spectrum, mfrac = self.get_galaxy_spectrum(**params)

        # --- Get stellar smoothing ---
        veldisp = params.get('velocity_dispersion', None)

        # --- Get nebular parameters ---
        add_lines = (params.get('add_neb_emission', False) and
                     (not self.ssp.params['nebemlineinspec']))
        if add_lines:
            nebveldisp = params.get('neb_vel_dispersion', veldisp)
            delv = params.get('neb_vel_offset', 0.0)
            linemask = params.get('linemask', slice(None))
            # Nebular lines in Lsun per solar mass formed
            eline_lum = self.nebline_luminosity[linemask]
            eline_lambda = self.nebline_wavelengths[linemask]
            if delv != 0.0:
                zneb = np.sqrt((1 +  delv / ckms) / (1 - delv/ckms))
                eline_lambda *= (1 + zneb)

        # --- Get different redshifts ---
        cosmo = params.get('cosmology', WMAP9)
        zcosmo = params.get('zred', 0.0)
        vpeculiar = params.get('vpec', 0.0)
        zobs = calculate_zobs(vpeculiar, zcosmo)
        # if fitting spectrum in different frame than photometry
        zspec = params.get('zspec', zobs)
        zphot = params.get('zphot', zobs)
        # if fixing or fitting luminosity distance independently of any given
        # redshifts or cosmology
        ldist = params.get('lumdist', None)

         # --- Distance and (1+zcosmo) dimming ---
        ldim = distance_dimming(zcosmo, ldist, cosmology=cosmo)

        # ----------------------
        # --- Get Photometry ---
        # ----------------------
        if filters is not None:
            # Use the defined logarithmic wavelength vector if present
            lnwavegrid = params.get('lnwavegrid', None)

            # Add emission lines for photometry
            if add_lines:
                photlines  = calculate_linespec(wave, eline_lambda, eline_lum,
                                                nebveldisp)
            else:
                photlines = 0

            # Actually get the photometry
            phot = get_photometry(wave, spectrum + photlines, filters,
                                  zobs=zphot, lnwavegrid=lnwavegrid)
        else:
            phot = 0.0

        # --------------------
        # --- Get Spectrum ---
        # --------------------
        if outwave is not None:
            instrument_lsf = params.get('instrument_lsf', None)
            library_lsf = params.get('library_lsf', None)
            wobs = wave * (1 + zspec)
            # Smooth by the galaxy stellar velocity dispersion.
            # Note that we only smooth the part that matters for the output
            # spectrum.  The output wavelength vector will be something close
            # to the outwave vector but not exactly the same.
            wobs_velsmooth, spec_velsmooth = smooth_galaxy(wobs, spectrum, veldisp,
                                                           outwave=outwave)

            # Smooth by the instrumental resolution accounting for the library
            # resolution and the wavelength solution
            wavecal = params.get('wavecal_coeffs', None)
            wobs, sobs = smooth_instrument(wobs_velsmooth, spec_velsmooth,
                                           outwave=outwave, zobs, instrument_lsf, library_lsf,
                                           wavecal)

            # Add nebular emission to the observed frame spectrum, accounting
            # for instrumental LSF
            if add_lines:
                sobs += calculate_linespec(wobs, eline_lambda * (1 + zobs), elines,
                                           nebveldisp, instrument_lsf)
            spec = np.interp(outwave, wobs, sobs)
        else:
            spec = spectrum + photlines

        # ----------------------------------
        # --- Unit conversion and output ---
        # (output is maggies for both spectra and photometry)
        # ----------------------------------
        # Mass normalization
        mass = np.sum(self.params.get('mass', 1.0))
        if np.all(self.params.get('mass_units', 'mformed') == 'mstar'):
            # Convert input normalization units from current stellar mass to mass formed
            mass /= mfrac

        return spec * (mass*ldim/jansky_cgs/3631), phot * (mass*ldim), mfrac


def calculate_linespec(wave_array, line_wavelengths, line_luminosities, line_dispersions,
                       instrument_lsf=None):
    """Calculate the spectrum of nebular lines on the input wavelength grid,
    assuming gaussian line profiles. If you pass `instrument_lsf` then we are
    assuming that the `line_wavelengths` (and `wave_array`) are in the observed
    frame, and we will add the instrument line-spread-function to the line
    width.

    :param wave_array:
        The wavelength array onto which the lines will be added.  If wave_array
        is input as ln(lambda) and `line_dispersion` and `line_wavelengths` are
        in appropriate units, then the line shapes will be appropriate for a
        velocity dispersion, otherwise they are assumed gaussian in wavelength.

    :param line_wavelengths:
        The location of the lines, same units as `wave_array`, scalar or ndarry
        of shape (nline,)

    :param line_luminosities:
        The luminosities of the lines.

    :param line_dispersions:
        The velocity dispersion of the line, in km/s.  Scalar or ndarray of
        shape (nline,)

    :param instrument_lsf: (optional)
        A function that returns the instrumental dispersion (in same units as
        `wave_array`) as a function of `wave_array`.  This is added to the

    :returns linespec:
       A summed spectrum of the lines, same shape as `wave_array`.
    """
    sigma = line_dispersions / ckms * line_wavelengths
    if instrument_lsf is not None:
        sigma = np.sqrt(sigma**2 + instrument_lsf(line_wavelengths)**2)
    linespec = gauss(wave_array, line_wavelengths, line_luminosities, sigma)
    return linespec


def smooth_galaxy(wave_obs, spectrum, veldisp, outwave=outwave, nsigma_pad=10):
    """Smooth the observed frame spectrum by the restframe velocity dispersion
    of the stellar population.

    :param wave_obs:
    """
    if veldisp <= 0:
        return wave_obs, spectrum
    # do padding
    wlim = np.array([outwave.min(), outwave.max()])
    wlim *= (1 + nsigma_pad * np.array([-1, 1]))
    # do smoothing
    wave, spec = smoothing.smoothspec(wave_obs, spectrum, resolution=veldisp,
                                      smoothtype="vel", fftsmooth=True,
                                      min_wave_smooth=wlim[0], max_wave_smooth=wmin[1])
    return wave, spec


def smooth_instrument(wave_obs, spec, zobs, wavecal=None, outwave=None,
                      instrument_lsf=lambda x: 0., library_lsf=lambda x: 0., wavecal=None):
    """Apply the instrumental line-spread function to the observed frame
    spectrum accounting for the restframe library resolution (via quadrature subtraction)
    """
    dsigma_sq = (instrument_lsf(wave_obs)**2 -
                 library_lsf(wave_obs / (1 + zobs))**2)

    assert np.all(dsigma_sq >= 0), 'instrumental resolution is better than the library resolution'
    # apply wavelength calibration
    if wavecal is not None:
        x = wave_obs - wave_obs.min()
        x = 2.0 * (x / x.max()) - 1.0
        c = np.insert(wavecal, 0, 0)
        # assume coeeficients give shifts in km/s
        b = chebval(x, c) / (lightspeed*1e-13)
        wave_inst = (1 + b) * wave_obs
    else:
        wave_inst = wave_obs

    if np.allclose(dsigma, 0):
        return wave_inst, spec
        
    w, s = smoothing.smoothspec(wave_inst, spec, resolution=np.sqrt(dsigma_sq),
                                outwave=outwave,
                                smoothtype="lsf", fftsmooth=True)
    return w, s


def calculate_zobs(vpec, zcosmo):
    """Only valid for vpec << c.
    """
    zobs = vpec / c * (1 + zcosmo) + zcosmo
    return zobs


def distance_dimming(zcosmo=0.0, lumdist=None, cosmology=WMAP9):
    """Factor to go from lsun/Hz to erg/s/cm^2/Hz at d_L (or at the d_L implied
    by zcosmo and the provided cosmology).  Includes cosmological (1+z)
    """
    if lumdist is not None:
        dfactor = (lumdist * 1e5)**2
    elif (zcosmo == 0):
        # Use 10pc for the luminosity distance
        dfactor = 1
    else:
        ld = cosmology.luminosity_distance(zcosmo).value
        dfactor = (ld * 1e5)**2

   return to_cgs_at_10pc / dfactor * (1 + zcosmo)


def get_photometry(wave, fnu, filters, zobs=0.0, lnwavegrid=None):
    """Return the photometry in linear units.  These will be maggies if the
    spectrum `fnu` is in ergs/c/cm^2/Hz.

    :param wave:
        Wavelength in angstroms, ndarray.

    :param fnu:
        Flux density, same shape as `wave`.

    :param filters:
        A list of `sedpy.observate.Filter` objects.

    :param zobs: (optional, default 0)
        The redshift of the object (including peculiar velocity).

    :param lnwavegrid: (optional)
        A logarithmic wavelength grid.  That is, an array of wavelengths (in
    angstroms) that are evenly spaved in ln(wavelength).  This is assumed to
    have the same ln(wave) spacing as the gridded filter transmission curves in
    the `Filter` objects, and can be used to speed up the filter projections."""
    wa = wave * (1 + zobs)

    if lnwavegrid is not None:
        spec = np.interp(lnwavegrid, wa, fnu * lightspeed / wa**2)
        wa = lnwavegrid
        gridded = True
    else:
        spec = lightspeed/wa**2 *fnu
        gridded = False

    mags = getSED(wa, spec, filters, gridded=gridded)
    phot = np.atleast_1d(10**(-0.4 * mags))
    return phot


def gauss(x, mu, A, sigma, sumspec=True):
    """Lay down mutiple gaussians on the x-axis.
    """
    mu, A, sigma = np.atleast_2d(mu), np.atleast_2d(A), np.atleast_2d(sigma)
    val = (A / (sigma * np.sqrt(np.pi * 2)) *
           np.exp(-(x[:, None] - mu)**2 / (2 * sigma**2)))
    if sumspec:
        return val.sum(axis=-1)
    else:
        return val.T
