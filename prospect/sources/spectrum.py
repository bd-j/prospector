# Useful constants
lsun = 3.846e33
pc = 3.085677581467192e18  # in cm
lightspeed = 2.998e18  # AA/s
ckms = 2.998e5 # km/s
jansky_cgs = 1e-23
# value to go from L_sun/AA to erg/s/cm^2/AA at 10pc
to_cgs_at10pc = lsun / (4.0 * np.pi * (pc*10)**2)

from astropy.cosmology import WMAP9

class sbasis(object):


    def get_spectrum_josh(self, outwave=None, filters=None, peraa=False, **params):
        raise(NotImplementedError)



    def get_spectrum(self, outwave=None, filters=None, **params):

        # --- Spectrum in Lsun/Hz per solar mass formed, restframe ---
        wave, spectrum, mfrac = self.get_galaxy_spectrum(**params)

        # --- Get stellar smoothing ---
        veldisp = params.get('velocity_dispersion', None)

        # --- Get nebular parameters ---
        if params.get('add_neb_emission', False):
            nebveldisp = params.get('neb_velocity_dispersion', veldisp)
            delv = params.get('neb_velocity_offset', 0.0)
            linemask = params.get('linemask', slice(None))
            # Nebular lines in Lsun per solar mass formed
            elines = self.nebline_luminosity[linemask]
            eline_lambda = self.nebline_wavelengths[linemask]
            if delv != 0.0:
                zneb = np.sqrt((1 +  delv / ckms) / (1 - delv/ckms))
                eline_lambda *= (1 + zneb)

        # --- Get different redshifts ---
        cosmo = params.get('cosmology', WMAP9)
        zcosmo = params.get('zred', 0.0)
        vpeculiar = params.get('vpec', 0.0)
        zobs = calculate_zobs(vpeculiar, zcosmo)
        ldist = params.get('lumdist', None)

         # --- Distance and (1+zcosmo) dimming ---
        ldim = distance_dimming(zcosmo, ldist, cosmology=cosmo)

        # -----------------------
        # --- Get Photometry ---
        # -----------------------
        if filters is not None:
            # Define a logarithmic wavelength vector
            lnwavegrid = params.get('lnwavegrid', None)

            if params.get('add_neb_emission', False):
                # Add emission lines for photometry!
                spectrum  += calculate_linespec(wave, eline_lambda, elines,
                                                nebveldisp / ckms * eline_lambda)

            # Actually get the photometry
            phot = get_photometry(wave, spectrum, filters,
                                  zobs=zobs, lnwavegrid=lnwavegrid)
        else:
            phot = 0.0

        # ------------------------
        # --- Get Spectrum ---
        # Here we have to account for:
        # gintrumental line-spread-function, library
        # resolution and adding emission lines in again more precisely.
        # -----------------------
        if outwave is not None:
            instrument_lsf = params.get('instrument_lsf', None)
            library_lsf = params.get('library_lsf', None)
            wobs = wave / (1 + zobs)
            # Smooth by the galaxy velocity dispersion
            # Note that we only smooth the part that matters for the output spectrum.
            wobs_velsmooth, spec_velsmooth = smooth_galaxy(wobs, spectrum, veldisp,
                                                           outwave=outwave)
        
            # Smooth by the instrumental resolution accounting for the library
            # resolution and the wavelength solution
            wavecal = params.get('wavecal_coeffs', None)
            wobs, sobs = smooth_instrument(wobs_velsmooth, spec_velsmooth,
                                           zobs, instrument_lsf, library_lsf,
                                           wavecal)

            # Add nebular emission
            if params.get('add_neb_emission', False):
                sobs += calculate_linespec(wobs, eline_lambda * (1 + zobs), elines,
                                           nebveldisp, instrument_lsf)
            if outwave is None:
                outwave = wave
            spec = np.interp(outwave, wobs, sobs)
        else:
            spec = spectrum

        # ----------------------------------
        # --- Unit conversion and output ---
        # (output is maggies for both spectra and photometry)
        # ----------------------------------
        # Mass normalization
        mass = np.sum(self.params.get('mass', 1.0))
        if np.all(self.params.get('mass_units', 'mformed') == 'mstar'):
            # Convert input normalization units from current stellar mass to mass formed
            mass /= mfrac

        return spec * mass * ldim / jansky_cgs / 3631, phot * mass * ldim, mfrac


def calculate_linespec(wave_array, line_wavelengths, line_luminosities, line_dispersions,
                       instrument_lsf=None):
    """Calculate the spectrum of the nebular lines on the input wavelength
    grid, assuming gaussian line profiles. If you pass `instrument_lsf` then we
    are assuming that the `line_wavelengths` (and `wave_array`) are in the
    observed frame, and we will add the instrument line-spread-function to the
    line width.

    :param wave_array:

    :param line_wavelengths:

    :param line_luminosities:

    :param line_dispersions:

    :param instrument_lsf:

    :returns linespec:
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


def smooth_instrument(wavecal):
    """Apply the instrumental line-spread function to the observed frame
    spectrum accounting for the restframe library resolution.
    """
    

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
        lumdist = cosmology.luminosity_distance(zcosmo).value
        dfactor = (lumdist * 1e5)**2
                
   return to_cgs_at_10pc / dfactor * (1 + zcosmo)


def get_photometry(wave, fnu, filters, zobs=0.0, lnwavegrid=None):
    """Return the photometry in linear units.  These will be maggies if the
    spectrum `fnu` is in ergs/c/cm^2/Hz.
    """
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
