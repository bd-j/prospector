# Useful constants
lsun = 3.846e33
pc = 3.085677581467192e18  # in cm
lightspeed = 2.998e18  # AA/s
ckms = 2.998e5 # km/s
jansky_cgs = 1e-23
# value to go from L_sun/AA to erg/s/cm^2/AA at 10pc
to_cgs_at10pc = lsun/(4.0 * np.pi * (pc*10)**2)

from astropy.cosmology import WMAP9

class sbasis(object):


    def get_spectrum_josh(self, outwave=None, filters=None, peraa=False, **params):
        raise(NotImplementedError)
    
    def get_spectrum(self, outwave=None, filters=None, **params):

        # Spectrum in Lsun/Hz per solar mass formed, restframe
        wave, spectrum, mfrac = self.get_galaxy_spectrum(**params)
 
        # get different redshifts
        cosmo = params.get('cosmology', WMAP9)
        zcosmo = params.get('zred', 0.0)
        vpeculiar = params.get('vpec', 0.0)
        zobs = calculate_zobs(vpeculiar, zcosmo)
        ldist = params.get('lumdist', None)
        
        # get various smoothings
        veldisp = self.params.get('veldisp', 100.)
        # get nebular parameters
        if params.get('add_neb_emission', False):
            nebveldisp = self.params.get('nebveldisp', veldisp)
            delv = self.params.get('nebvel_offset', 0.0)
            linemask = params.get('linemask', slice(None))
             # Nebular lines in Lsun per solar mass formed
            elines = self.nebline_luminosity[linemask]
            eline_lambda = self.nebline_wavelengths[linemask] * (1 + delv / ckms) # approximation for low delv

         # dimming.
        ldim = distance_dimming(zcosmo, ldist, cosmology=cosmo)

        # Define a logarithmic wavelength vector /// and interpolate onto it
        lnwavegrid = params.get('lnwavegrid', None)
        
        if filters is not None:
            if params.get('add_neb_emission', False):
                # Add emission lines for photometry!
                linespec = gauss(wave, eline_lambda, elines, nebveldisp / ckms * eline_lambda)
            else:
                linespec = 0.0

            # Get photometry
            phot = get_photometry(wave, spectrum + linespec, filters,
                                  zobs=zobs, lnwavegrid=lnwavegrid)
        else:
            phot = 0.0

        # This was the photometry part.  Still have to do spectroscopy with
        # galaxy velocity dispersion, intrumental line-spread-function,
        # accounting for library resolution and adding emission lines in again
        # more precisely.

        if obs['spectrum'] is not None:
            wobs = wave / (1 + zobs)
            # Smooth in the restframe (galaxy velocity dispersion)
            # Note that we only smooth the part that matters for the output spectrum.
            wobs_velsmooth, spec_velsmooth = smooth_galaxy(wobs, spectrum, veldisp, outwave=outwave)
        
            # Instrumental resolution and wavelength solution
            # get wavelength calibration
            wavecal = self.params.get('wavecal_coeffs', None)
            wobs, sobs = smooth_instrument(wobs_velsmooth, spec_velsmooth,
                                           zobs, instrument_lsf, library_lsf, wavecal)

            if params.get('add_neb_emission', False):
                sobs += calculate_linespec(wobs, eline_lambda * (1 + zobs), elines,
                                           nebveldisp, instrument_lsf)
            if outwave is None:
                outwave = wave
            spec = np.interp(outwave, wobs, sobs)

        # unit conversion (output is maggies for both spectra and photometry)
        # Mass normalization
        mass = np.sum(self.params.get('mass', 1.0))
        if np.all(self.params.get('mass_units', 'mformed') == 'mstar'):
            # Convert input normalization units from current stellar mass to mass formed
            mass /= mfrac

        return spec * mass * ldim / jansky_cgs / 3631, phot * mass * ldim, mfrac


def calculate_linespec(wave_array, line_wavelengths, line_luminosities, line_dispersions,
                       instrument_lsf=None):
    """If you pass `instrument_lsf` then we are assuming that the
    `line_wavelengths` (and `wave_array`) are in the observed frame.
    """
    sigma = line_dispersions / ckms * line_wavelengths
    if instrument_lsf is not None:
        sigma += instrument_lsf(line_wavelengths)
    linespec = gauss(wave_array, line_wavelengths, line_luminosities, sigma)
    return linespec


def smooth_galaxy(wobs, spectrum, veldisp, outwave=outwave, nsigma_pad=10):
    """Smooth the observed frame spectrum by the restframe velocity dispersion
    of the stellar population.
    """
    if veldisp <= 0:
        return wobs, spectrum
    # do padding
    wlim = np.array([outwave.min(), outwave.max()])
    wlim *= (1 + nsigma_pad * np.array([-1, 1]))
    # do smoothing
    wave, spec = smoothing.smoothspec(wobs, spectrum, resolution=veldisp,
                                      smoothtype="vel", fftsmooth=True,
                                      min_wave_smooth=wlim[0], max_wave_smooth=wmin[1])
    return wave, spec


def smooth_instrument(wavecal):
    """Apply the instrumental line-spread function to the observed frame
    spectrum accounting for the restframe library resolution.
    """
    

def calculate_zobs(vpec, zcosmo):
    zobs = vpec / c * (1 + zcosmo) + zcosmo
    return zobs

def distance_dimming(zcosmo=0.0, lumdist=None, cosmology=WMAP9):
    """Factor to go from lsun/Hz to erg/s/cm^2/Hz
    includes cosmological (1+z)
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
    """
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
    

def gauss(x, mu, A, sigma):
    """Lay down mutiple gaussians on the x-axis.
    """
    mu, A, sigma = np.atleast_2d(mu), np.atleast_2d(A), np.atleast_2d(sigma)
    val = (A / (sigma * np.sqrt(np.pi * 2)) *
           np.exp(-(x[:, None] - mu)**2 / (2 * sigma**2)))
    return val.sum(axis=-1)

  
