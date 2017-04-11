# Useful constants
lsun = 3.846e33
pc = 3.085677581467192e18  # in cm
lightspeed = 2.998e18  # AA/s
ckms = 2.998e5 # km/s
jansky_mks = 1e-26
# value to go from L_sun/AA to erg/s/cm^2/AA at 10pc
to_cgs_at10pc = lsun/(4.0 * np.pi * (pc*10)**2)

from astropy.cosmology import WMAP9

class sbasis(object):


    def get_spectrum_josh(self, outwave=None, filters=None, peraa=False, **params):
        raise(NotImplementedError)
    
    def get_spectrum(self, outwave=None, filters=None, peraa=False, **params):

        # Spectrum in Lsun/Hz per solar mass formed, restframe
        wave, spectrum, mfrac = self.get_galaxy_spectrum(**params)
 
        # get different redshifts
        cosmo = params.get('cosmology', WMAP9)
        zcosmo = params.get('zred', 0.0)
        zpeculiar = params.get('zpec', 0.0)
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
                                  zcosmo=zcosmo, zpec=zpeculiar,
                                  lnwavegrid=lnwavegrid)
        else:
            phot = 0.0


        # This was the photometry part.  Still have to do spectroscopy with
        # galaxy velocity dispersion, intrumental line-spread-function,
        # accounting for library resolution and adding emission lines in again
        # more precisely.
            
        # get wavelength calibration
        wavecal = self.params.get('wavecal_coeffs', None)

        # Smooth in the restframe (galaxy velocity dispersion)
        # Note that we only smooth the part that matters for the output spectrum.
        # That is, we will assume the velocity dispersion does not affect the phtometry
        # UUUghhhh
        smwave, smspec = smooth_galaxy(wave, spectrum, library_lsf, outwave=w)

        # Instrumental resolution and wavelength solution
        wobs, sobs = smooth_instrument(smwave * (1 + zcosmo), smspec * ldim,
                                       zpec=zpeculiar, zcosmo=zcosmo,
                                       instrument_lsf, wavecal)

        # unit conversion

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


def get_photometry(wave, fnu, filters, zcosmo=0.0, zpeculiar=0, lnwavegrid=None):
    
    wa = wave * (1 + zcosmo) * (1 + zpeculiar)

    if lnwavegrid is not None:
        spec = np.interp(lnwavegrid, wa, fnu)
        wa = lnwavegrid
        gridded = True
    else:
        spec = fnu
        gridded = False

    mags = getSED(wa, lightspeed/wa**2 * spec, filters, gridded=gridded)
    phot = np.atleast_1d(10**(-0.4 * mags))
    return phot   
    

def gauss(x, mu, A, sigma):
    """Lay down mutiple gaussians on the x-axis.
    """
    mu, A, sigma = np.atleast_2d(mu), np.atleast_2d(A), np.atleast_2d(sigma)
    val = (A / (sigma * np.sqrt(np.pi * 2)) *
           np.exp(-(x[:, None] - mu)**2 / (2 * sigma**2)))
    return val.sum(axis=-1)

  
