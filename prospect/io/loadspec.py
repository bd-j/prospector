import sys
import numpy as np
try:
    import astropy.io.fits as pyfits
except(ImportError):
    import pyfits

lsun, pc = 3.846e33, 3.085677581467192e18 #in cgs
to_cgs = lsun/10**( np.log10(4.0*np.pi)+2*np.log10(pc*10) )

def query_phatcat(objname, phottable='data/f2_apcanfinal_6phot_v2.fits',
                  crosstable=None,
                  filtcols=['275','336','475','814','110','160'],
                  **extras):
    
    """
    Read LCJ's catalog for a certain object and return the magnitudes
    and their uncertainties. Can take either AP numbers (starting with
    'AP') or ALTIDs.
    """
    print(phottable, crosstable, objname)
    ap = pyfits.getdata(phottable, 1)
    if objname[0:2].upper() == 'AP':
        objname = int(objname[2:])
        ind = (ap['id'] == objname)
    else:
        if crosstable is None:
            crosstable = phottable.replace('canfinal_6phot_v2', 'match_known')
        cross = pyfits.getdata(crosstable)
        ind = (cross['altid'] == objname)
        ind = (ap['id'] == cross[ind]['id'][0])
    
    dat = ap[ind][0]
    mags = np.array([dat['MAG'+f] for f in filtcols]).flatten()
    mags_unc = np.array([dat['SIG'+f] for f in filtcols]).flatten()
    flags = ap[ind]['id'] #return the ap number
    
    return mags, mags_unc, flags
        

def load_obs_mmt(filename=None, objname=None, #dist = 1e-5, vel = 0.0,
                  wlo=3750., whi=7200., verbose=False,
                  phottable='data/f2_apcanfinal_6phot_v2.fits',
                  **kwargs):
    """
    Read one of Caldwell's MMT spectra and find the matching PHAT
    photometry, return a dictionary containing the observations.
    """
    from sedpy import observate

    obs ={}

    ####### SPECTRUM #######
    if verbose:
        print('Loading data from {0}'.format(filename))

    scale = 1e0 #testing
    #fluxconv = np.pi * 4. * (dist * 1e6 * pc)**2 / lsun #erg/s/AA/cm^2 to L_sun/AA
    fluxconv =  1.0#5.0e-20 * scale #approximate counts to erg/s/AA/cm^2
    #redshift = 0.0 #vel / 2.998e8
    dat = np.squeeze(pyfits.getdata(filename))
    hdr = pyfits.getheader(filename)
    
    crpix = (hdr['CRPIX1'] -1) #convert from FITS to numpy indexing
    try:
        cd = hdr['CDELT1']
    except (KeyError):
        cd = hdr['CD1_1']

    obs['wavelength'] = (np.arange(dat.shape[1]) - crpix) * cd + hdr['CRVAL1']
    obs['spectrum'] = dat[0,:] * fluxconv
    obs['unc'] = np.sqrt(dat[1,:]) * fluxconv
    
    #Masking.  should move to a function that reads a mask definition file
    #one should really never mask in the rest frame - that should be modeled!
    obs['mask'] =  ((obs['wavelength'] >= wlo ) & (obs['wavelength'] <= whi))
    obs['mask'] = obs['mask'] & ((obs['wavelength'] <= 5570) |
                                 (obs['wavelength'] >= 5590)) #mask OI sky line
    obs['mask'] = obs['mask'] & ((obs['wavelength'] <= 6170) |
                                 (obs['wavelength'] >= 6180)) #mask...something.

    #obs['wavelength'] /= (1.0 + redshift)

    ######## PHOTOMETRY ########
    if verbose:
        print('Loading mags from {0} for {1}'.format(phottable, objname))
    mags, mags_unc, flag = query_phatcat(objname, phottable = phottable, **kwargs)
    
    obs['filters'] = observate.load_filters(['wfc3_uvis_'+b.lower() for b in
                                             ["F275W", "F336W", "F475W", "F814W"]] +
                                             ['wfc3_ir_'+b.lower() for b in
                                              ["F110W", "F160W"]])
    obs['maggies'] = 10**(-0.4 * (mags -
                                  np.array([f.ab_to_vega for f in obs['filters']]) -
                                  2.5*np.log10(scale) ))
    obs['maggies_unc'] = mags_unc * obs['maggies'] / 1.086

    return obs

def load_obs_lris(filename=None, objname=None, #dist = 1e-5, vel = 0.0,
                  wlo=3550., whi=5500., verbose=False,
                  phottable='data/f2_apcanfinal_6phot_v2.fits',
                  **kwargs):
    """
    Read one of the Keck LRIS spectra and find the matching PHAT
    photometry, return a dictionary containing the observations.
    """
    from sedpy import observate

    obs ={}
    
    ####### SPECTRUM #######
    if verbose:
        print('Loading data from {0}'.format())

    #fluxconv = np.pi * 4. * (dist * 1e6 * pc)**2 / lsun #erg/s/AA/cm^2 to L_sun/AA
    fluxconv = 1.0
    scale = 1e0 #testing
    #redshift = vel / 2.998e8
    dat = pyfits.getdata(filename)
    hdr = pyfits.getheader(filename)
    
    obs['wavelength'] = dat[0]['wave_opt']
    obs['spectrum'] = dat[0]['spec']
    obs['unc'] = 1./np.sqrt(dat[0]['ivar'])
    #masking
    obs['mask'] =  ((obs['wavelength'] >= wlo ) & (obs['wavelength'] <= whi))
    #obs['wavelength'] /= (1.0 + redshift)
    

    ######## PHOTOMETRY ######
    if verbose:
        print('Loading mags from {0} for {1}'.format(phottable, objname))
    mags, mags_unc, flag = query_phatcat(objname, phottable = phottable, **kwargs)
     
    obs['filters'] = observate.load_filters(['wfc3_uvis_'+b.lower() for b in
                                             ["F275W", "F336W", "F475W", "F814W"]] +
                                             ['wfc3_ir_'+b.lower() for b in
                                              ["F110W", "F160W"]])
    obs['maggies'] = 10**(-0.4 * (mags -
                                  np.array([f.ab_to_vega for f in obs['filters']]) -
                                  2.5*np.log10(scale) ))
    obs['maggies_unc'] = mags_unc * obs['maggies'] / 1.086

    return obs

def load_obs_3dhst(filename, objnum):
    """Load a 3D-HST data file and choose a particular object.  Unfinished.
    """
    obs ={}
    with open(filename, 'r') as f:
        hdr = f.readline().split()
    dat = np.loadtxt(filename, comments = '#',
                     dtype = np.dtype([(n, np.float) for n in hdr[1:]]))

    flux_fields = [f for f in dat.dtype.names if f[0:2] == 'f_']
    unc_fields = [f for f in dat.dtype.names if f[0:2] == 'e_']
    filters = [f[2:] for f in flux_fields]
    
    mags = -2.5*np.log10(dat[flux_fields])
    
    return obs
