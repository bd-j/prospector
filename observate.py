#PYTHON MODULE FOR STORING FILTER INFORMATION
#AND TOOLS FOR PROJECTING THEM ONTO SPECTRA
#ALSO INCLUDES TOOLS FOR CONVOLVING AND
#REDSHIFTING SPECTRA
#Assumed input units are erg/s/cm^2/AA and AA

#To Do: -- Add methods for obtaining magnitude uncertainties
#          given an error spectrum along with the source spectrum
#       -- Add index measurements
#       -- Test the broadening functions for accuracy and speed.
#       -- Add some redshifting methods/classes


import numpy as np
import os
import matplotlib.pyplot as plt
import pyfits
import yanny

##Load useful reference spectra######
lightspeed=2.998E18 #AA/s
vega_file=os.getenv('hrdspy')+'/data/alpha_lyr_stis_005.fits'
#this file should be in AA and erg/s/cm^2/AA
if os.path.isfile( vega_file ):
    fits = pyfits.open( vega_file )
    vega = np.column_stack( (fits[1].data.field('WAVELENGTH'),fits[1].data.field('FLUX')) )
    fits.close()
else:
    raise ValueError('Could not find Vega spectrum at %s',vega_file)
rat = (1.0/(3600*180/np.pi*10))**2.0 # conversion to d=10 pc from 1 AU
solar_file = os.getenv('hrdspy')+'/data/sun_kurucz93.fits'
#this file should be in AA and erg/s/cm^2/AA at 1AU
if os.path.isfile( solar_file ):
    fits = pyfits.open( solar_file )
    solar = np.column_stack( (fits[1].data.field('WAVELENGTH'),fits[1].data.field('FLUX')*rat) )
    fits.close()
else:
    raise ValueError('Could not find Solar spectrum at %s', solar_file)


class Filter(object):
    """This class operates on filter transmission files.  It reads SDSS-style yanny
    files containing filter transmissions (these are easy to create) and determines
    a number of useful filter files.  Methods are provided to convolve a source
    spectrum with the filter and return the magnitude."""

    ab_gnu=3.631e-20   #AB reference spctrum in erg/s/cm^2/Hz
    npts=0
    
    def __init__(self, kname='sdss_r0', nick=None):
        """Constructor"""
        self.name = kname
        if nick is None :
            self.nick = kname
        else:
            self.nick=nick

        self.filename = os.getenv('hrdspy')+'/data/filters/'+kname+'.par'
        if type( self.filename ) == type( '' ):
            if not os.path.isfile( self.filename ): raise ValueError( 'Filter transmission file %s does not exist!' %self.filename )
            self.loadKFilter(self.filename)

    def loadKFilter(self,filename):
        """loadKFilter
        Read a filter in kcorrect (yanny) format and populate the
        wavelength and transmission arrays.  Then determine a
        number of filter properties and store in the object."""

        #This should be replaced with the sdsspy yanny file readers # done, kept here in case reversion required
        #f=open(filename,'rU')
        #wave=[]
        #trans=[]
        #for line in f:
        #    cols=line.split()
        #    if len(cols) > 2 and cols[0].find('KFILTER') > -1:
        #        wave.append(float(cols[1]))
        #        if cols[0].find('SDSS') > -1:
        #            trans.append(float(cols[4])) #use airmass=1.3 passband.  HACKY
        #        else:
        #            trans.append(float(cols[2]))
        #f.close()

        ff = yanny.read(filename,one=True)
        wave = ff['lambda']
        trans = ff['pass']
        #clean negatives, NaNs, and Infs, then sort, then store
        ind=np.where(np.logical_and( np.isfinite(trans), (trans >= 0.0) ))[0]
        order = wave[ind].argsort()
        self.npts = ind.shape[0]
        self.wavelength = wave[ind[order]]
        self.transmission = trans[ind[order]]

        self.getProperties()

        
    def getProperties(self):
        """getProperties
        Determine a number of properties of the filter and store them in the object.
        These properties include several 'effective' wavelength definitions and several
        width definitions, as well as the in-band absolute AB solar magnitude, the Vega and
        AB reference detector signal, and the conversion between AB and Vega magnitudes. """

        i0 = np.trapz(self.transmission*np.log(self.wavelength),np.log(self.wavelength))
        i1 = np.trapz(self.transmission,np.log(self.wavelength))
        i2 = np.trapz(self.transmission*self.wavelength,self.wavelength)
        i3 = np.trapz(self.transmission,self.wavelength)
        i4 = np.trapz(self.transmission * ( np.log(self.wavelength) )**2.0,np.log(self.wavelength))
        
        self.wave_effective = np.exp(i0/i1)
        self.wave_pivot = np.sqrt(i2/i1)
        self.wave_mean = self.wave_effective
        self.wave_average = i2/i3

        self.gauss_width = (i4/i1)**(0.5)
        self.effective_width = 2.0*np.sqrt( 2.0*np.log(2.0) )*self.gauss_width*self.wave_mean
        self.rectangular_width = i3/self.transmission.max()
        #self.norm           = np.trapz(transmission,wavelength)

        self.ab_counts = self.objCounts( self.wavelength,self.ab_gnu*lightspeed/(self.wavelength**2) )
        self.vega_counts = self.objCounts(vega[:,0],vega[:,1]) 
        self.ab_to_vega = -2.5*np.log10(self.ab_counts/self.vega_counts)
        self.solar_mag = self.ABMag(solar[:,0],solar[:,1])

    def display(self):
        """display
        Plot the filter transmission curve"""
        if self.npts > 0:
            plt.plot(self.wavelength,self.transmission)
            plt.title(self.name)

    def objCounts(self, sourcewave, sourceflux, sourceflux_unc=0):
        """getCounts: Convolve source spectrum with filter and return the detector signal
               sourcewave - spectrum wavelength (in AA), ndarray of shape (nwave)
               sourceflux -  associated flux (in erg/s/cm^2/AA), ndarray of shape (nspec,nwave)
               output: float Detector signals (nspec)"""


        #interpolate filter transmission to source spectrum
        newtrans = np.interp(sourcewave, self.wavelength,self.transmission, 
                                left=0.,right=0.)
            #print(self.name,sourcewave.shape,self.wavelength.shape,newtrans.shape)
        
        #integrate lambda*f_lambda*R
	if True in (newtrans > 0.):
            ind = np.where(newtrans > 0.)
            ind=ind[0]
            counts = np.trapz(sourcewave[ind]*newtrans[ind]*sourceflux[...,ind], sourcewave[ind],axis=-1)
            #if  np.isinf(counts).any() : print(self.name, "Warn for inf value")
            return np.squeeze(counts)
	else:
            return 0.

    def ABMag(self,sourcewave,sourceflux,sourceflux_unc=0):
        """ABMag: Convolve source spectrum  with filter and return the AB magnitude
        """
        return 0-2.5*np.log10(self.objCounts(sourcewave,sourceflux)/self.ab_counts)

    def vegaMag(self,sourcewave,sourceflux,sourceflux_unc=0):
        """vegaMag: Convolve source spectrum  with filter and return the Vega magnitude
        """
        return 0-2.5*np.log10(self.objCounts(sourcewave,sourceflux)/self.vega_counts)        


###Useful utilities#####

def load_filters(filternamelist):
    """Given a list of filter names, this method returns a list of Filter objects"""
    filterlist=[]
    for f in filternamelist:
        print(f)
        filterlist.append(Filter(f))
    return filterlist

def getSED(sourcewave,sourceflux,filterlist):
    """Takes wavelength array [ndarray of shape (nwave)], a flux
    array [ndarray of shape (nsource,nwave)] and list of Filter objects
    and returns the AB mag SED [ndarray of shape (nsource,nfilter)]"""

    sourceflux=np.atleast_2d(sourceflux)
    sedshape = [sourceflux.shape[0], len(filterlist)]
    sed = np.zeros(sedshape)
    for i,f in enumerate(filterlist):
        sed[:,i]=f.ABMag(sourcewave,sourceflux)
    return np.squeeze(sed)

def filter_dict(filterlist):
    fdict = {}
    for i,f in enumerate(filterlist):
        fdict[f.nick] = i
    return fdict

###Routines for spectra######

def Lbol(wave,spec,wave_min=90,wave_max = 1e6):
    """assumes wavelength varies along last axis of spec"""
    inds=np.where(np.logical_and(wave < wave_max, wave >= wave_min))
    return np.trapz(spec[...,inds[0]],wave[inds])


def air2vac(wave):
    """Convert from in-air wavelengths to vacuum wavelengths.  Based on Allen's Astrophysical Quantities"""
    ss=1E4/wave
    wv=wave*(1+6.4328e-5 + 2.94981e-2/(146-ss^2) + 2.5540e-4/(41-ss^2))
    return wv

#def velBroaden(self,sourcewave,sourceflux,sigma,sigma0=0,outwave=-1):
#    sigma=np.sqrt(sigma**2-sigma0**2)
#    K=(sigma*sqrt(2*!PI))^(-1.)
#    wr=outwave#(1/wave)
#    v=c*(wr-1.)
#    ee=exp(0.-v^2./(2.*sigma^2.))
#    dv=(v[1:nw-1,*]-v[0:nw-2,*])
#    broadflux=total(dv*flux[0:nw-2,*]*ee[0:nw-2,*],2)
#    return K*broadflux #/1.553??????

#def waveBroaden(self,sourcewave,sourceflux,fwhm,fwhm_spec=0,outwave=-1):
#    sigma=sqrt(fwhm^2.-fwhm_spec^2)/2.3548
#    K=(sigma*sqrt(2.*!PI))^(-1.)
#    for iw in range(len(outwave)):
#        dl=(outwave[iw]-wave)**2.
#        this=np.where(dl < length*sigma**2.,count)
#        if count > 0:
#            ee=exp(-0.5*dl[this]/sigma^2)
#            broadflux[iw]=int_tabulated(wave[this],flux[this]*ee)
#    return,broadflux

#def redshift(sourcewave,sourceflux, z):

def selftest():
    """Compare to the values obtained from the K-correct code
    (which uses a slightly different Vega spectrum)"""

    filternames=['galex_FUV','sdss_u0','sdss_g0','sdss_r0','sdss_i0','spitzer_irac_ch2']
    weff_kcorr=[1528.0,3546.0,4669.6,6156.25,7471.57,44826.]
    msun_kcorr=[18.8462,6.38989,5.12388,4.64505,4.53257,6.56205]
    ab2vega_kcorr=[2.3457,0.932765,-0.0857,0.155485,0.369598,3.2687]

    filterlist=loadFilters(filternames)
    for i in range(len(filterlist)):
        print(filterlist[i].wave_effective, filterlist[i].solar_mag, filterlist[i].ab_to_vega)
        assert abs(filterlist[i].wave_effective-weff_kcorr[i]) < weff_kcorr[i]*0.01
        assert abs(filterlist[i].solar_mag-msun_kcorr[i]) < 0.05
        #assert abs(filterlist[i].ab_to_vega+ab2vega_kcorr[i]) < 0.05

