import numpy as np

try:
    from sedpy.observate import getSED
except(ImportError):
    pass

__all__ = ["BlackBodyDustBasis"]

# cgs constants
from .constants import lsun, pc, kboltz, hplanck
lightspeed = 29979245800.0


class BlackBodyDustBasis(object):
    """
    """
    def __init__(self, **kwargs):
        self.dust_parlist = ['mass', 'T', 'beta', 'kappa0', 'lambda0']
        self.params = {}
        self.params.update(**kwargs)
        self.default_wave = np.arange(1000) # in microns

    def get_spectrum(self, outwave=None, filters=None, **params):
        """Given a params dictionary, generate spectroscopy, photometry and any
        extras (e.g. stellar mass).

        :param outwave:
            The output wavelength vector.

        :param filters:
            A list of sedpy filter objects.

        :param **params:
            Keywords forming the parameter set.

        :returns spec:
            The restframe spectrum in units of erg/s/cm^2/AA

        :returns phot:
            The apparent (redshifted) maggies in each of the filters.

        :returns extras:
            A list of None type objects, only included for consistency with the
            SedModel class.
        """
        self.params.update(**params)
        if outwave is None:
            outwave = self.default_wave
        # Loop over number of MBBs
        ncomp = len(self.params['mass'])
        seds = [self.one_sed(icomp=ic, wave=outwave, filters=filters)
                for ic in range(ncomp)]
        # sum the components
        spec = np.sum([s[0] for s in seds], axis=0)
        maggies = np.sum([s[1] for s in seds], axis=0)
        extra = [s[2] for s in seds]

        norm = self.normalization()
        spec, maggies = norm * spec, norm * maggies
        return spec, maggies, extra
        
    def one_sed(self, icomp=0, wave=None, filters=None, **extras):
        """Pull out individual component parameters from the param dictionary
        and generate spectra for those components
        """
        cpars = {}
        for k in self.dust_parlist:
            try:
                cpars[k] = np.squeeze(self.params[k][icomp])
            except(IndexError, TypeError):
                cpars[k] = np.squeeze(self.params[k])

        spec = cpars['mass'] * modified_BB(wave, **cpars)
        phot = 10**(-0.4 * getSED(wave*1e4, spec, filters))
        return spec, phot, None

    def normalization(self):
        """This method computes the normalization (due do distance dimming,
        unit conversions, etc.) based on the content of the params dictionary.
        """
        return 1


def modified_BB(wave, T=20, beta=2.0, kappa0=1.92, lambda0=350, **extras):
    """Return a modified blackbody.

    the normalization of the emissivity curve can be given as kappa0 and
    lambda0 in units of cm^2/g and microns, default = (1.92, 350).  Ouput units
    are erg/s/micron/g.
    """
    term = (lambda0 / wave)**beta
    return planck(wave, T=T, **extras) * term * kappa0
    

def planck(wave, T=20.0, **extras):
    """Return planck function B_lambda (erg/s/micron) for a given T (in Kelvin) and
    wave (in microns)
    """
    # Return B_lambda in erg/s/micron
    w = wave * 1e4 #convert from microns to cm
    conv = 2 * hplank * lightspeed**2 / w**5 / 1e4
    denom = (np.exp(hplanck * lightspeed / (kboltz * T)) - 1)
    return conv / denom
