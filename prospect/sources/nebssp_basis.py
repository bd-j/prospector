### SSP for cue
import numpy as np

from .galaxy_basis import FastStepBasis, CSPSpecBasis
from .fake_fsps import add_dust, add_igm, idx

try:
    import cue
except:
    pass


__all__ = ["NebStepBasis", "NebCSPBasis"]


class NebStepBasis(FastStepBasis):

    """This is a class that wraps the fsps.StellarPopulation object, which is
    used for producing SSPs.  The ``fsps.StellarPopulation`` object is accessed
    as ``SSPBasis().ssp``.

    This class allows for the custom calculation of relative SSP weights (by
    overriding ``all_ssp_weights``) to produce spectra from arbitrary composite
    SFHs. Alternatively, the entire ``get_galaxy_spectrum`` method can be
    overridden to produce a galaxy spectrum in some other way, for example
    taking advantage of weight calculations within FSPS for tabular SFHs or for
    parameteric SFHs.

    The base implementation here produces an SSP interpolated to the age given
    by ``tage``, with initial mass given by ``mass``.  However, this is much
    slower than letting FSPS calculate the weights, as implemented in
    :py:class:`FastSSPBasis`.

    Furthermore, smoothing, redshifting, and filter projections are handled
    outside of FSPS, allowing for fast and more flexible algorithms.

    :param reserved_params:
        These are parameters which have names like the FSPS parameters but will
        not be passed to the StellarPopulation object because we are overriding
        their functionality using (hopefully more efficient) custom algorithms.
    """

    def __init__(self, cue_kwargs={},
                 **kwargs):

        self.emul = cue.Emulator(**cue_kwargs)
        # we do these now
        rp = ["dust1", "dust2", "dust3", "add_dust_emission",
              "add_igm_absorption", "igm_factor",
              "add_neb_emission", "add_neb_continuum", "nebemlineinspec",
              "fagn", "agn_tau"]
        reserved_params = kwargs.pop("reserved_params", []) + rp
        super().__init__(reserved_params=reserved_params, **kwargs)
        for k in ["add_igm_absorption", "add_dust_emission", "add_neb_emission", "nebemlineinspec"]:
            self.ssp.params[k] = False

    def get_galaxy_spectrum(self, **params):
        """Construct the tabular SFH and feed it to the ``ssp``.
        """
        self.update(**params)
        # --- check to make sure agebins have minimum spacing of 1million yrs ---
        #       (this can happen in flex models and will crash FSPS)
        if np.min(np.diff(10**self.params['agebins'])) < 1e6:
            raise ValueError

        mtot = self.params['mass'].sum()
        time, sfr, tmax = self.convert_sfh(self.params['agebins'], self.params['mass'])
        self.ssp.params["sfh"] = 3  # Hack to avoid rewriting the superclass
        self.ssp.set_tabular_sfh(time, sfr)

        wave, spec, lines = get_spectrum(self.ssp, self.params, self.emul, tage=tmax)
        self._line_specific_luminosity = lines/mtot
#         # mimic the "nebemlineinspec" function in FSPS
#         if self.params.get("nebemlineinspec", False):
#             if self.ssp.params["smooth_velocity"] == True:
#                 dlam = self.ssp.emline_wavelengths*self.ssp.params["sigma_smooth"]/2.9979E18*1E13 #smoothing variable is in km/s
#             else:
#                 dlam = self.ssp.params["sigma_smooth"] #smoothing variable is in AA
#             nearest_id = np.searchsorted(wave, self.ssp.emline_wavelengths)
#             neb_res_min = wave[nearest_id]-wave[nearest_id-1]
#             dlam = np.max([dlam,neb_res_min], axis=0)
#             gaussnebarr = [1./np.sqrt(2*np.pi)/dlam[i]*np.exp(-(wave-self.ssp.emline_wavelengths[i])**2/2/dlam[i]**2) \
#             /2.9979E18*self.ssp.emline_wavelengths[i]**2 for i in range(len(lines))]
#             for i in range(len(lines)):
#                 spec += lines[i]*gaussnebarr[i]
        return wave, spec / mtot, self.ssp.stellar_mass / mtot

    
class NebCSPBasis(FastStepBasis):

    """A subclass of :py:class:`SSPBasis` for combinations of N composite
    stellar populations (including single-age populations). The number of
    composite stellar populations is given by the length of the ``"mass"``
    parameter. Other population properties can also be vectors of the same
    length as ``"mass"`` if they are independent for each component.
    """

    def __init__(self, zcontinuous=1, reserved_params=['sigma_smooth'],
                 vactoair_flag=False, compute_vega_mags=False, 
                 cue_kwargs={}, **kwargs):

    # This is a StellarPopulation object from fsps
    self.ssp = fsps.StellarPopulation(compute_vega_mags=compute_vega_mags,
                                      zcontinuous=zcontinuous,
                                      vactoair_flag=vactoair_flag)
    self.emul = cue.Emulator(**cue_kwargs)
    # we do these now
    rp = ["dust1", "dust2", "dust3", "add_dust_emission",
          "add_igm_absorption", "igm_factor",
          "add_neb_emission", "add_neb_continuum", "nebemlineinspec",
          "fagn", "agn_tau"]
    reserved_params = kwargs.pop("reserved_params", []) + rp
    super().__init__(reserved_params=reserved_params, **kwargs)
    for k in ["add_igm_absorption", "add_dust_emission", "add_neb_emission", "nebemlineinspec"]:
        self.ssp.params[k] = False

    def get_galaxy_spectrum(self, **params):
        """Update parameters, then loop over each component getting a spectrum
        for each and sum with appropriate weights.

        :param params:
            A parameter dictionary that gets passed to the ``self.update``
            method and will generally include physical parameters that control
            the stellar population and output spectrum or SED.

        :returns wave:
            Wavelength in angstroms.

        :returns spectrum:
            Spectrum in units of Lsun/Hz/solar masses formed.

        :returns mass_fraction:
            Fraction of the formed stellar mass that still exists.
        """
        self.update(**params)
        spectra, linelum = [], []
        mass = np.atleast_1d(self.params['mass']).copy()
        mfrac = np.zeros_like(mass)
        # Loop over mass components
        for i, m in enumerate(mass):
            self.update_component(i)
            wave, spec, lines = get_spectrum(self.ssp, self.params, self.emul, 
                                             tage=self.ssp.params['tage'], peraa=False)
            spectra.append(spec)
            mfrac[i] = (self.ssp.stellar_mass)
            linelum.append(lines)

        # Convert normalization units from per stellar mass to per mass formed
        if np.all(self.params.get('mass_units', 'mformed') == 'mstar'):
            mass /= mfrac
        spectrum = np.dot(mass, np.array(spectra)) / mass.sum()
        self._line_specific_luminosity = np.dot(mass, np.array(linelum)) / mass.sum()
        mfrac_sum = np.dot(mass, mfrac) / mass.sum()

        return wave, spectrum, mfrac_sum
    

def get_spectrum(ssp, params, emul, tage=0):
    """
    Add the nebular continuum from Cue to the young and old population and do the dust attenuation and igm absorption.
    Also, calculate the line luminoisities from the young csp and old csp and do the dust attenuation and igm absorption.
    The output spec/line luminosity need to be divided by total formed mass to get the specific number.
    :param use_stellar_ionizing:
        If true, fit CSPs and to get the ionizing spectrum parameters, else read from ssp
    """
    add_neb = params["add_neb_emission"]
    use_stars = params["use_stellar_ionizing"]
    ewave = ssp.emline_wavelengths
    wave, tspec = ssp.get_spectrum(tage=tage, peraa=False)
    young, old = ssp._csp_young_old
    csps = [young, old] # could combine with previous line
    lines = []
    mass = np.sum(params.get('mass', 1.0))
    if not add_neb:
        lines = [np.zeros_like(ewave), np.zeros_like(ewave)]
    elif add_neb:
        if not use_stars:
            line_prediction = np.zeros_like(ewave)
            line_prediction[idx] = emul.predict_lines(**params)
            #if ssp.params["sfh"] == 3:
            #    line_prediction /= mass
            lines = [line_prediction, np.zeros_like(ewave)]
            csps[0][wave>=912] += emul.predict_cont(wave[wave>=912], **params)
        elif use_stars:
            for spec in csps:
                ion_params = cue.fit_4loglinear_ionparam(wave, spec)
                params.update(**ion_params)
                line_prediction = np.zeros_like(ewave)
                line_prediction[idx] = emul.predict_lines(**params)
                #if ssp.params["sfh"] == 3:
                #    line_prediction /= mass
                lines.append(line_prediction)
                spec[wave>=912] += emul.predict_cont(wave[wave>=912], **params)
        else:
            raise KeyError('No "use_stellar_ionizing" in model')

    sspec, lines = add_dust(wave, csps, ewave, lines, **params)
    sspec = add_igm(wave, sspec, **params)
    return wave, sspec, lines
