
import numpy as np

from .ssp_basis import FastStepBasis
from .fake_fsps import add_dust, add_igm

try:
    import cue
except:
    pass


__all__ = ["NebSSPBasis"]


class NebSSPBasis(FastStepBasis):

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
              "add_neb_emission", "add_neb_continuum", "neblemlineinspec",
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
        self._line_specific_luminosity = lines
        return wave, spec / mtot, self.ssp.stellar_mass / mtot


def get_spectrum(ssp, params, emul, tage=0):
    add_neb = params["add_neb_emission"]
    use_stars = params["use_stellar_ionizing"]
    ewave = ssp.emline_wavelengths
    wave, tspec = ssp.get_spectrum(tage=tage, peraa=False)
    young, old = ssp._csp_young_old
    csps = [young, old] # could combine with previous line
    lines = []
    for spec in csps:
        if add_neb:
            if use_stars:
                ion_params = fit_log_linear_ionparam(wave, spec)
                params.update(**ion_params)
            line_prediction = emul.predict_lines(**params)
            lines.append(line_prediction)
            spec += emul.predict_cont(wave, **params)
        else:
            lines.append(np.zeros_like(ewave))

    sspec, lines = add_dust(wave, csps, ewave, lines, **params)
    sspec = add_igm(wave, sspec, **params)
    return wave, sspec, lines
