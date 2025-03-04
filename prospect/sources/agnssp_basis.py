import os
import numpy as np
from astropy.cosmology import WMAP9 as cosmo

from .galaxy_basis import FastStepBasis
from .fake_fsps import add_dust, add_igm, agn_torus


__all__ = ["AGNSSPBasis"]


class AGNSSPBasis(FastStepBasis):
    """AGN acceration disk continuum model.
    For details, see Wang et al. 2024: https://ui.adsabs.harvard.edu/abs/2024arXiv240302304W/abstract
    Note that this only works with non-param SFH.
    """

    def __init__(self, 
                 **kwargs):

        rp = []
        reserved_params = kwargs.pop("reserved_params", []) + rp
        super().__init__(reserved_params=reserved_params, **kwargs)

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

        blob = get_spectrum(self.ssp, self.params, tage=tmax)
        self._line_specific_luminosity = blob['slines']/mtot

        other = {'gal_tot_nodust': blob['gal_tot_nodust']/mtot, 'agn_nodust': blob['agn_nodust']/mtot}

        return blob['wave'], blob['sspec']/mtot, blob['aspec']/mtot, blob['aspec_torus']/mtot, self.ssp.stellar_mass/mtot, other


    def get_galaxy_elines(self):

        ewave = self.ssp.emline_wavelengths
        elum = getattr(self, "_line_specific_luminosity", None)

        if elum is None:
            elum = self.ssp.emline_luminosity.copy()
            if elum.ndim > 1:
                elum = elum[0]
            if self.ssp.params["sfh"] == 3:
                # tabular sfh
                mass = np.sum(self.params.get('mass', 1.0))
                elum /= mass

        return ewave, elum


def temple_template(ssp):
    """a 3-piece power law adapted from Temple+21
    """
    agn_data_dir = os.path.join(os.path.dirname(__file__), 'agn_data')
    w, temp = np.loadtxt( os.path.join(agn_data_dir, 'temple.txt'), unpack=True)

    return np.interp(ssp.wavelengths, w, temp)


def get_spectrum(ssp, params, tage=0):

    add_agn_bbb_cont = params.get("add_agn_bbb_cont", False)
    add_agn_bbb_elines = params.get("add_agn_bbb_elines", False) # placeholder
    add_agn_dust = params.get("add_agn_dust", False)
    tie_agn_torus_to_disk = params.get("tie_agn_torus_to_disk", False)
    
    ewave = ssp.emline_wavelengths
    wave, tspec = ssp.get_spectrum(tage=tage, peraa=False)
    young, old = ssp._csp_young_old # no dust, no igm
    csps = [young, old]
    gal_tot = young + old # only used for scaling the AGN spectrum

    agn_csps = []
    idx_1500A = (wave>=5500-20) & (wave<=5500+20)
    agn_bbb_cont = np.zeros_like(wave)

    # AGN acceration disk
    if not add_agn_bbb_cont:
        agn_csps = [np.zeros_like(agn_bbb_cont), np.zeros_like(agn_bbb_cont)]
    elif add_agn_bbb_cont:
        # AGN spectrum
        norm_agn_bbb_cont = temple_template(ssp)
        scale = np.median(norm_agn_bbb_cont[idx_1500A]/gal_tot[idx_1500A])
        norm_agn_bbb_cont /= scale

        agn_bbb_cont = norm_agn_bbb_cont * params.get("fagn_bbb", 0.0)
        agn_csps = [agn_bbb_cont, np.zeros_like(agn_bbb_cont)]
    else:
        raise KeyError('No "add_agn_bbb_cont" in model')
    
    # alines are simply a placeholder for now
    dust_type = params['dust_type']
    dust4_type = params.get('dust4_type', 0)
    d2 = params.get('dust2', 0)
    dust_index = params['dust_index']

    fake_elines = [np.zeros_like(ewave), np.zeros_like(ewave)]
    slines = np.zeros_like(ewave)

    if dust4_type == 0:
        # no dust4
        d4 = 0.0
        d4_index = 0.0

    elif dust4_type == 1:
        # galaxy & AGN both reddened by dust2, whereas the AGN receives additional reddening from dust4
        d4 = params.get('dust4', 0)
        d4_index = params.get('dust4_index', -0.8)

    elif dust4_type == 2:
        # galaxy & AGN are reddened separately. the AGN only receives reddening from dust4
        d4 = params.get('dust4', 0)
        d4_index = params.get('dust4_index', -0.8)
        raise(NotImplementedError)

    else:
        d4 = 0.0
        d4_index = 0.0
        # raise KeyError('No "dust4_type" in model')

    # AGN acceration disk
    aspec, alines = add_dust(wave, agn_csps, ewave, fake_elines, # XXX 
                             dust_type=dust_type, dust_index=dust_index, dust2=d2,
                             dust1_index=0.0, dust1=0.0,
                             dust4_type=dust4_type, dust4_index=d4_index, dust4=d4,
                             frac_nodust=0,frac_obrun=0
                             )
    
    # AGN torus
    alines_torus = np.zeros_like(alines)
    aspec_torus = agn_torus(wave, params.get('agn_tau', 50))
    # scale AGN torus, following the definition in fsps
    aspec_torus = 10**ssp.log_lbol * params.get('fagn', 0) * aspec_torus

    aspec = add_igm(wave, aspec, **params)
    aspec_torus = add_igm(wave, aspec_torus, **params)

    blob = {'wave':wave, 'sspec':tspec, 'slines':slines, 'aspec':aspec, 'alines':alines, 
            'aspec_torus':aspec_torus, 'alines_torus':alines_torus,
            'gal_tot_nodust':gal_tot, 'agn_nodust':agn_csps[0]}
    
    return blob

