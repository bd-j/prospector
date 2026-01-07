import os
import numpy as np
from scipy.interpolate import interp1d

__all__ = ["add_dust", "add_igm", "agn_torus",
           "compute_absorbed_luminosity", "add_dust_with_absorption_tracking"]

# Speed of light in Angstrom/s
C_AA = 2.998e18


def add_dust(wave,specs,line_waves,lines,dust_type=0,dust_index=0.0,dust2=0.0,dust1_index=0.0,dust1=0.0,
             frac_nodust=0,frac_obrun=0,
             dust4_type=0,dust4_index=0.0,dust4=0.0,**kwargs):

    d1_curve = attenuate(specs[0], wave, dust_type=dust_type, dust_index=dust_index, dust2=0, dust1_index=dust1_index, dust1=dust1, dust4=0.0)
    cspi = specs[0]*d1_curve*(1-frac_obrun) + specs[0]*frac_obrun + specs[1]

    diff_dust = attenuate(specs[1], wave, dust_type=dust_type, dust_index=dust_index, dust2=dust2, dust1=0.0,
                          dust4_type=dust4_type, dust4_index=dust4_index, dust4=dust4)
    specdust = (1-frac_nodust) * cspi*diff_dust + cspi*frac_nodust


    # emission lines
    d1_curve = attenuate(lines[0], line_waves, dust_type=dust_type, dust_index=dust_index, dust2=0, dust1_index=dust1_index, dust1=dust1, dust4=0.0)
    ncspi = lines[0]*d1_curve*(1-frac_obrun) + lines[0]*frac_obrun + lines[1]

    diff_dust = attenuate(lines[1], line_waves, dust_type=dust_type, dust_index=dust_index, dust2=dust2, dust1=0.0,
                          dust4_type=dust4_type, dust4_index=dust4_index, dust4=dust4)
    nebdust = ncspi*diff_dust*(1-frac_nodust) + ncspi*frac_nodust

    return specdust, nebdust


def attenuate(spec,lam,dust_type=0,dust_index=0.0,dust2=0.0,dust1_index=0.0,dust1=0.0,
              dust4_type=0,dust4_index=0.0,dust4=0.0):
    """returns F(obs) / F(emitted) for a given attenuation curve + dust1 + dust2
    """

    ### constants from FSPS
    dd63 = 6300.00
    lamv = 5500.0
    dlam = 350.0
    lamuvb = 2175.0

    ### check for out-of-bounds dust type
    if (dust_type < 0) | (dust_type > 6):
        raise ValueError('ATTN_CURVE ERROR: dust_type out of range:{0}'.format(dust_type))

    ### power-law attenuation
    if (dust_type == 0):
        attn_curve = (lam/lamv)**dust_index * dust2

    ### CCM89 extinction curve
    elif (dust_type == 1):
        raise(NotImplementedError)

    ### Calzetti et al. 2000 attenuation
    elif (dust_type == 2):
        #Calzetti curve, below 6300 Angstroms, else no addition
        cal00 = np.zeros_like(lam)
        gt_dd63 = lam > dd63
        le_dd63 = ~gt_dd63
        if gt_dd63.sum() > 0:
            cal00[gt_dd63] = 1.17*( -1.857+1.04*(1e4/lam[gt_dd63]) ) + 1.78
        if le_dd63.sum() > 0:
            cal00[le_dd63]  = 1.17*(-2.156+1.509*(1e4/lam[le_dd63])-\
                              0.198*(1E4/lam[le_dd63])**2 + \
                              0.011*(1E4/lam[le_dd63])**3) + 1.78
        cal00 = cal00/0.44/4.05  # R = 4.05
        cal00 = np.clip(cal00, 0.0, np.inf)  # no negative attenuation

        attn_curve = cal00 * dust2

    ### Witt & Gordon 2000 attenuation
    elif (dust_type == 3):
        raise(NotImplementedError)


    ### Kriek & Conroy 2013 attenuation
    elif (dust_type == 4):
        #Calzetti curve, below 6300 Angstroms, else no addition
        cal00 = np.zeros_like(lam)
        gt_dd63 = lam > dd63
        le_dd63 = ~gt_dd63
        if gt_dd63.sum() > 0:
            cal00[gt_dd63] = 1.17*( -1.857+1.04*(1e4/lam[gt_dd63]) ) + 1.78
        if le_dd63.sum() > 0:
            cal00[le_dd63]  = 1.17*(-2.156+1.509*(1e4/lam[le_dd63])-\
                              0.198*(1E4/lam[le_dd63])**2 + \
                              0.011*(1E4/lam[le_dd63])**3) + 1.78
        cal00 = cal00/0.44/4.05  # R = 4.05
        cal00 = np.clip(cal00, 0.0, np.inf)  # no negative attenuation

        eb = 0.85 - 1.9 * dust_index  #KC13 Eqn 3

        #Drude profile for 2175A bump
        drude = eb*(lam*dlam)**2 / ( (lam**2-lamuvb**2)**2 + (lam*dlam)**2 )

        attn_curve = dust2*(cal00+drude/4.05)*(lam/lamv)**dust_index

    ### Gordon et al. (2003) SMC exctincion
    elif (dust_type == 5):
        raise(NotImplementedError)

    ### Reddy et al. (2015) attenuation
    elif (dust_type == 6):
        reddy = np.zeros_like(lam)

        # see Eqn. 8 in Reddy et al. (2015)
        w1 = np.abs(lam - 1500).argmin()
        w2 = np.abs(lam - 6000).argmin()
        reddy[w1:w2] = -5.726 + 4.004/(lam[w1:w2]/1e4) - 0.525/(lam[w1:w2]/1e4)**2 + \
                       0.029/(lam[w1:w2]/1e4)**3 + 2.505
        reddy[:w1] = reddy[w1]  # constant extrapolation blueward

        w1 = np.abs(lam - 6000).argmin()
        w2 = np.abs(lam - 28500).argmin()
        # note the last term is not in Reddy et al. but was included to make the
        # two functions continuous at 0.6um
        reddy[w1:w2] = -2.672 - 0.010/(lam[w1:w2]/1e4) + 1.532/(lam[w1:w2]/1e4)**2 + \
                       -0.412/(lam[w1:w2]/1e4)**3 + 2.505 - 0.036221981

        # convert k_lam to A_lam/A_V assuming Rv=2.505
        reddy = reddy/2.505

        attn_curve = dust2*reddy

    # -------------- AGN
    ### power-law attenuation
    attn_curve_4 = 0
    if dust4_type != 0:
        attn_curve_4 = (lam/lamv)**dust4_index * dust4

    dust1_ext = np.exp(-dust1*(lam/5500.)**dust1_index)
    dust2_ext = np.exp(-attn_curve)
    dust4_ext = np.exp(-attn_curve_4)

    ext_tot = dust2_ext*dust1_ext *dust4_ext

    return ext_tot


def compute_absorbed_luminosity(wave, intrinsic_spec, attenuated_spec):
    """
    Compute total absorbed luminosity from attenuation.

    The absorbed luminosity is the difference between the intrinsic
    (unattenuated) and attenuated spectra, integrated over frequency.

    Parameters
    ----------
    wave : ndarray
        Wavelength in Angstroms
    intrinsic_spec : ndarray
        Intrinsic (unattenuated) spectrum in L_sun/Hz
    attenuated_spec : ndarray
        Attenuated spectrum in L_sun/Hz

    Returns
    -------
    L_absorbed : float
        Total absorbed luminosity in L_sun
    """
    # Convert wavelength to frequency
    nu = C_AA / wave  # Hz

    # L_absorbed = L_intrinsic - L_observed
    delta_spec = intrinsic_spec - attenuated_spec

    # Integrate over frequency (note: nu decreases as wave increases)
    L_absorbed = -np.trapz(delta_spec, nu)  # L_sun

    return max(0.0, L_absorbed)  # Ensure non-negative


def add_dust_with_absorption_tracking(wave, specs, line_waves, lines,
                                       dust_type=0, dust_index=0.0, dust2=0.0,
                                       dust1_index=0.0, dust1=0.0,
                                       frac_nodust=0, frac_obrun=0,
                                       dust4_type=0, dust4_index=0.0, dust4=0.0,
                                       **kwargs):
    """
    Apply dust attenuation and return both attenuated spectrum and absorbed energy.

    This function wraps add_dust() but additionally computes the total
    luminosity absorbed by dust, which is needed for energy-balance
    dust emission models like DL2014.

    Parameters
    ----------
    wave : ndarray
        Wavelength array in Angstroms
    specs : list of ndarrays
        [young_spec, old_spec] - spectra of young and old stellar populations
    line_waves : ndarray
        Emission line wavelengths
    lines : list of ndarrays
        [young_lines, old_lines] - emission line luminosities
    dust_type : int
        Dust attenuation law type (0-6)
    dust_index : float
        Power-law slope modification
    dust2 : float
        Diffuse dust optical depth at 5500AA
    dust1_index : float
        Birth cloud dust power-law index
    dust1 : float
        Birth cloud dust optical depth
    frac_nodust : float
        Fraction of light escaping without dust
    frac_obrun : float
        Fraction of young stars outside birth cloud
    dust4_type : int
        AGN dust type
    dust4_index : float
        AGN dust power-law index
    dust4 : float
        AGN dust optical depth

    Returns
    -------
    attenuated_spec : ndarray
        Attenuated spectrum in L_sun/Hz
    attenuated_lines : ndarray
        Attenuated emission line luminosities
    L_absorbed : float
        Total absorbed luminosity in L_sun
    """
    # Compute intrinsic spectrum before attenuation
    intrinsic_total = specs[0] + specs[1]

    # Apply attenuation using existing function
    attenuated_spec, attenuated_lines = add_dust(
        wave, specs, line_waves, lines,
        dust_type=dust_type, dust_index=dust_index, dust2=dust2,
        dust1_index=dust1_index, dust1=dust1,
        frac_nodust=frac_nodust, frac_obrun=frac_obrun,
        dust4_type=dust4_type, dust4_index=dust4_index, dust4=dust4,
        **kwargs
    )

    # Compute absorbed luminosity from continuum
    L_absorbed = compute_absorbed_luminosity(wave, intrinsic_total, attenuated_spec)

    # Add absorption from emission lines
    # lines[0] = young lines, lines[1] = old lines (usually zero)
    intrinsic_line_lum = np.sum(lines[0] + lines[1])
    attenuated_line_lum = np.sum(attenuated_lines)
    L_absorbed += max(0.0, intrinsic_line_lum - attenuated_line_lum)

    return attenuated_spec, attenuated_lines, L_absorbed


def add_igm(wave, spec, zred=None, igm_factor=1.0, add_igm_absorption=None, **kwargs):
    """IGM absorption based on Madau+1995
    wave: rest-frame wavelength
    spec: spectral flux density

    returns an attenuated spectrum
    """

    if add_igm_absorption == False:
        return spec

    redshifted_wave = np.asarray(wave) * (1+zred)

    lylim = 911.75
    lyw = np.array([1215.67, 1025.72, 972.537, 949.743, 937.803, 930.748, 926.226, 923.150, 920.963, 919.352,918.129, 917.181, 916.429, 915.824, 915.329, 914.919, 914.576])
    lycoeff = np.array([0.0036,0.0017,0.0011846,0.0009410,0.0007960,0.0006967,0.0006236,0.0005665,0.0005200,0.0004817,0.0004487, 0.0004200,0.0003947,0.000372,0.000352,0.0003334,0.00031644])
    nly = len(lyw)

    tau_line = np.zeros_like(redshifted_wave)
    for i in range(nly):
        lmin = lyw[i]
        lmax = lyw[i]*(1.0+zred)

        idx0 = np.where((redshifted_wave>=lmin) & (redshifted_wave<=lmax))[0]
        tau_line[idx0] += lycoeff[i]*np.exp(3.46*np.log(redshifted_wave[idx0]/lyw[i]))

    xc = redshifted_wave/lylim
    xem = 1.0+zred

    idx = np.where(xc<1.0)[0]
    xc[idx] = 1.0

    idx = np.where(xc>xem)[0]
    xc[idx] = xem

    tau_cont = 0.25*xc**3*(np.exp(0.46*np.log(xem)) - np.exp(0.46*np.log(xc)))
    tau_cont = tau_cont + 9.4*np.exp(1.5*np.log(xc))*(np.exp(0.18*np.log(xem)) - np.exp(0.18*np.log(xc)))
    tau_cont = tau_cont - 0.7*xc**3*(np.exp(-1.32*np.log(xc)) - np.exp(-1.32*np.log(xem)))
    tau_cont = tau_cont - 0.023*(np.exp(1.68*np.log(xem)) - np.exp(1.68*np.log(xc)))
    tau = tau_line + tau_cont

    # attenuate the input spectrum by the IGM
    # include a fudge factor to dial up/down the strength
    res = spec*np.exp(-tau*igm_factor)
    tiny_number = 10**(-70.0)
    return np.clip(res, a_min=tiny_number, a_max=None)


#---------------- AGN torus emission ----------------

SPS_HOME = os.getenv('SPS_HOME')
Nenkova2008 = np.genfromtxt(os.path.join(SPS_HOME, 'dust', 'Nenkova08_y010_torusg_n10_q2.0.dat'),
                            dtype=[('wave', 'f8'),
                                   ('fnu_5', '<U20'), ('fnu_10', '<U20'), ('fnu_20', '<U20'), 
                                   ('fnu_30', '<U20'), ('fnu_40', '<U20'), ('fnu_60', '<U20'), 
                                   ('fnu_80', '<U20'), ('fnu_100', '<U20'), ('fnu_150', '<U20')],
                            delimiter='   ', skip_header=4)
agndust_tau = np.array([5.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 150.0])
agndust_lam = Nenkova2008['wave']
nagndust = agndust_tau.shape[0] # =9, number of optical depths for AGN dust models
nagndust_spec = Nenkova2008['wave'].shape[0] # =125, number of spectral points in the input library

agndust_specinit = [Nenkova2008['fnu_5'],  Nenkova2008['fnu_10'], Nenkova2008['fnu_20'],
                    Nenkova2008['fnu_30'], Nenkova2008['fnu_40'], Nenkova2008['fnu_60'],
                    Nenkova2008['fnu_80'], Nenkova2008['fnu_100'], Nenkova2008['fnu_150']]
agndust_specinit = np.vstack(agndust_specinit).T
agndust_specinit = np.array(agndust_specinit, dtype=np.float64)
# agndust_specinit[agndust_lam<3e4,:] = 0

def agn_torus(wave, agn_tau):
    """AGN torus emission based on Nenkova+2008
    wave: rest-frame wavelength in Angstrom
    agn_tau: optical depth of the AGN dust torus, which affects the shape of the AGN SED;
             outside the range (5, 150) the AGN SED is those at 5 and 150.

    returns an AGN torus spectrum, interpolated on the input wavelength grid and agn_tau
    """
    agndust_spec = np.zeros((len(wave), nagndust), dtype=np.float64)

    # interpolate data onto wave
    i1 = np.argmin(np.abs(wave - agndust_lam[0]))
    i2 = np.argmin(np.abs(wave - agndust_lam[nagndust_spec - 1]))
    for i in range(nagndust):
        agndust_spec[i1:i2 + 1, i] = 10**np.interp(np.log10(wave[i1:i2 + 1]),
                                                   np.log10(agndust_lam),
                                                   np.log10(agndust_specinit[:, i] + 1e-30)) - 1e-30

    # interpolate in tau_agn
    # extrapolate if outside the bounds
    ## no extrapolation -- if outside the bounds, use the NN
    linfit = interp1d(agndust_tau, agndust_spec, axis=1, bounds_error=False, fill_value='extrapolate')
        # fill_value=(agndust_spec[:,0], agndust_spec[:,-1]))

    agndust_speci = linfit(agn_tau)

    return agndust_speci


