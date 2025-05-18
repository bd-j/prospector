import numpy as np
import os

# index for the 128 emulated emission lines in fsps new line list
idx = np.array([0,1,2,3,4,5,6,9,13,14,15,16,17,18,19,20,21,22,23,24,24,25,26,28,29,30,31,
          32,34,35,37,38,39,40,41,43,44,45,46,47,48,49,50,51,52,53,54,57,59,61,62,
          63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,84,85,86,87,88,89,
          90,91,92,93,94,95,96,97,100,101,101,102,103,104,105,106,107,108,111,112,
          114,116,118,119,122,123,125,127,129,130,134,137,139,140,143,145,146,148,
          151,152,153,154,155,156,157,158,159,160,161,162,163,164,165])

__all__ = ["add_dust", "add_igm"]


def add_dust(wave,specs,line_waves,lines,dust_type=0,dust_index=0.0,dust2=0.0,dust1_index=-1.0,dust1=0.0,**kwargs):
    """
    wave: wavelength vector in Angstroms
    specs: spectral flux density, in (young, old) pairs
    line_waves: list of emission line wavelengths (Angstroms)
    lines: emission line flux density, in (young, old) pairs
    """

    attenuated_specs = np.zeros_like(specs)
    attenuated_lines = np.zeros_like(lines)

    # Loop over the (young,old) pairs for both lines and continuum
    for i, (spec, line) in enumerate(zip(specs,lines)):
        if (i == 0):
            d1 = dust1
        else:
            d1 = 0.0

        attenuated_lines[i], diff_dust = attenuate(line,line_waves,dust_type=dust_type,dust_index=dust_index,dust2=dust2,dust1_index=dust1_index,dust1=d1)
        attenuated_specs[i], diff_dust = attenuate(spec,wave,dust_type=dust_type,dust_index=dust_index,dust2=dust2,dust1_index=dust1_index,dust1=d1)
        
    attenuated_specs = attenuated_specs[0] + attenuated_specs[1]
    attenuated_lines = attenuated_lines[0] + attenuated_lines[1]
    return attenuated_specs, attenuated_lines


def attenuate(spec,lam,dust_type=0,dust_index=0.0,dust2=0.0,dust1_index=0.0,dust1=0.0):
    """returns F(obs) for a given attenuation curve + dust1 + dust2
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

    dust1_ext = np.exp(-dust1*(lam/5500.)**dust1_index)
    dust2_ext = np.exp(-attn_curve)

    ext_tot = dust2_ext*dust1_ext

    return ext_tot*spec, dust2_ext

def add_igm(wave, spec, zred=0., igm_factor=1.0, add_igm_absorption=None, **kwargs):
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
