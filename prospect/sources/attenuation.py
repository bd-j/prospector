import numpy as np

def extinction(lam=None,dtype=0,dust_index=0.0,dust2=0.0,dust1_index=0.0,dust1=0.0):
    """returns F(obs) / F(emitted) for a given attenuation curve + dust1 + dust2
    """

    ### constants from FSPS
    dd63 = 6300.00
    lamv = 5500.0
    dlam = 350.0
    lamuvb = 2175.0

    ### check for out-of-bounds dust type
    if (dtype < 0) | (dtype > 6):
        raise ValueError('ATTN_CURVE ERROR: dust_type out of range:{0}'.format(dtype))

    ### power-law attenuation
    if (dtype == 0):
        attn_curve = (lam/lamv)**dust_index * dust2

    ### CCM89 extinction curve
    elif (dtype == 1):
        raise(NotImplementedError)

    ### Calzetti et al. 2000 attenuation
    elif (dtype == 2):
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
    elif (dtype == 3):
        raise(NotImplementedError)


    ### Kriek & Conroy 2013 attenuation
    elif (dtype == 4):
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
    elif (dtype == 5):
        raise(NotImplementedError)

    ### Reddy et al. (2015) attenuation
    elif (dtype == 6):
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
     
    return ext_tot
