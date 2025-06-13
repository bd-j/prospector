import numpy as np
import os
import dill as pickle
from pkg_resources import resource_filename

import jax.numpy as jnp
import jax
from pathlib import Path



# index for the 128 emulated emission lines in fsps new line list
idx = np.array([0,1,2,3,4,5,6,9,13,14,15,16,17,18,19,20,21,22,23,24,24,25,26,28,29,30,31,
          32,34,35,37,38,39,40,41,43,44,45,46,47,48,49,50,51,52,53,54,57,59,61,62,
          63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,84,85,86,87,88,89,
          90,91,92,93,94,95,96,97,100,101,101,102,103,104,105,106,107,108,111,112,
          114,116,118,119,122,123,125,127,129,130,134,137,139,140,143,145,146,148,
          151,152,153,154,155,156,157,158,159,160,161,162,163,164,165])

out = pickle.load(open(resource_filename("cuejax", "data/nn_stats_v0.pkl"), "rb"))
frac_line_err = 1./out['SN_quantile'][1][np.argsort(out['wav'])] # 1 / upper 2 sigma quantike of SN of the cue test set

__all__ = ["add_dust", "add_igm", "DustEmission"]


def add_dust(wave,specs,line_waves,lines,dust_type=0,dust_index=-0.7,dust2=0.0,dust1_index=-1.0,dust1=0.0,**kwargs):
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
    
#     if kwargs.get("add_dust_emission", None):
#         dust_specs = DustEmission(dust_file = os.getenv('SPS_HOME'),
#                                   spec_lambda = wave, **kwargs).compute_dust_emission(
#             attenuated_specs, specs[0]+specs[1], wave, 
#             diff_dust, attenuated_lines, lines[0]+lines[1])[0]
#         return dust_specs, attenuated_lines
    
#     else:
#         return attenuated_specs, attenuated_lines
    return attenuated_specs, attenuated_lines



def attenuate(spec,lam,dust_type=0,dust_index=-0.7,dust2=0.0,dust1_index=0.0,dust1=0.0):
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


class DustEmission:

    def __init__(self, duste_model="DL07",
                 dust_file=None, spec_lambda=None, **kwargs):
        """
        Initialize the DustEmission object with parameters for dust emission modeling.

        Parameters
        ----------
        duste_model : str
            Dust emission model to use: 'DL07' or 'THEMIS'.
        dust_file : str
            Path to the dust emission file (required).
        spec_lambda : ndarray
            Wavelength grid over which dust emission will be evaluated (required).
        kwargs : dict
            Optional keyword arguments to override default dust parameters.
            Supported: duste_qpah, duste_umin, duste_gamma
        """

        # Store model choice (e.g., 'DL07' or 'THEMIS')
        self.duste_model = duste_model

        # Dust parameter values
        self.duste_qpah = None
        self.duste_umin = None
        self.duste_gamma = None

        # Model grid arrays for allowed values of qPAH and Umin
        self.qpaharr = None
        self.uminarr = None

        # Placeholder for loaded dust emission spectra
        self.dustem2_dustem = None

        # File path and wavelength grid
        self.dust_file = None
        self.spec_lambda = None

        # Store any additional keyword arguments for later use
        self.dwargs = kwargs

        # Read in key dust parameters, allowing overrides via kwargs
        self.duste_qpah = kwargs.pop("duste_qpah", 1.1)
        self.duste_umin = kwargs.pop("duste_umin", 0.72)
        self.duste_gamma = kwargs.pop("duste_gamma", 0.5)

        # Set parameter grids based on selected dust model
        if self.duste_model == "DL07":
            self.qpaharr = jnp.array([0.47, 1.12, 1.77, 2.50, 3.19, 3.90, 4.58])
            self.uminarr = jnp.array([
                0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0,
                2.5, 3.0, 4.0, 5.0, 7.0, 8.0, 12.0, 15.0, 20.0, 25.0
            ])
        elif self.duste_model == "THEMIS":
            # THEMIS model uses smaller qPAH values rescaled to percent
            self.qpaharr = jnp.array([0.02, 0.06, 0.10, 0.14, 0.17, 0.20, 0.24,
                                      0.28, 0.32, 0.36, 0.40]) / 2.2 * 100
            self.uminarr = jnp.array([
                0.1, 0.12, 0.15, 0.17, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6,
                0.7, 0.8, 1.0, 1.2, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0,
                6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 17.0, 20.0, 25.0, 30.0,
                35.0, 40.0, 50.0, 80.0
            ])
        else:
            raise ValueError("Invalid duste_model. Choose 'DL07' or 'THEMIS'.")

        # Ensure that necessary data is provided
        if dust_file is None or spec_lambda is None:
            raise ValueError("If `duste=True`, both `dust_file` and `spec_lambda` must be provided.")

        self.dust_file = dust_file
        self.spec_lambda = spec_lambda

        # Load emission templates or model data from file
        self.load_dust_emission(dust_file, spec_lambda)

    def __repr__(self):
        """
        Custom string representation of the DustEmission object.
        Provides a readable summary of model settings and parameters.
        """

        def format_array(arr):
            """Helper to format short arrays inline; longer arrays multiline."""
            if arr is None:
                return "None"
            if arr.ndim == 1 and len(arr) <= 5:
                return f"[{', '.join(map(str, arr))}]"
            return f"\n    " + "\n    ".join(map(str, arr))

        attributes = {
            "Duste model": self.duste_model,
            "DUST qPAH": self.duste_qpah,
            "DUST Umin": self.duste_umin,
            "DUST Gamma": self.duste_gamma,
            f"qpaharr ({self.duste_model})": format_array(self.qpaharr),
            f"uminarr ({self.duste_model})": format_array(self.uminarr),
            "dust_file": self.dust_file,
            "spec_lambda": self.spec_lambda.shape if self.spec_lambda is not None else None,
        }

        if self.dwargs:
            attributes["Extra parameters (dwargs)"] = self.dwargs

        attr_str = "\n".join(f"  {k:<30}: {v}" for k, v in attributes.items() if v is not None)
        return f"\nDustEmission Model:\n{'='*50}\n{attr_str}\n{'='*50}"

    def load_dust_emission(self, dust_file=None, spec_lambda=None):
        
        # Use default paths if not provided
        if dust_file is None:
            dust_file = self.dust_file
        if spec_lambda is None:
            spec_lambda = self.spec_lambda

        # Select dust model parameters
        dust_model_params = {
            "DL07": (7, 1001, 22),
            "THEMIS": (11, 576, 37),
        }
        
        if self.duste_model not in dust_model_params:
            raise ValueError("Invalid duste_model. Choose 'DL07' or 'THEMIS'.")

        nqpah_dustem, ndim_dustem, numin_dustem = dust_model_params[self.duste_model]

        # Initialize storage for interpolated spectra (JAX-compatible)
        dustem2_dustem = np.zeros((len(spec_lambda), nqpah_dustem, numin_dustem * 2))

        # Read and interpolate dust emission spectra
        for k in range(nqpah_dustem):
            filename = Path(dust_file) / "dust" / "dustem" / f"{self.duste_model}_MW3.1_{'100' if k == 10 else f'{k}0'}.dat"

            if not filename.exists():
                raise FileNotFoundError(f"Error opening dust emission file: {filename}. File does not exist.")

            with filename.open('r') as f:
                next(f)  # Skip first header line
                next(f)  # Skip second header line

                lambda_dustem = np.zeros(ndim_dustem)
                dustem_dustem = np.zeros((ndim_dustem, numin_dustem * 2))

                for i in range(ndim_dustem):
                    try:
                        values = list(map(float, f.readline().strip().split()))
                        lambda_dustem[i], dustem_dustem[i, :] = values[0], values[1:]
                    except Exception:
                        raise RuntimeError(f"Error reading dust emission file: {filename}")

            # Convert wavelength from microns to Angstroms
            lambda_dustem *= 1E4  

            # Interpolate dust spectra onto the master wavelength array
            jj = jnp.searchsorted(spec_lambda / 1E4, 1, side='left')
            for j in range(numin_dustem * 2):
                dustem2_dustem[jj:, k, j] = jnp.interp(spec_lambda[jj:], lambda_dustem, dustem_dustem[:, j])

        self.dustem2_dustem = jnp.array(dustem2_dustem)

    def compute_dust_emission(self, specdust, csp_spectra, spec_lambda, diff_dust, 
                              linedust, line,
                              duste_qpah=None, duste_umin=None, duste_gamma=None):
        """
        Compute dust emission using JAX-optimized vectorization for GPU acceleration.

        Parameters:
            specdust (jnp.ndarray): Attenuated spectrum after dust absorption.
            csp_spectra (jnp.ndarray): Stellar spectrum before attenuation.
            spec_lambda (jnp.ndarray): Wavelength array in Angstroms.
            duste_qpah (float, optional): PAH fraction. Defaults to self.duste_qpah if None.
            duste_umin (float, optional): Minimum U radiation field. Defaults to self.duste_umin if None.
            duste_gamma (float, optional): Fraction of high U component. Defaults to self.duste_gamma if None.

        Returns:
            tuple: (Updated spectrum with dust emission added, Estimated dust mass)
        """

        # Use provided parameters if given, otherwise fallback to self attributes
        duste_qpah = duste_qpah if duste_qpah is not None else self.duste_qpah
        duste_umin = duste_umin if duste_umin is not None else self.duste_umin
        duste_gamma = duste_gamma if duste_gamma is not None else self.duste_gamma


        # Compute total luminosity before and after attenuation
        nu = 2.9979E18 / spec_lambda  # Frequency in Hz (c / λ)
        lbold = jax.scipy.integrate.trapezoid(nu * specdust, -nu) + jnp.sum(linedust)  # L_bol after attenuation
        lboln = jax.scipy.integrate.trapezoid(nu * csp_spectra, -nu) + jnp.sum(line)  # L_bol before attenuation
                
        # Interpolation indices for PAH fraction and Umin
        qlo = jnp.clip(jnp.searchsorted(self.qpaharr, duste_qpah) - 1, 0, len(self.qpaharr) - 2)
        dq = jnp.clip((duste_qpah - self.qpaharr[qlo]) / (self.qpaharr[qlo + 1] - self.qpaharr[qlo]), 0.0, 1.0)
        ulo = jnp.clip(jnp.searchsorted(self.uminarr, duste_umin) - 1, 0, len(self.uminarr) - 2)
        du = jnp.clip((duste_umin - self.uminarr[ulo]) / (self.uminarr[ulo + 1] - self.uminarr[ulo]), 0.0, 1.0)
    

        # Ensure gamma fraction is within [0,1]
        gamma = jnp.clip(duste_gamma, 0.0, 1.0)

        # Perform bilinear interpolation over qpah and Umin using `vmap`
        def interpolate_dustem(i):
            return (
                (1 - dq) * (1 - du) * self.dustem2_dustem[i, qlo, 2 * ulo - 1] +
                dq * (1 - du) * self.dustem2_dustem[i, qlo + 1, 2 * ulo - 1] +
                dq * du * self.dustem2_dustem[i, qlo + 1, 2 * (ulo + 1) - 1] +
                (1 - dq) * du * self.dustem2_dustem[i, qlo, 2 * (ulo + 1) - 1]
            ), (
                (1 - dq) * (1 - du) * self.dustem2_dustem[i, qlo, 2 * ulo] +
                dq * (1 - du) * self.dustem2_dustem[i, qlo + 1, 2 * ulo] +
                dq * du * self.dustem2_dustem[i, qlo + 1, 2 * (ulo + 1)] +
                (1 - dq) * du * self.dustem2_dustem[i, qlo, 2 * (ulo + 1)]
            )

        dumin, dumax = jax.vmap(interpolate_dustem)(jnp.arange(len(spec_lambda)))

        # Compute dust emission spectrum
        mduste = (1 - gamma) * dumin + gamma * dumax
        mduste = jnp.maximum(mduste, 1e-70)
        
        # Normalize to absorbed luminosity
        labs = lboln - lbold  # Energy absorbed by dust
        norm = jax.scipy.integrate.trapezoid(nu * mduste, -nu)  # Normalization factor
        duste = mduste / norm * labs  # Normalize dust emission
        duste = jnp.maximum(duste, 1e-70)

        # Iterative correction for dust self-absorption using `jax.lax.while_loop`
        # sometimes this was stalled; use python while function instead
#         def cond_fn(state):
#             lbold, lboln, _ = state
#             return jnp.abs(lboln - lbold) > 1e-2

#         def body_fn(state):
#             lbold, lboln, tduste = state
#             oduste = duste
#             duste_att = duste * diff_dust # Apply diffuse attenuation, duste *jnp.exp(-self.diffuse_tau)  
#             tduste = tduste + duste_att

#             lbold = jax.scipy.integrate.trapezoid(nu * duste_att, -nu)  # Update L_bol after self-absorption
#             lboln = jax.scipy.integrate.trapezoid(nu * oduste, -nu)  # Before self-absorption

#             duste = jnp.maximum(mduste / norm * (lboln - lbold), 1e-70)
#             return lbold, lboln, tduste

#         _, _, tduste = jax.lax.while_loop(cond_fn, body_fn, (lbold, lboln, jnp.zeros_like(duste)))

        tduste = jnp.zeros_like(duste)
        while jnp.abs(lboln - lbold) > 1e-2:
            oduste = duste
            duste_att = duste * diff_dust # Apply diffuse attenuation, duste *jnp.exp(-self.diffuse_tau)  
            tduste = tduste + duste_att

            lbold = jax.scipy.integrate.trapezoid(nu * duste_att, -nu)  # Update L_bol after self-absorption
            lboln = jax.scipy.integrate.trapezoid(nu * oduste, -nu)  # Before self-absorption

            duste = jnp.maximum(mduste / norm * (lboln - lbold), 1e-70)

        # Compute estimated dust mass
        mdust = 3.21E-3 / (4 * jnp.pi) * labs / norm
  
        # Add dust emission to the stellar spectrum
        specdust = specdust + tduste

        return specdust, mdust
