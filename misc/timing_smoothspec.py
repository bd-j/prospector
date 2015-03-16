import numpy as np
import matplotlib.pyplot as pl
import fsps, time, sys
from sedpy.observate import vac2air, lsf_broaden
from scipy import sparse

    
def lsf_broaden_sparse(wave, spec, lsf=None, outwave=None,
                return_kernel=False, fwhm=False, dyr = 1e-4, **kwargs):
    """Broaden a spectrum using a wavelength dependent line spread
    function.  This function is only approximate because it doesn't
    actually do the integration over pixels, so for sparsely sampled
    points you'll have problems.

    :param wave:
        input wavelengths
    :param lsf:
        A function that returns the gaussian dispersion at each
        wavelength.  This is assumed to be in simga unless ``fwhm`` is
        ``True``
    :param outwave:
        Optional output wavelengths

    :param kwargs:
        Passed to lsf()
        
    :returns newspec:
        The broadened spectrum
        
    """
    if outwave is None:
        outwave = wave
    if lsf is None:
        return np.interp(outwave, wave, spec)
    dw = np.gradient(wave)
    sigma = lsf(wave, **kwargs)
    if fwhm:
        sigma = sigma/2.35
    kernel = outwave[:,None] - wave[None,:]
    kernel = (1/(sigma * np.sqrt(np.pi * 2))[None, :] *
              np.exp(-kernel**2/(2*sigma[None,:]**2)) *
              dw[None,:])
    
    kernel[kernel < kernel.max()*dyr] = 0
    skernel = sparse.csr_matrix(kernel)
    skernel = skernel/skernel.sum(axis=1)
    newspec = skernel.dot(spec)
    if return_kernel:
        return newspec, kernel
    return newspec

def lsf(wave, sigma_smooth=0, **extras):
    return np.zeros(len(wave)) + sigma_smooth

def smoothspec(inwave, spec, lsf, dosparse=False,
               min_wave_smooth=1e3, max_wave_smooth=1e4,
               **kwargs):
    
    smask = (inwave > min_wave_smooth) & (inwave < max_wave_smooth)
    ospec = spec.copy()
    if dosparse:
        ospec[smask] = lsf_broaden_sparse(inwave[smask],
                                          spec[smask], lsf, **kwargs)
    else:
        ospec[smask] = lsf_broaden(inwave[smask], spec[smask], lsf, **kwargs)
    return ospec

if __name__ == "__main__":
    sps = fsps.StellarPopulation()
    sps.params['smooth_velocity'] = False
    wave, spec = sps.get_spectrum(tage=1.0, peraa=True)
    ns = 10
    sigma = np.random.uniform(1, 3, size=ns)

    fsps_dur, bsfh_dur, bsfh_sparse_dur = [], [], []
    for s in sigma:
        t = time.time()
        ospec_fsps = sps.smoothspec(wave, spec,s)
        fsps_dur.append(time.time()-t)
        t = time.time()
        ospec_bsfh = smoothspec(wave, spec, lsf, sigma_smooth=s)
        bsfh_dur.append(time.time()-t)
        t = time.time()
        ospec_bsfh_sparse = smoothspec(wave, spec, lsf, sigma_smooth=s, dosparse=True)
        bsfh_sparse_dur.append(time.time()-t)

    bsfh_dur = np.array(bsfh_dur)
    bsfh_sparse_dur = np.array(bsfh_sparse_dur)
    fsps_dur = np.array(fsps_dur)
    
    pl.figure()
    pl.plot(wave, spec)
    pl.plot(wave, ospec_bsfh)
    pl.plot(wave, ospec_fsps)
    pl.plot(wave, ospec_bsfh_sparse)
    pl.xlim(1e3,1e4)
