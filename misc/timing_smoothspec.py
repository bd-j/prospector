import numpy as np
import matplotlib.pyplot as pl
import fsps, time, sys
from sedpy.observate import vac2air, lsf_broaden


def lsf(wave, sigma_smooth=0, **extras):
    return np.zeros(len(wave)) + sigma_smooth

def smoothspec(inwave, spec, lsf,
               min_wave_smooth=1e3, max_wave_smooth=1e4,
               **kwargs):
    
    smask = (inwave > min_wave_smooth) & (inwave < max_wave_smooth)
    ospec = spec.copy()
    ospec[smask] = lsf_broaden(inwave[smask], spec[smask], lsf, **kwargs)
    return ospec

def smoothspec_complete(inwave, spec, lsf, outwave=None,
                        zred=0.0, vacuum=False,
                        min_wave_smooth=3e3, max_wave_smooth=1e4,
                        **kwargs):
        
    a = (1 + zred)
    smask = (inwave > min_wave_smooth) & (inwave < max_wave_smooth)
    if not vacuum:
        w = vac2air(a * inwave[smask])
    else:
        w = a * inwave[smask]
    ospec = lsf_broaden(w, spec[smask], lsf, outwave=outwave)
     
    return ospec



if __name__ == "__main__":
    sps = fsps.StellarPopulation()
    sps.params['smooth_velocity'] = False
    wave, spec = sps.get_spectrum(tage=1.0, peraa=True)
    ns = 10
    sigma = np.random.uniform(1, 3, size=ns)

    fsps_dur, bsfh_dur = [], []
    for s in sigma:
        t = time.time()
        ospec_fsps = sps.smoothspec(wave, spec,s)
        fsps_dur.append(time.time()-t)
        t = time.time()
        ospec_bsfh = smoothspec(wave, spec, lsf, sigma_smooth=s)
        bsfh_dur.append(time.time()-t)
    bsfh_dur = np.array(bsfh_dur)
    fsps_dur = np.array(fsps_dur)
    
    pl.figure()
    pl.plot(wave, spec)
    pl.plot(wave, ospec_bsfh)
    pl.plot(wave, ospec_fsps)
    pl.xlim(1e3,1e4)
