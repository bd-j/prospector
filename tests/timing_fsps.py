import numpy as np
from sedpy import observate
import fsps
sps = fsps.StellarPopulation(zcontinuous=1)


filters = observate.load_filters( 20 * ['sdss_r0'])

time = np.linspace(0, 13.7, 12)
sfr = np.ones_like(time)

def sfh0(neb, z, mag=False):
    sps.params['add_neb_emission'] = neb
    sps.params['sfh'] = 0
    sps.params['logzsol'] = z
    w, s = sps.get_spectrum(tage=0)
    if mag:
        mag = observate.getSED(w, s, filterlist=filters)

def sfh3(neb, z, mag=False):
    sps.params['add_neb_emission'] = neb
    sps.params['sfh'] = 3
    sps.set_tabular_sfh(time, sfr)
    sps.params['logzsol'] = z
    return sps.get_spectrum(tage=13.6)
    if mag:
        mag = observate.getSED(w, s, filterlist=filters)

def sfh1(neb, z, mag=False):
    sps.params['add_neb_emission'] = neb
    sps.params['sfh'] = 4
    sps.params['logzsol'] = z
    return sps.get_spectrum(tage=13.699)
    if mag:
        mag = observate.getSED(w, s, filterlist=filters)
