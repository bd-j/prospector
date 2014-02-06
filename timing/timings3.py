#compare a lookup table of spectra at ages and metallicities to
#calls to fsps.sps.get_spectrum() for different metallicities

import numpy as np
import fsps
import time
import matplotlib.pyplot as pl
#import attenuation
from numpy import exp


def from_lookup(z, t, a):
    sps.params['zmet'] = 2
    sps.params['vel_broad'] = 0.0
    sps.params['dust2'] = 0.0
    spec_store[i,:] = (spb[tind,z-1,:]).sum(axis = 0) #* exp(-taucurve*tt[i])


def from_fsps(z, t, a):
    sps.params['zmet'] = z
    sps.params['vel_broad'] = 0.0
    sps.params['dust2'] = t
    #wave, spec  = sps.get_spectrum(tage = age[i])
    #spec_store2[:,i,:] = spec
    wave, spec  = sps.get_spectrum(peraa = True)
    spec_store2[i,:] = (spec[tind, :]).sum(axis = 0) 



sps = fsps.StellarPopulation()
sps.params['sfh'] = 0
#taucurve = attenuation.calzetti(sps.wavelengths)

ntry = 50
nage = 188
nmet = 5
nssp = 20

spec_store = np.zeros([ntry, len(sps.wavelengths)])
spec_store2 = np.zeros_like(spec_store)

tind = (np.arange(20) * 8).astype(int)
nssp = len(tind)

zz = np.random.uniform(1,5,ntry).astype(int)
tt = np.random.uniform(0,4,ntry)
aa = np.random.uniform(0,1,[ntry,nssp])
    
print('lookup')
dur_comb = np.zeros(ntry)
#build the lookup table
t0 = time.time()
spb = np.zeros([nage, nmet, len(sps.wavelengths)])
for iz in np.arange(nmet)+1:
    wave, spec = sps.get_spectrum(peraa = True, zmet = int(np.round(iz)))
    spb[:,iz-1,:] = spec
print('{0} sec to build lookup table'.format(time.time()-t0))

for i in xrange(ntry):
    t0 = time.time()
    from_lookup(zz[i], tt[i], aa[i])
    dur_comb[i] = time.time() - t0
    
print('fsps call')
dur_many = np.zeros(ntry)

for i in xrange(ntry):
    t0 = time.time()
    from_fsps(zz[i], 0.0, aa[i])
    dur_many[i] = time.time() - t0


#output
aa = np.arange(ntry)+1
pl.figure(1)
pl.clf()
pl.plot(dur_many, label = 'fsps call')
pl.plot(dur_comb, label = 'lookup')
pl.xlabel('call #')
pl.ylabel('sec/call')
pl.legend(loc = 'upper left')
pl.yscale('log')
pl.show()
pl.savefig('timings3.png')

#check that spectra are the same
pl.figure(2)
pl.clf()
pl.plot(sps.wavelengths, spec_store[int(ntry/2),:],'b')
pl.plot(sps.wavelengths, spec_store2[int(ntry/2),:], 'r:')
#pl.plot(sps.wavelengths, spb[148,2,:])
pl.xlim(3e3,9e3)
pl.yscale('log')
pl.show()
