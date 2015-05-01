# script to calculate the fractional change in SSP flux as a function
# of time.
import matplotlib.pyplot as pl
import numpy as np
import fsps
sps = fsps.StellarPopulation(zcontinuous=0)

# compile all metallicities
for i, z in enumerate(sps.zlegend):
    w, s = sps.get_spectrum(zmet=i+1)
spec, mass, lbol = sps.all_ssp_spec(peraa=True)


wmin, wmax, amin, amax, zmet = 1.5e3, 2e4, 0.01, 10, 4

ages = 10**(sps.ssp_ages-9)
waves = sps.wavelengths

gwave = (waves < wmax) & (waves > wmin)
gage = (ages < amax) & (ages > amin)

fdot = np.diff(spec, axis=1)
fbar = (spec[:,:-1,:] + spec[:,1:,:])/2.0

pl.imshow(np.squeeze((fdot/fbar)[np.ix_(gwave, gage, [zmet])]),
          interpolation='nearest', aspect='auto')




