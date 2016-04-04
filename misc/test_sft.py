import sys, os, time
import numpy as np
import matplotlib.pyplot as pl
import fsps
from bsfh.source_basis import CompositeSFH
from sedpy import observate

sfhtype = {1:'tau', 4: 'delaytau', 5: 'simha'}


compute_vega_mags = False
zcontinuous = 1
sps = fsps.StellarPopulation(compute_vega_mags=compute_vega_mags,
                             zcontinuous=zcontinuous)
mysps = CompositeSFH(sfh_type='tau', interp_type='logarithmic',
                     flux_interp='linear', zcontinuous=zcontinuous,
                     compute_vega_mags=compute_vega_mags)
mysps.configure()
sspages = np.insert(mysps.logage, 0, 0)

badsimha = {'logtau': [1.34, 0.62], 'delt_trunc': [0.91, 0.98],
             'sf_tanslope': [1.24, -1.54], }
i, sfh = 1, 5
ages = np.linspace(1.4, 1.7, 45)
tau = 10**badsimha['logtau'][i]
delt_trunc = badsimha['delt_trunc'][i]
sf_slope = np.tan(badsimha['sf_tanslope'][i])

filters = observate.load_filters(['galex_FUV', 'sdss_r0'])

spec = np.zeros([len(ages), len(sps.wavelengths)])
myspec = np.zeros([len(ages), len(sps.wavelengths)])

pname = 'tage'
sps.params['sfh'] = sfh
mysps.sfh_type = sfhtype[sfh]
mysps.configure()

sfig, saxes = pl.subplots(2, 1, figsize=(11, 8.5))
rax, dax = saxes
wfig, wax = pl.subplots()
for i, tage in enumerate(ages):
    sf_trunc = tage * delt_trunc
    sps.params['tau'] = tau
    sps.params['tage'] = tage
    sps.params['sf_slope'] = sf_slope
    sps.params['sf_trunc'] = sf_trunc
    sfh_params = {'tage': tage*1e9, 'tau': tau*1e9,
                  'sf_slope': sf_slope / 1e9, 'sf_trunc': sf_trunc*1e9}
    w, s = sps.get_spectrum(tage=tage, peraa=True)
    spec[i, :] = s
    mw, mys = mysps.get_galaxy_spectrum(**sfh_params)
    myspec[i, :] = mys
    wax.plot(sspages, mysps.all_ssp_weights, '-o', label=r'{}={:4.2f}'.format(pname, tage))
    rax.plot(mw, mys / s, label=r'{}={:4.2f}'.format(pname, tage))
    dax.plot(mw, s - mys, label=r'{}={:4.2f}'.format(pname, tage))

mags = observate.getSED(sps.wavelengths, spec, filterlist=filters)
mymags = observate.getSED(sps.wavelengths, myspec, filterlist=filters)

fig, ax = pl.subplots()
ax.plot(ages, mags[:, 0], '-o')
ax.plot(ages, mymags[:, 0], '-o')


rax.set_xlim(1e3, 2e4)
rax.set_ylabel('pro / FSPS')
dax.set_xlim(1e3, 2e4)
dax.set_ylabel('FSPS - pro')
[ax.legend(loc=0, prop={'size': 10}) for ax in [rax, dax, wax]]
wax.set_yscale('log')
wax.set_xlabel('log t$_{lookback}$')
wax.set_ylabel('weight')
[ax.set_title('SFH={} ({} model)'.format(sfh, sfhtype[sfh]))
 for ax in [rax, wax]]

pl.show()
    


