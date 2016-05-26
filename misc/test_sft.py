import sys, os, time
import numpy as np
import matplotlib.pyplot as pl
import fsps
from prospect.sources import CompositeSFH
from sedpy import observate

sfhtype = {1:'tau', 4: 'delaytau', 5: 'simha'}

# build FSPS and Prospector sps objects
zcontinuous = 1
sps = fsps.StellarPopulation(zcontinuous=zcontinuous)
tres = np.round(len(sps.ssp_ages) / 94.)
mysps = CompositeSFH(sfh_type='tau', interp_type='logarithmic', mint_log=5.45,
                     flux_interp='linear', zcontinuous=zcontinuous)
mysps.configure()

# Save the Prospector SSP time axis
sspages = np.insert(mysps.logage, 0, mysps.mint_log)

# Set up some parameters that cause trouble in FSPS
pname = 'tage' # the parameter that will vary
badsimha = {'logtau': [1.34, 0.62], 'delt_trunc': [0.91, 0.98],
             'sf_tanslope': [1.24, -1.54], }
i, sfh = 1, 5
ages = np.linspace(1.4, 1.7, 20)
tau = 10**badsimha['logtau'][i]
delt_trunc = badsimha['delt_trunc'][i]
sf_slope = np.tan(badsimha['sf_tanslope'][i])

# filters to project onto
filters = observate.load_filters(['galex_FUV', 'sdss_r0'])

# set up output for spectra
spec = np.zeros([len(ages), len(sps.wavelengths)])
myspec = np.zeros([len(ages), len(sps.wavelengths)])

# Set the SFH type for both sps objects
sps.params['sfh'] = sfh
mysps.sfh_type = sfhtype[sfh]
mysps.configure()

# nstantiate figures and axes
sfig, saxes = pl.subplots(2, 1, figsize=(11, 8.5))
rax, dax = saxes
wfig, wax = pl.subplots()
# Loop over the varying parameter
for i, tage in enumerate(ages):
    # Set FSPS parameters, and get ans store spectrum
    sf_trunc = tage * delt_trunc
    sps.params['tau'] = tau
    sps.params['tage'] = tage
    sps.params['sf_slope'] = sf_slope
    sps.params['sf_trunc'] = sf_trunc
    w, s = sps.get_spectrum(tage=tage, peraa=True)
    spec[i, :] = s
    # Set up Pro parameters, with unit conversions, get spectrum, and store it.
    sfh_params = {'tage': tage*1e9, 'tau': tau*1e9,
                  'sf_slope': -sf_slope / 1e9, 'sf_trunc': sf_trunc*1e9}
    mw, mys, mstar = mysps.get_galaxy_spectrum(**sfh_params)
    myspec[i, :] = mys
    # Do some plotting for each age
    wax.plot(sspages, mysps.all_ssp_weights, '-o', label=r'{}={:4.2f}'.format(pname, tage))
    rax.plot(mw, mys / s, label=r'{}={:4.2f}'.format(pname, tage))
    dax.plot(mw, s - mys, label=r'{}={:4.2f}'.format(pname, tage))

# Get synthetic photometry for both sps objects
mags = observate.getSED(sps.wavelengths, spec, filterlist=filters)
mymags = observate.getSED(sps.wavelengths, myspec, filterlist=filters)

# Plot mags vs age
iband = 0
fig, ax = pl.subplots()
ax.plot(ages, mags[:, iband], '-o', label='FSPS, tres={}'.format(int(tres)))
ax.plot(ages, mymags[:, iband], '-o', label='Pro')
ax.set_xlabel('tage (Gyr)')
ax.set_ylabel(r'$M_{{AB}}$ ({})'.format(filters[iband].name))
ax.text(0.1, 0.85, r'$\tau_{{SF}}={:4.2f}, \, \Delta t_{{trunc}}={:4.2f}$'.format(tau, delt_trunc),
        transform=ax.transAxes)
ax.legend(loc=0)
fig.savefig('figures/sft_compare.pdf')
# Prettify axes
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

rax.set_ylim(0, 10)

pl.show()
    


