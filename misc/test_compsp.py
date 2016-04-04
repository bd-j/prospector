# Test the analytic full SFR integrals implemented in SSPBasis against the
# COMPSP implementations of same.

import sys, os, time
import numpy as np
import matplotlib.pyplot as pl
import fsps
from bsfh.source_basis import CompositeSFH

sfhtype = {1:'tau', 4: 'delaytau', 5: 'simha'}


compute_vega_mags = False
zcontinuous = 1
sps = fsps.StellarPopulation(compute_vega_mags=compute_vega_mags,
                             zcontinuous=zcontinuous)
mysps = CompositeSFH(sfh_type='tau', interp_type='logarithmic', flux_interp='linear',
                     compute_vega_mags=compute_vega_mags, zcontinuous=zcontinuous)
mysps.configure()
sspages = np.insert(mysps.logage, 0, 0)



def main():
    tage = 1.4
    sf_trunc = tage * np.linspace(0.90, 1.02, 9)
    figlist, speclist = test_taumodel_sft(values=sf_trunc, tau=4.2,
                                          sf_slope=10.0, tage=tage, sfh=5)
    #figlist = test_taumodel_sfslope()
    pl.show()

    
def test_mint_convergence():
    """Test convergence of most recent bin.
     """
    sfh_params = {'tage': 1e9, 'tau':14e9}
    # set up an array of minimum values, and get the weights for each minimum value
    mint = np.linspace(-4, 2, 100)
    w = np.zeros([len(mint), len(mysps.logage)+1])
    for i, m in enumerate(mint):
        mysps.update(**sfh_params)
        mysps.mint_log = m
        mysps.configure()
        w[i,:] = mysps.all_ssp_weights

    # Plot summed weight for zeroth and 1st SSP
    pl.figure()
    zero = (w[:,0] + w[:,1])
    pl.plot(mint, zero, '-o')
    pl.yscale('log')
    pl.show()

    # Plot fractional weight error (relative to smallest tmin) as a function of tmin
    pl.figure()
    pl.plot(np.log10(mint), 1 - w[:,0]/w[0,0], '-o')
    pl.plot(np.log10(mint), 1 - w[:,1]/w[0,1], '-o')
    pl.show()

def test_normalization():
    sfh_params = {'tage': 1e9, 'tau':14e9}
    mtot = None
    w = mysps.all_ssp_weights


def test_simha(values=np.linspace(0.0, 4.0, 9), tage=3.0):
    pname = r'sf_trunc'
    sps.params['sfh'] = 5
    mysps.sfh_type = sfhtype[sfh]
    mysps.configure()
    sfig, saxes = pl.subplots(2, 1, figsize=(11, 8.5))
    #for sf_trunc in 

def test_taumodel_dust(values=np.linspace(0.0, 4.0, 9),
                       tage=3.0, tau=1.0, sfh=1):
    pname = r'$\tau_V$'
    sps.params['sfh'] = sfh
    mysps.sfh_type = sfhtype[sfh]
    mysps.configure()
    sfig, saxes = pl.subplots(2, 1, figsize=(11, 8.5))
    rax, dax = saxes
    #wfig, wax = pl.subplots()
    for dust2 in values:
        sps.params['tau'] = tau
        sps.params['tage'] = tage
        sps.params['dust2'] = dust2
        sfh_params = {'tage': tage*1e9, 'tau': tau*1e9, 'dust2': dust2}
        w, spec = sps.get_spectrum(tage=tage, peraa=True)
        mw, myspec = mysps.get_galaxy_spectrum(**sfh_params)
        rax.plot(mw, myspec / spec, label=r'{}={:4.2f}'.format(pname, dust2))
        dax.plot(mw, spec - myspec, label=r'{}={:4.2f}'.format(pname, dust2))
        #wax.plot(sspages, mysps.all_ssp_weights, '-o', label=r'{}={:4.2f}'.format(pname, dust2))
    rax.set_xlim(1e3, 1e7)
    rax.set_ylabel('pro / FSPS')
    dax.set_xlim(1e3, 1e7)
    dax.set_ylabel('FSPS - pro')
    [ax.set_xscale('log') for ax in [rax, dax]]
    [ax.legend(loc=0, prop={'size': 10}) for ax in [rax, dax]]
    [ax.text(0.1, 0.85, r'$\tau_{{SF}}={}, tage={}$'.format(tau, tage), transform=ax.transAxes)
     for ax in [rax, dax]]
    #wax.set_yscale('log')
    #wax.set_xlabel('log t$_{lookback}$')
    #wax.set_ylabel('weight')
    [ax.set_title('SFH={} ({} model)'.format(sfh, sfhtype[sfh]))
     for ax in [rax]]
    return [sfig]
    
    
def test_taumodel_tau(values=10**np.linspace(-1, 1, 9),
                      tage=10.0, sfh=1):
    """Test (delayed-) tau models
    """
    pname = '$\tau$'
    sps.params['sfh'] = sfh
    mysps.sfh_type = sfhtype[sfh]
    mysps.configure()
    sfig, saxes = pl.subplots(2, 1, figsize=(11, 8.5))
    rax, dax = saxes
    wfig, wax = pl.subplots()
    for tau in values:
        sps.params['tau'] = tau
        sps.params['tage'] = tage
        sfh_params = {'tage': tage*1e9, 'tau': tau*1e9}
        w, spec = sps.get_spectrum(tage=tage, peraa=True)
        mw, myspec = mysps.get_galaxy_spectrum(**sfh_params)
        rax.plot(mw, myspec / spec, label=r'{}={:4.2f}'.format(pname, tau))
        dax.plot(mw, spec - myspec, label=r'{}={:4.2f}'.format(pname, tau))
        wax.plot(sspages, mysps.all_ssp_weights, '-o', label=r'{}={:4.2f}'.format(pname, tau))
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
    return [sfig, wfig]

def test_taumodel_tage(values=10**np.linspace(np.log10(0.11), 1, 9),
                       tau=1.0, sf_slope=0.0, sfh=1):
    """Test (delayed-) tau models
    """
    pname = 'tage'
    sps.params['sfh'] = sfh
    mysps.sfh_type = sfhtype[sfh]
    mysps.configure()
    sfig, saxes = pl.subplots(2, 1, figsize=(11, 8.5))
    rax, dax = saxes
    wfig, wax = pl.subplots()
    for tage in values:
        sps.params['tau'] = tau
        sps.params['tage'] = tage
        #sps.params['sf_slope'] = sf_slope
        sfh_params = {'tage': tage*1e9, 'tau': tau*1e9}
        w, spec = sps.get_spectrum(tage=tage, peraa=True)
        mw, myspec = mysps.get_galaxy_spectrum(**sfh_params)
        rax.plot(mw, myspec / spec, label=r'{}={:4.2f}'.format(pname, tage))
        dax.plot(mw, spec - myspec, label=r'{}={:4.2f}'.format(pname, tage))
        wax.plot(sspages, mysps.all_ssp_weights, '-o', label=r'{}={:4.2f}'.format(pname, tage))
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
    return [sfig, wfig]

def test_taumodel_sft(values=11 - 10**np.linspace(np.log10(0.11), 1, 9),
                      tau=1.0, sf_slope=0, tage=11.0, sfh=1):
    """Test (delayed-) tau models
    """
    pname = 'sf_trunc'
    sps.params['sfh'] = sfh
    mysps.sfh_type = sfhtype[sfh]
    mysps.configure()
    sfig, saxes = pl.subplots(2, 1, figsize=(11, 8.5))
    rax, dax = saxes
    wfig, wax = pl.subplots()
    for sf_trunc in values:
        sps.params['tau'] = tau
        sps.params['tage'] = tage
        sps.params['sf_trunc'] = sf_trunc
        sps.params['sf_slope'] = sf_slope
        sfh_params = {'tage': tage*1e9, 'tau': tau*1e9, 'sf_trunc': sf_trunc*1e9,
                      'sf_slope': sf_slope / 1e9}
        w, spec = sps.get_spectrum(tage=tage, peraa=True)
        mw, myspec = mysps.get_galaxy_spectrum(**sfh_params)
        rax.plot(mw, myspec / spec, label=r'{}={:4.2f}'.format(pname, sf_trunc))
        dax.plot(mw, spec - myspec, label=r'{}={:4.2f}'.format(pname, sf_trunc))
        wax.plot(sspages, mysps.all_ssp_weights, '-o', label=r'{}={:4.2f}'.format(pname, sf_trunc))
        wax.axvline(np.log10((tage - sf_trunc) * 1e9), linestyle=':', color='k')
    rax.set_xlim(1e3, 2e4)
    rax.set_ylabel('pro / FSPS')
    dax.set_xlim(1e3, 2e4)
    dax.set_ylabel('FSPS - pro')
    [ax.legend(loc=0, prop={'size': 10}) for ax in [rax, dax, wax]]
    [ax.text(0.1, 0.85, r'$\tau_{{SF}}={}, tage={}$'.format(tau, tage), transform=ax.transAxes)
     for ax in [rax, dax, wax]]
    wax.set_yscale('log')
    wax.set_xlabel('log t$_{lookback}$')
    wax.set_ylabel('weight')
    logttrunc = np.log10((tage - values) * 1e9)
    wax.set_xlim(logttrunc.min() - 0.5, logttrunc.max() + 0.5)
    [ax.set_title('SFH={} ({} model)'.format(sfh, sfhtype[sfh]))
     for ax in [rax, wax]]
    return [sfig, wfig]

def test_taumodel_sfslope(values=np.linspace(-10, 10, 9),
                          tau=10.0, sf_trunc=10.0, tage=11.0, sfh=5):
    """Test (delayed-) tau models
    """
    pname = 'sf_slope'
    sps.params['sfh'] = sfh
    mysps.sfh_type = sfhtype[sfh]
    mysps.configure()
    sfig, saxes = pl.subplots(2, 1, figsize=(11, 8.5))
    rax, dax = saxes
    wfig, wax = pl.subplots()
    for sf_slope in values:
        sps.params['tau'] = tau
        sps.params['tage'] = tage
        sps.params['sf_trunc'] = sf_trunc
        sps.params['sf_slope'] = sf_slope
        sfh_params = {'tage': tage*1e9, 'tau': tau*1e9, 'sf_trunc': sf_trunc*1e9,
                      'sf_slope': sf_slope / 1e9}
        w, spec = sps.get_spectrum(tage=tage, peraa=True)
        mw, myspec = mysps.get_galaxy_spectrum(**sfh_params)
        rax.plot(mw, myspec / spec, label=r'{}={:4.2f}'.format(pname, sf_slope))
        dax.plot(mw, spec - myspec, label=r'{}={:4.2f}'.format(pname, sf_slope))
        wax.plot(sspages, mysps.all_ssp_weights, '-o', label=r'{}={:4.2f}'.format(pname, sf_slope))
        wax.axvline(np.log10((tage - sf_trunc) * 1e9), linestyle=':', color='k')
    rax.set_xlim(1e3, 2e4)
    rax.set_ylabel('pro / FSPS')
    dax.set_xlim(1e3, 2e4)
    dax.set_ylabel('FSPS - pro')
    [ax.legend(loc=0, prop={'size': 10}) for ax in [rax, dax, wax]]
    [ax.text(0.1, 0.85, r'$\tau_{{SF}}={}, tage={}$'.format(tau, tage), transform=ax.transAxes)
     for ax in [rax, dax, wax]]
    wax.set_yscale('log')
    wax.set_xlabel('log t$_{lookback}$')
    wax.set_ylabel('weight')
    logttrunc = np.log10((tage - values) * 1e9)
    wax.set_xlim(logttrunc.min() - 0.5, logttrunc.max() + 0.5)
    [ax.set_title('SFH={} ({} model)'.format(sfh, sfhtype[sfh]))
     for ax in [rax, wax]]
    return [sfig, wfig]


if __name__ == "__main__":
    main()
