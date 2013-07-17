import numpy as np
import matplotlib.pyplot as pl

def plot_sfh(pars, outsamples, angst_sfh, fine_sfh = None):
    # SFH
    delta =  np.ma.masked_invalid( (angst_sfh - outsamples).std(axis = 0)  / angst_sfh )
    ns = pars['nsample']

    if fine_sfh is not None:
        xx = np.log10( np.append(pars['bin_ends'], pars['bin_starts'][-1]) )
        xx[0] = 6.0
        pl.step(xx, np.append(fine_sfh,0), where = 'post',
            linewidth = 2, color = 'green', label = 'fine_input')

            #        pl.plot(np.log10(pars['bin_centers']*1e9),fine_sfh,
            #linewidth = 2, color = 'green', label = 'fine input', alpha = 0.7)

#    pl.plot(np.log10(pars['rebin_centers']*1e9),angst_sfh,
#            linewidth = 2, color = 'red', label = 'input')
    xx = np.log10( np.append(pars['rebin_ends'], pars['rebin_starts'][-1]) )
    xx[0] = 6.0
    pl.step(xx, np.append(angst_sfh,0), where = 'post',
            linewidth = 2, color = 'red', label = 'input')

    for i in np.random.uniform(size=30) :
        pl.plot(np.log10(pars['rebin_centers']*1e9),outsamples[np.floor(i * ns),:],
                color = 'grey', alpha = 0.5, label = 'post. sample')


    #pl.legend()
    pl.xlabel('lookback time')
    pl.ylabel('SFR')
    pl.title(r"{name}: $\lambda\lambda = {wlo},{whi}$,S/N = {snr},$\sigma_v ={veldisp}$ km/s, $t_{{sample}} = {tsample:.1f} s$".format(**pars))
    pl.annotate(r'$\langle \sigma (\Delta SFR_i/SFR_i) \rangle =$ %s' % delta.mean(), (6.5,0))

    pl.savefig("{figname}_sfh.png".format(**pars))
    pl.close()
    
def plot_spectrum(pars, outsamples, angst_sfh, wave, spec_array):
    # SPECTRA
    ns = pars['nsample']
    inspec = (angst_sfh * spec_array.T).sum(axis = 1)
    
    pl.plot(wave, inspec,
            linewidth = 2, color = 'red', label = 'input')
    for i in np.random.uniform(size=30) :
        pl.plot( wave,(outsamples[np.floor(i * ns),:]* spec_array.T).sum(axis = 1),
                 color = 'grey', alpha = 0.5, label = 'post. sample')

    pl.xlim(pars['wlo']*0.7,pars['whi']*1.3)
    pl.xscale('log')
    pl.ylim(inspec.max()*1e-3, inspec.max() )
    pl.yscale('log')    

    pl.xlabel(r'$\lambda (AA)$')
    pl.ylabel(r'F$_\lambda$')
    pl.title(r"{name}: $\lambda\lambda = {wlo},{whi}$, SNR = {snr}, $\sigma_v ={veldisp}$ km/s".format(**pars))

    pl.savefig("{figname}_spectrum.png".format(**pars))
    pl.close()

def plot_sfr_mass(pars, outsamples, angst_sfh, sfr_time_index = 3):
    # MASS and SFR8
    
    dt = pars['rebin_starts']-pars['rebin_ends']
    mass = (dt*outsamples).sum(axis = 1)
    sfr = (dt*outsamples).T[:sfr_time_index].sum(axis = 0)/(dt[:sfr_time_index].sum())

    #hist, bins = np.hist(mass, bins = 20)
    #widths = np.diff(bins)
    #pl.bar(bins[:-1],hist/hist.max(),widths, linewidth = 0, alpha = 0.7, color = 'r'
    pl.hist(mass, bins = 20, histtype = 'stepfilled',alpha = 0.7, color = 'r' )
    ymin, ymax = pl.ylim()
    pl.plot(np.array([0,0])+(dt*angst_sfh).sum(), np.array([ymin,ymax]), ':k', label = 'input')
    pl.ylim(ymin, ymax)
    
    pl.xlabel('total mass formed')
    pl.title(r"{name}: $\lambda\lambda = {wlo},{whi}$, SNR = {snr}, $\sigma_v ={veldisp}$ km/s".format(**pars))
    pl.savefig("{figname}_mass.png".format(**pars))
    pl.close()
    
    pl.hist(sfr,bins = 20, histtype = 'stepfilled', alpha = 0.7, color = 'b')
    ymin, ymax = pl.ylim()
    pl.plot(np.array([0,0])+(dt*angst_sfh).T[:sfr_time_index].sum()/(dt[:sfr_time_index].sum()),
            np.array([ymin,ymax]), ':k', label = 'input')
    pl.ylim(ymin, ymax)

    pl.xlabel( r'$\langle SFR\rangle _{{{0:.1f}}}$'.format(np.log10( pars['rebin_starts'][sfr_time_index-1] )) )
    pl.title(r"{name}: $\lambda\lambda = {wlo},{whi}$, SNR = {snr}, $\sigma_v ={veldisp}$ km/s".format(**pars))
    pl.savefig("{figname}_sfr.png".format(**pars))
    pl.close()
    
def plot_covariance(pars, outsamples, bin1 = 0, bin2 = 1, sfh_input = None):
    # COMPONENT COVARIANCE
    
    pl.plot(outsamples[:,bin1], outsamples[:,bin2],'ro', alpha = 0.1, mec ='red')

    pl.xlabel(r'SFR$_{{{0:.1f}}}$'.format(np.log10(pars['rebin_centers'][bin1]*1e9)) )
    pl.ylabel(r'SFR$_{{{0:.1f}}}$'.format(np.log10(pars['rebin_centers'][bin2]*1e9)) )
    pl.title(r"{name}: $\lambda\lambda = {wlo},{whi}$, SNR = {snr}, $\sigma_v ={veldisp}$ km/s".format(**pars))
    print(sfh_input is not None)

    if sfh_input is not None:
        pl.plot(sfh_input[bin1],sfh_input[bin2],'go', alpha = 1.0, mec = 'green')

    pl.savefig("{figname}_covar.png".format(**pars))
    pl.close()

# CHAIN FOR EARLIEST SFR COMPONENT
#for i in np.random.uniform(size=ndim) : pl.plot(sampler.chain[np.floor(i*nwalkers),:,-1] )
