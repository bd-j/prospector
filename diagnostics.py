#Take the results from MCMC fitting of clusters
# and make diagnostic plots, or derive predictions for
# observables, etc..

import numpy as np
import matplotlib.pyplot as pl
import triangle
import pickle


def diagnostic_plots(powell_file, sample_file, sps, 
                     nspec = 5, thin = 10, start = 0, outname = None, inmod = None):
    """Plots a number of diagnostics.  These include:
            spectrum -
                the observed spectrum, the spectra produced from a given number of samples of the
                posterior parameter space, the spectrum produced from marginalized means of each
                parameter, the spectrum at the initial position from Powell minimization, and the
                applied calibration model.
            spectrum_blue -
                same as above but for the blue region of the spectrum                       
            sed -
                as for spectrum, but f_nu at the effective wavelngth of the
                filters is shown instead.
            spectrum_residuals -
                plots of spectrum residuals for a given number of samples of the posterior
            sed_residuals -
                broadband photometry residuals, in units of f_nu
            x_vs_step -
                the evolution of the walkers in each parameter as a function of iteration
            lnp_vs_step -
                the evolution of the walkers in likelihood
            triangle  -
                a triangle plot of paramter covariances
            
    

    """
    
    powell_results = pickle.load( open(powell_file, 'rb'))
    sample_results = pickle.load( open(sample_file, 'rb'))
    if outname is None:
        outname = sample_file#''.join(sample_file.split('.')[:-1])
    try:
        model = sample_results['model']
    except (KeyError):
        model = inmod
        model.theta_desc = sample_results['theta']
    for k, v in model.sps_fixed_params.iteritems():
        sps.params[k] = v
    parnames = theta_labels(model.theta_desc)
        
    chain = sample_results['chain'][:,start::thin,:]
    nwalk, nchain, ndim = chain.shape
    flatchain = chain.reshape(nwalk * nchain, ndim)
    rindex = np.random.uniform(0,nwalk * nchain, nspec).astype( int )
    point = chain.mean(axis = 0).mean(axis = 0)
    point_model = model.model(point, sps =sps)

    pflag, pcal = [], [0,0]
    sobs = sample_results['obs']
    if 'wavelength' in sobs:
        pflag += [0]
        pcal[0] = model.calibration()
    #make an observed SED
    if sobs['filters'] is not None:
        pobs = sobs.copy()
        pobs['wavelength'] = np.array([f.wave_effective for f in sobs['filters']])
        pobs['spectrum'] = 10**(0-0.4 * sobs['mags'])
        pobs['unc'] = sobs['mags_unc'] * pobs['spectrum']
        pflag += [1]
        pcal[1] = np.ones(len(pobs['wavelength']))
        
    allobs = [sobs, pobs]
    ylabel = [r'L$_\odot/\AA$', r'f$_\nu$ (Maggies)']
    outn = ['spectrum', 'sed']
    marker = [None, 'o']
    
    # Plot spectra and SED
    #
    for photflag in pflag:
        obs = allobs[photflag]
        point_cal = pcal[photflag]
        pl.figure()
        pl.plot(obs['wavelength'], obs['spectrum'],
                marker = marker[photflag], color = 'blue', linewidth = 1.5, label = 'observed')
        pl.plot(obs['wavelength'], model.model(flatchain[rindex[0],:], sps =sps)[photflag],
                marker = marker[photflag], color = 'green', alpha = 0.5, label = 'model (sample)')
        for i in range(nspec-1):
            pl.plot(obs['wavelength'], model.model(flatchain[rindex[i+1],:], sps =sps)[photflag],
                    marker = marker[photflag], color = 'green', alpha = 0.5)
        pl.plot(obs['wavelength'], point_model[photflag],
                marker = marker[photflag], color = 'red', label = 'model (mean params)')
        pl.plot(obs['wavelength'], point_cal,
                marker = marker[photflag], color = 'magenta', label = 'calibration (mean)')
        pl.plot(obs['wavelength'], model.model(sample_results['initial_center'], sps =sps)[photflag],
                marker = marker[photflag], color = 'cyan', label = 'initial', alpha = 0.5)
        pl.legend(loc = 'lower left', fontsize = 'small', bbox_to_anchor = (0.7,0.8), mode="expand")
        pl.ylabel(ylabel[photflag])
        pl.savefig('{0}_{1}.png'.format(outname, outn[photflag]), dpi = 600)
        if photflag ==0:
            pl.xlim(3750, 4300)
            pl.savefig('{0}_{1}_blue.png'.format(outname, outn[photflag]), dpi = 300)
        pl.close()
   
    # Plot spectral and SED residuals
    #
    for photflag in pflag:
        obs = allobs[photflag]
        fig, axes = pl.subplots(3,1)
        axes[0].axhline(1,linestyle = ':', color ='black')
        axes[1].axhline(0,linestyle = ':', color ='black')
        axes[2].axhline(0,linestyle = ':', color ='black')
        axes[0].set_ylabel('obs/model (mean)')
        axes[0].set_ylim(0.5,1.5)
        axes[1].set_ylabel(r'(obs-model)/$\sigma$')
        axes[1].set_ylim(-10,10)
        axes[2].set_ylabel(r'(obs-model)')
 
        for i in range(nspec-1):
            spec = model.model(flatchain[rindex[i+1],:], sps =sps)[photflag]
            axes[0].plot(obs['wavelength'], obs['spectrum'] / spec,
                         marker = marker[photflag], alpha = 0.3, color = 'blue')
            axes[1].plot(obs['wavelength'], (obs['spectrum'] - spec) / obs['unc'],
                         marker = marker[photflag], alpha = 0.3, color = 'blue')
            axes[2].plot(obs['wavelength'], (obs['spectrum'] - spec),
                         marker = marker[photflag], alpha = 0.3, color = 'blue')
        
        if photflag == 0:
            [a.set_xlim(3700,7200) for a in axes]
        fig.savefig('{0}_{1}_residuals.png'.format(outname, outn[photflag]), dpi = 300)
        fig.clf()

    # Plot parameters versus step
    #
    nx = int(np.floor(np.sqrt(ndim)))
    ny = int(np.ceil(ndim*1.0/nx))
    sz = np.array([nx,ny])
    factor = 2.0           # size of one side of one panel
    lbdim = 0.7 * factor   # size of left/bottom margin
    trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.05         # w/hspace size
    plotdim = factor * sz + factor *(sz-1)* whspace
    dim = lbdim + plotdim + trdim
    
    print(nx,ny)
    fig, axes = pl.subplots(nx, ny, figsize = (dim[1], dim[0]))

    for i in range(ndim):
        ax = axes.flatten()[i]
        for j in range(nwalk):
            ax.plot(sample_results['chain'][j,:,i])
        ax.set_title(parnames[i])
        #ax.set_xlabel('step #')
    fig.savefig('{0}_x_vs_step.png'.format(outname))
    fig.clf()
    
    # Plot lnprob vs step (with a zoom-in)
    #
    pl.figure(1)
    pl.clf()
    for j in range(nwalk):
        pl.plot(sample_results['lnprobability'][j,:])
        pl.ylabel('lnP')
        pl.xlabel('step #')
    pl.savefig('{0}_lnP_vs_step.png'.format(outname))
    yl = sample_results['lnprobability'].max() + np.array([-3.0 * sample_results['lnprobability'][:,-1].std(), 10])
    pl.ylim(yl[0], yl[1])
    pl.savefig('{0}_lnP_vs_step_zoom.png'.format(outname))
    pl.close()
    
    # Triangle plot
    #
    fig = triangle.corner(flatchain,labels = parnames,
                          quantiles=[0.16, 0.5, 0.84], verbose =False,
                          truths = sample_results['initial_center'])
    fig.savefig('{0}_triangle.png'.format(outname))

    return powell_results, sample_results, model

def theta_labels(desc):
    label, index = [], []
    for p in desc.keys():
        nt = desc[p]['N']
        name = p
        if p is 'amplitudes': name = 'A'
        if nt is 1:
            label.append(name)
            index.append(desc[p]['i0'])
        else:
            for i in xrange(nt):
                label.append(name+'{0}'.format(i+1))
                index.append(desc[p]['i0']+i)

    return [l for (i,l) in sorted(zip(index,label))]


def sample_photometry(sample_results, sps, filterlist, start = 0, wthin = 16, tthin = 10):

    chain, model = sample_results['chain'], sample_results['model']
    for k, v in model.sps_fixed_params.iteritems():
        sps.params[k] = v
    model.filters = filterlist
    nwalkers, nt, ndim = chain.shape
    wit = range(0,nwalkers,wthin) #walkers to use
    tit = range(start, nt, thin) #time steps to use
    phot = np.zeros( len(wit), len(tit), len(filterlist)) #build storage
    for i in wit:
        for j in tit:
            s, p, m = model.model(chain[i,j,:], sps =sps)
            phot[i,j,:] = p
            #mass[i,j] = m
    return phot, wit, tit
