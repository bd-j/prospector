#Take the results from MCMC fitting of clusters
# and make diagnostic plots, or derive predictions for
# observables, etc..

import numpy as np
import matplotlib.pyplot as pl
import triangle
import pickle


def diagnostic_plots(sample_file, sps, powell_file = None,
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
    if powell_file:
        powell_results = pickle.load( open(powell_file, 'rb'))
    else:
        powell_results = None
    sample_results = pickle.load( open(sample_file, 'rb'))
    if outname is None:
        outname = sample_file#''.join(sample_file.split('.')[:-1])
    try:
        model = sample_results['model']
    except (KeyError):
        model = inmod
        model.theta_desc = sample_results['theta']
    for k, v in model.params.iteritems():
        try:
            sps.params[k] = v
        except KeyError:
            pass
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
                marker = marker[photflag], color = 'blue', linewidth = 0.5, label = 'observed')
        pl.plot(obs['wavelength'], model.model(flatchain[rindex[0],:], sps =sps)[photflag]*0.,
                marker = marker[photflag], color = 'green', alpha = 0.5, label = 'model (sample)')
        for i in range(nspec-1):
            ypred = model.model(flatchain[rindex[i+1],:], sps =sps)[photflag]
            if (photflag == 0) and sample_results.has_key('gp'):
                mask = obs['mask']
                res = sample_results['gp'].predict( (obs['spectrum'] - ypred)[mask] )
                pl.plot(obs['wavelength'][mask], np.zeros(mask.sum()) + res,
                        linewidth = 0.5, color = 'red', alpha = 0.5)
                pl.plot(obs['wavelength'], model.calibration(),
                        color = 'magenta', linewidth = 0.5)
            else:
                res = 0
                mask = np.arange(len(obs['wavelength']))
            pl.plot(obs['wavelength'][mask], ypred[mask] + res,
                    marker = marker[photflag], color = 'green', alpha = 0.5)
                      
            
        #pl.plot(obs['wavelength'], point_model[photflag],
        #        marker = marker[photflag], color = 'red', label = 'model (mean params)')
        
        #pl.plot(obs['wavelength'], point_cal,
        #        marker = marker[photflag], color = 'magenta', label = 'calibration poly (mean)', linewidth = 0.5)
        
        pl.plot(obs['wavelength'], model.model(sample_results['initial_center'], sps =sps)[photflag],
                marker = marker[photflag], color = 'cyan', label = 'minimization result', alpha = 0.5, linewidth = 0.3)
        
        pl.legend(loc = 'lower left', fontsize = 'small', bbox_to_anchor = (0.7,0.8), mode="expand")        
        pl.ylabel(ylabel[photflag])
        pl.savefig('{0}_{1}.png'.format(outname, outn[photflag]), dpi = 600)
        if photflag ==0:
            pl.xlim(3500, 4300)
            pl.savefig('{0}_{1}_blue.png'.format(outname, outn[photflag]), dpi = 300)
        pl.close()
   
    # Plot spectral and SED residuals
    #
    residuals(sample_results, sps, photflag = 0, outname = outname, nsample = nspec,
              linewidth = 0.5, alpha = 0.3, color = 'blue', marker = None)
    residuals(sample_results, sps, photflag = 1, outname = outname, nsample = 15,
              linewidth = 0.5, alpha = 0.3, color = 'blue', marker = 'o')
    
    # Plot parameters versus step
    #
    param_evol(sample_results, outname = outname)
    
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
    subtriangle(sample_results, outname = outname,
                start = start, thin = thin)
        
    return outname, sample_results, model


def model_obs(sample_results, sps, photflag = 0):
    
    flatchain = sample_results['chain'][:,start:,:]
    flatchain = flatchain.reshape(flatchain.shape[0] * flatchain.shape[1],
                                  flatchain.shape[2])
    #draw samples
    if rindex is None:
        rindex = np.random.uniform(0, flatchain.shape[0], nsample).astype( int )
    #set up the observation dictionary for spectrum or SED
    obs, outn, marker = obsdict(sample_results, photflag)

    # set up plot window and legend
    pl.figure()
    pl.plot(obs['wavelength'], obs['spectrum'],
            marker = marker, linewidth = 0.5, color = 'blue',
            label = 'observed')
    ymini = sample_results['model'].model(sample_results['initial_center'], sps =sps)
    pl.plot(obs['wavelength'], ymini[photflag],
                marker = marker, alpha = 0.5, linewidth = 0.3, color = 'cyan',
                label = 'minimization result (w/o GP)')
    
    for i in range(nsample):
        if i == 0:
            label = ['model (posterior sample)', 'calibration poly', 'GP']
        else:
            label = 3 * [None]
        
        ypred = sample_results['model'].model(flatchain[rindex[i],:], sps =sps)[photflag]
        mask = obs['mask']
        
        if (photflag == 0) and sample_results.has_key('gp'):
            
            res = sample_results['gp'].predict( (obs['spectrum'] - ypred)[mask] )
            pl.plot(obs['wavelength'][mask], np.zeros(mask.sum()) + res,
                    linewidth = 0.5, alpha = 0.5, color = 'red',
                    label = label[2])
            pl.plot(obs['wavelength'], model.calibration(),
                    linewidth = 0.5, color = 'magenta',
                    label = label[1])
        else:
            mask = np.arange(len(obs['wavelength']))
            res = 0
            
        pl.plot(obs['wavelength'][mask], ypred[mask] + res,
                marker = marker, alpha = 0.5 , color = 'green',
                label = label[0])
    
        
        pl.legend(loc = 'lower left', fontsize = 'small', bbox_to_anchor = (0.7,0.8), mode="expand")        



def residuals(sample_results, sps, photflag = 0, outname = None,
              nsample = 5, rindex = None, start = 0,
              wlo = 3600, whi = 7500, **kwargs):
    """
    Plot residuals of the observations from samples of the model posterior.
    This is done in terms of relative, uncertainty normalized, and absolute
    residuals.  Extra keywords are passed to plot(). 
    """
    
    flatchain = sample_results['chain'][:,start:,:]
    flatchain = flatchain.reshape(flatchain.shape[0] * flatchain.shape[1],
                                  flatchain.shape[2])
    #draw samples
    if rindex is None:
        rindex = np.random.uniform(0, flatchain.shape[0], nsample).astype( int )
            
    #set up the observation dictionary for spectrum or SED
    obs, outn, marker = obsdict(sample_results, photflag)
            
    #set up plot window
    fig, axes = pl.subplots(3,1)
    #draw guidelines
    [a.axhline( int(i==0), linestyle = ':', color ='black') for i,a in enumerate(axes)]
    axes[0].set_ylabel('obs/model (mean)')
    axes[0].set_ylim(0.5,1.5)
    axes[1].set_ylabel(r'(obs-model)/$\sigma$')
    axes[1].set_ylim(-10,10)
    axes[2].set_ylabel(r'(obs-model)')

    #loop over the drawn samples
    for i in range(nsample):
        theta = flatchain[rindex[i],:]
        spec = sample_results['model'].model(theta, sps =sps)[photflag]
        if  (photflag == 0) and sample_results.has_key('gp'):
            mask = obs['mask']
            res = sample_results['gp'].predict( (obs['spectrum'] - spec)[mask] )
        else:
            mask = np.ones(len(obs['wavelength']), dtype = bool)
            res = 0
        wave, ospec, mod = obs['wavelength'][mask],  obs['spectrum'][mask], (spec[mask]+res)
        axes[0].plot(wave, ospec / mod, **kwargs)
        axes[1].plot(wave, (ospec - mod) / obs['unc'][mask], **kwargs)
        axes[2].plot(wave, (ospec - mod), **kwargs)
        
    if photflag == 0:
        [a.set_xlim(wlo,whi) for a in axes]
    if outname is not None:
        fig.savefig('{0}_{1}_residuals.png'.format(outname, outn), dpi = 300)
        pl.close()
        
def obsdict(sample_results, photflag):
    if photflag == 0:
        outn = 'spectrum'
        marker = None
        obs = sample_results['obs']            
    elif photflag == 1:
        outn = 'sed'
        marker = 'o'
        obs = sample_results['obs'].copy()
        obs['wavelength'] = np.array([f.wave_effective for f in obs['filters']])
        obs['spectrum'] = 10**(0-0.4 * obs['mags'])
        obs['unc'] = obs['mags_unc'] * obs['spectrum']
        obs['mask'] = obs['mags_unc'] >= 0
        
    return obs, outn, marker
        
def param_evol(sample_results, outname =None, showpars =None, start = 0):
    """
    Plot the evolution of each parameter value with
    iteration #, for each chain.
    """
    
    chain = sample_results['chain'][:,start:,:]
    nwalk = chain.shape[0]
    parnames = np.array(theta_labels(sample_results['model'].theta_desc))

    #restrict to desired parameters
    if showpars is not None:
        ind_show = np.array([p in showpars for p in parnames], dtype = bool)
        parnames = parnames[ind_show]
        chain = chain[:,:,ind_show]

    #set up plot windows
    ndim = len(parnames)
    nx = int(np.floor(np.sqrt(ndim)))
    ny = int(np.ceil(ndim*1.0/nx))
    sz = np.array([nx,ny])
    factor = 2.0           # size of one side of one panel
    lbdim = 0.7 * factor   # size of left/bottom margin
    trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.05         # w/hspace size
    plotdim = factor * sz + factor *(sz-1)* whspace
    dim = lbdim + plotdim + trdim
    
    fig, axes = pl.subplots(nx, ny, figsize = (dim[1], dim[0]))

    #sequentially plot the chains in each parameter
    for i in range(ndim):
        ax = axes.flatten()[i]
        for j in range(nwalk):
            ax.plot(chain[j,:,i])
        ax.set_title(parnames[i])
    if outname is not None:
        fig.savefig('{0}_x_vs_step.png'.format(outname))
        pl.close()


def subtriangle(sample_results, outname =None, showpars =None,
             start = 0, thin = 1):
    """
    Make a triangle plot of the (thinned, latter) samples of the posterior
    parameter space.  Optionally make the plot only for a supplied subset
    of the parameters.
    """

    #pull out the parameter names and flatten the thinned chains
    parnames = np.array(theta_labels(sample_results['model'].theta_desc))
    flatchain = sample_results['chain'][:,start::thin,:]
    flatchain = flatchain.reshape(flatchain.shape[0] * flatchain.shape[1],
                                  flatchain.shape[2])
    truths = sample_results['initial_center']

    #restrict to parameters you want to show
    if showpars is not None:
        ind_show = np.array([p in showpars for p in parnames], dtype = bool)
        flatchain = flatchain[:,ind_show]
        truths = truths[ind_show]
        parnames= parnames[ind_show]
        
    fig = triangle.corner(flatchain, labels = parnames,
                          quantiles=[0.16, 0.5, 0.84], verbose =False,
                          truths = truths)
    if outname is not None:
        fig.savefig('{0}_triangle.png'.format(outname))
        pl.close()


def theta_labels(desc):
    """
    Using the theta_desc parameter dictionary, return a list of the model
    parameter names that has the same aorder as the sampling chain array
    """
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
