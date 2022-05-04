#Take the results from MCMC fitting of clusters
# and make diagnostic plots, or derive predictions for
# observables, etc..

import numpy as np
import matplotlib.pyplot as pl
import triangle
import pickle
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO


def diagnostic_plots(sample_file, sps, model_file=None,
                     powell_file=None, inmod=None,
                     showpars=None,
                     nspec=5, thin=10, start=0, outname=None):
    """
    Plots a number of diagnostics.  These include:
        spectrum -
            the observed spectrum, the spectra produced from a given number of samples of the
            posterior parameter space, the spectrum produced from marginalized means of each
            parameter, the spectrum at the initial position from Powell minimization, and the
            applied calibration model.
        spectrum_blue -
            same as above but for the blue region of the spectrum                       
        sed -
            as for spectrum, but f_nu at the effective wavelength of the
            filters is shown instead.
        stars -
            just the stellar  dust model for samples of the posterior.
        spectrum_residuals -
            plots of spectrum residuals for a given number of samples of the posterior
        sed_residuals -
            broadband photometry residuals, in units of f_nu
        x_vs_step -
            the evolution of the walkers in each parameter as a function of iteration
        lnp_vs_step -
            the evolution of the walkers in likelihood
        triangle  -
                a corner plot of parameter covariances    
    """

    #read results and set up model
    if outname is None:
        outname = sample_file#''.join(sample_file.split('.')[:-1])
    sample_results, pr, model = read_pickles(sample_file, model_file=model_file,
                                             powell_file=powell_file, inmod=inmod)
    for k, v in model.params.iteritems():
        try:
            sps.params[k] = v
        except KeyError:
            pass

    ## Plot spectra and SEDs
    ##
    #rindex = model_obs(sample_results, sps, photflag=0, outname=outname, nsample=nspec,
    #                   wlo=3400, whi =10e3, start=start)
    #_ = model_obs(sample_results, sps, photflag=0, outname=outname, rindex=rindex,
    #             wlo=3600, whi=4450, extraname='_blue', start=start)
    #_ = model_obs(sample_results, sps, photflag=1, outname=outname, nsample=15,
    #              wlo=2500, whi=8.5e3, start=start)

    #stellar_pop(sample_results, sps, outname=outname, nsample=nspec,
    #              wlo=3500, whi=9.5e3, start=start,
    #              alpha = 0.5, color = 'green')
    
    ## Plot spectral and SED residuals
    ##
    #residuals(sample_results, sps, photflag=0, outname=outname, nsample=nspec,
    #          linewidth=0.5, alpha=0.3, color='blue', marker=None, start=start, rindex=rindex)
    #residuals(sample_results, sps, photflag=1, outname = outname, nsample = 15,
    #          linewidth=0.5, alpha=0.3, color='blue', marker='o', start=start, rindex=rindex)
    
    ## Plot parameters versus step
    ##
    param_evol(sample_results, outname=outname, showpars=showpars)
    
    ## Plot lnprob vs step (with a zoom-in)
    ##
    pl.figure()
    pl.clf()
    nwalk = sample_results['lnprobability'].shape[0]
    for j in range(nwalk):
        pl.plot(sample_results['lnprobability'][j,:])
        pl.ylabel('lnP')
        pl.xlabel('step #')
    pl.savefig('{0}.lnP_vs_step.png'.format(outname))
    pl.close()
    #yl = sample_results['lnprobability'].max() + np.array([-3.0 * sample_results['lnprobability'][:,-1].std(), 10])
    #pl.ylim(yl[0], yl[1])
    #pl.savefig('{0}.lnP_vs_step_zoom.png'.format(outname))
    #pl.close()
    
    ## Triangle plot
    ##
    subtriangle(sample_results, outname=outname,
                showpars=showpars,
                start=start, thin=thin)
        
    return outname, sample_results, model, pr


def model_comp(theta, model, sps, photflag=0, inlog=True):
    """
    Generate and return various components of the total model for a
    given set of parameters
    """
    obs, _, _ = obsdict(model.obs, photflag=photflag)
    mask = obs['mask']
    mu = model.mean_model(theta, sps=sps)[photflag][mask]
    spec = obs['spectrum'][mask]
    wave = obs['wavelength'][mask]
    
    if photflag == 0:
        cal = model.calibration()[mask]
        try:
            #model.gp.sigma = obs['unc'][mask]/mu
            s = model.params['gp_jitter']
            a = model.params['gp_amplitude']
            l = model.params['gp_length']
            model.gp.factor(s, a, l, check_finite = False, force=True)
            if inlog:
                mu = np.log(mu)
                delta = model.gp.predict(spec - mu - cal)
            else:
                delta = model.gp.predict(spec - mu*cal)
        except:
            delta = 0
    else:
        mask = np.ones(len(obs['wavelength']), dtype= bool)
        cal = np.ones(len(obs['wavelength']))
        delta = np.zeros(len(obs['wavelength']))
        
    return mu, cal, delta, mask, wave


def model_obs(sample_results, sps, photflag=0, outname=None,
              start=0, rindex =None, nsample=10,
              wlo=3500, whi=9e3, extraname=''):

    """
    Plot the observed spectrum and overlay samples of the model
    posterior, including different components of that model.
    """
    
    title = ['Spectrum', 'SED (photometry)']
    start = np.min([start, sample_results['chain'].shape[1]])
    flatchain = sample_results['chain'][:,start:,:]
    flatchain = flatchain.reshape(flatchain.shape[0] * flatchain.shape[1],
                                  flatchain.shape[2])
    
    # draw samples
    if rindex is None:
        rindex = np.random.uniform(0, flatchain.shape[0], nsample).astype( int )
    else:
        nsample = len(rindex)
    # set up the observation dictionary for spectrum or SED
    obs, outn, marker = obsdict(sample_results, photflag)

    # set up plot window and plot data
    pl.figure()
    pl.axhline( 0, linestyle = ':', color ='black') 
    pl.plot(obs['wavelength'], obs['spectrum'],
            marker=marker, linewidth=0.5,
            color='blue', label='observed')
    
    # plot the minimization result
    theta = sample_results['initial_center']
    ypred, res, cal, mask, spop = model_components(theta, sample_results, obs, sps, photflag=photflag)
    pl.plot(obs['wavelength'][mask], ypred + res,
            marker=marker, alpha=0.5, linewidth=0.3,
            color='cyan', label='minimization result')
    
    # loop over drawn samples and plot the model components
    label = ['full model', 'calib.', 'GP']
    for i in range(nsample):
        theta = flatchain[rindex[i],:]
        ypred, res, cal, mask, spop = model_components(theta, sample_results, obs, sps, photflag=photflag)
        
        pl.plot(obs['wavelength'][mask], np.zeros(mask.sum()) + res,
                linewidth=0.5, alpha=0.5, color='red', label=label[2])
        pl.plot(obs['wavelength'], cal * sample_results['model'].params.get('linescale', 1.0),
                linewidth=0.5, color='magenta', label=label[1])
        pl.plot(obs['wavelength'][mask], ypred + res,
                marker=marker, alpha=0.5 , color='green', label=label[0])
        label = 3 * [None]
        
    pl.legend(loc=0, fontsize='small')
    pl.xlim(wlo, whi)
    pl.xlabel(r'$\AA$')
    pl.ylabel('Rate')
    pl.title(title[photflag])
    if outname is not None:
        pl.savefig('{0}.{1}{2}.png'.format(outname, outn, extraname), dpi=300)
        pl.close()
    return rindex

def stellar_pop(sample_results, sps, outname=None, normalize_by=None,
                start=0, rindex=None, nsample=10,
                wlo=3500, whi=9e3, extraname='', **kwargs):
    """
    Plot samples of the posterior for just the stellar population and
    dust model.
    """
    start = np.min([start, sample_results['chain'].shape[1]])
    flatchain = sample_results['chain'][:,start:,:]
    flatchain = flatchain.reshape(flatchain.shape[0] * flatchain.shape[1],
                                  flatchain.shape[2])
    # draw samples
    if rindex is None:
        rindex = np.random.uniform(0, flatchain.shape[0], nsample).astype( int )
    # set up the observation dictionary for spectrum or SED
    obs, outn, marker = obsdict(sample_results, 0)

    # set up plot window 
    pl.figure()
    pl.axhline( 0, linestyle=':', color='black')
    
    # loop over drawn samples and plot the model components
    label = ['Stars & Dust']
    xl = ''
    for i in range(nsample):
        theta = flatchain[rindex[i],:]
        ypred, res, cal, mask, spop = model_components(theta, sample_results, obs, sps, photflag=0)
        if normalize_by is not None:
            spop /= spop[normalize_by]
            xl = '/C'
        pl.plot(obs['wavelength'], spop,
                label = label[0], **kwargs)
        label = 3 * [None]
        
    pl.legend(loc = 0, fontsize = 'small')
    pl.xlim(wlo, whi)
    pl.xlabel(r'$\AA$')
    pl.ylabel(r'L$_\lambda {0}$ (L$_\odot/\AA$)'.format(xl))
    if outname is not None:
        pl.savefig('{0}.{1}{2}.png'.format(outname, 'stars', extraname), dpi=300)
        pl.close()


def residuals(sample_results, sps, photflag=0, outname=None,
              nsample=5, rindex=None, start=0,
              wlo=3600, whi=7500, **kwargs):
    """
    Plot residuals of the observations from samples of the model
    posterior.  This is done in terms of relative, uncertainty
    normalized, and absolute residuals.  Extra keywords are passed to
    plot().
    """
    
    start = np.min([start, sample_results['chain'].shape[1]])
    flatchain = sample_results['chain'][:,start:,:]
    flatchain = flatchain.reshape(flatchain.shape[0] * flatchain.shape[1],
                                  flatchain.shape[2])
    # draw samples
    if rindex is None:
        rindex = np.random.uniform(0, flatchain.shape[0], nsample).astype( int )
    nsample = len(rindex)
        
    # set up the observation dictionary for spectrum or SED
    obs, outn, marker = obsdict(sample_results, photflag)
            
    # set up plot window
    fig, axes = pl.subplots(3,1)
    # draw guidelines
    [a.axhline( int(i==0), linestyle=':', color='black') for i,a in enumerate(axes)]
    axes[0].set_ylabel('obs/model')
    axes[0].set_ylim(0.5,1.5)
    axes[0].set_xticklabels([])
    axes[1].set_ylabel(r'(obs-model)/$\sigma$')
    axes[1].set_ylim(-10,10)
    axes[1].set_xticklabels([])
    axes[2].set_ylabel(r'(obs-model)')
    axes[2].set_xlabel(r'$\AA$')
    
    # loop over the drawn samples
    for i in range(nsample):
        theta = flatchain[rindex[i],:]
        ypred, res, cal, mask, spop = model_components(theta, sample_results, obs, sps, photflag=photflag)
        wave, ospec, mod = obs['wavelength'][mask],  obs['spectrum'][mask], (ypred + res)
        axes[0].plot(wave, ospec / mod, **kwargs)
        axes[1].plot(wave, (ospec - mod) / obs['unc'][mask], **kwargs)
        axes[2].plot(wave, (ospec - mod), **kwargs)
        
    if photflag == 0:
        [a.set_xlim(wlo,whi) for a in axes]

    fig.subplots_adjust(hspace =0)
    if outname is not None:
        fig.savefig('{0}.{1}_residuals.png'.format(outname, outn), dpi=300)
        pl.close()
        
def obsdict(inobs, photflag):
    """
    Return a dictionary of observational data, generated depending on
    whether you're matching photometry or spectroscopy.
    """
    obs = inobs.copy()
    if photflag == 0:
        outn = 'spectrum'
        marker = None
    elif photflag == 1:
        outn = 'sed'
        marker = 'o'
        obs['wavelength'] = np.array([f.wave_effective for f in obs['filters']])
        obs['spectrum'] = 10**(0-0.4 * obs['mags'])
        obs['unc'] = obs['mags_unc'] * obs['spectrum']
        obs['mask'] = obs['mags_unc'] > 0
        
    return obs, outn, marker
        
def param_evol(sample_results, outname=None, showpars=None, start=0):
    """
    Plot the evolution of each parameter value with iteration #, for
    each chain.
    """
    
    chain = sample_results['chain'][:,start:,:]
    nwalk = chain.shape[0]
    parnames = np.array(theta_labels(sample_results['model'].theta_desc))

    #restrict to desired parameters
    if showpars is not None:
        ind_show = np.array([p in showpars for p in parnames], dtype= bool)
        parnames = parnames[ind_show]
        chain = chain[:,:,ind_show]

    #set up plot windows
    ndim = len(parnames)
    nx = int(np.floor(np.sqrt(ndim)))
    ny = int(np.ceil(ndim*1.0/nx))
    sz = np.array([nx,ny])
    factor = 3.0           # size of one side of one panel
    lbdim = 0.2 * factor   # size of left/bottom margin
    trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.05*factor         # w/hspace size
    plotdim = factor * sz + factor *(sz-1)* whspace
    dim = lbdim + plotdim + trdim
    
    fig, axes = pl.subplots(nx, ny, figsize = (dim[1], dim[0]))
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb[1], bottom=lb[0], right=tr[1], top=tr[0],
                        wspace=whspace, hspace=whspace)

    #sequentially plot the chains in each parameter
    for i in range(ndim):
        ax = axes.flatten()[i]
        for j in range(nwalk):
            ax.plot(chain[j,:,i])
        ax.set_title(parnames[i])
    if outname is not None:
        fig.savefig('{0}.x_vs_step.png'.format(outname))
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
        if p is 'amplitudes':
            name = 'A'
        if nt is 1:
            label.append(name)
            index.append(desc[p]['i0'])
        else:
            for i in xrange(nt):
                label.append(name+'{0}'.format(i+1))
                index.append(desc[p]['i0']+i)

    return [l for (i,l) in sorted(zip(index,label))]

def sample_photometry(sample_results, sps, filterlist,
                      start=0, wthin=16, tthin=10):

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
            s, p, m = model.model(chain[i,j,:], sps=sps)
            phot[i,j,:] = p
            #mass[i,j] = m
    return phot, wit, tit

## All this because scipy changed
# the name of one class, which shouldn't even be a class.

renametable = {
    'Result': 'OptimizeResult',
    }

def mapname(name):
    if name in renametable:
        return renametable[name]
    return name

def mapped_load_global(self):
    module = mapname(self.readline()[:-1])
    name = mapname(self.readline()[:-1])
    klass = self.find_class(module, name)
    self.append(klass)

def load(file):
    unpickler = pickle.Unpickler(file)
    unpickler.dispatch[pickle.GLOBAL] = mapped_load_global
    return unpickler.load()
