import matplotlib.pyplot as pl
import numpy as np
import pickle
import triangle



def results_plot(sample_file, outname = 'demo', nspec = 10, nsfh = 20, sps =None, start =0, thin =1):
    
    sample_results = pickle.load( open(sample_file, 'rb'))
    model = sample_results['model']
    parnames = theta_labels(model.theta_desc)
    chain = sample_results['chain'][start::thin,:]
    nchain, ndim = chain.shape
    flatchain = chain.reshape(nchain, ndim)
    obs = sample_results['obs']
    model.sps = sps
#plot a triangle plot
    fig = triangle.corner(flatchain,labels = parnames,
                        quantiles=[0.16, 0.5, 0.84], verbose =False,
                        truths = sample_results['mock_input_theta'])
    fig.savefig('{0}_triangle.png'.format(outname))
    pl.close()

#plot SFHs
    pl.figure(1)
    pl.clf()
    rindex = np.random.uniform(0, nchain, nsfh).astype( int )
    tage = np.log10(model.params['tage'] * 1e9)

    pl.plot(tage, flatchain[rindex[0],:] * 0., color = 'gray', label = 'posterior sample')
    for i in range(nspec-1):
        pl.plot(tage, flatchain[rindex[i],:], color = 'gray', alpha = 0.5)
    pl.plot(tage, sample_results['mock_input_theta'], '-o',
            color = 'red', label = 'input/truth', linewidth = 2)
    pl.plot(tage, sample_results['initial_center'], '-',
            color = 'cyan', label = 'minimization result')
    

    pl.xlabel('log Age (years)')
    pl.ylabel(r'Stellar mass formed (M$_\odot$)')
    pl.legend(loc = 'upper right')
    pl.xlim(6.5, 10.5)
    pl.savefig('{0}_sfh.png'.format(outname))
    pl.close(1)
    
#plot spectrum
    pl.figure(1)
    rindex = np.random.uniform(0, nchain, nspec).astype( int )

    pl.plot(obs['wavelength'], obs['spectrum'],
            color = 'blue', linewidth = 1.5, label = 'observed')
    pl.plot(obs['wavelength'], model.model(flatchain[rindex[0],:], sps =sps)[0]*0.,
            color = 'green', alpha = 0.5, label = 'model ({0} samples)'.format(nspec))
    for i in range(nspec-1):
        pl.plot(obs['wavelength'],model.model(flatchain[rindex[i],:], sps =sps)[0],
                color = 'green', alpha = 0.5)

    pl.ylabel(r'L$_\odot/\AA$')
    pl.xlabel(r'wavelength ($\AA$)')
    pl.xlim(3000,10000)
    pl.ylim(0,0.05)
    pl.legend()
    pl.savefig('{0}_spectrum.png'.format(outname), dpi = 300)
    pl.close()
    

def theta_labels(desc):
    label, index = [], []
    for p in desc.keys():
        nt = desc[p]['N']
        name = p
        #if p is 'mass': name = 'm'
        if nt is 1:
            label.append(name)
            index.append(desc[p]['i0'])
        else:
            for i in xrange(nt):
                label.append(r'${1}_{{{0}}}$'.format(i+1, name))
                index.append(desc[p]['i0']+i)

    return [l for (i,l) in sorted(zip(index,label))]
