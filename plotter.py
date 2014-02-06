import numpy as np
import matplotlib.pyplot as pl
import triangle


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


def theta_triangle(chain, theta_desc, outname = None, truths = None):
    pl.figure(1)
    pl.clf()
    labels = theta_labels(theta_desc)
    figure = triangle.corner(chain,
                             labels = labels,
                             quantiles=[0.16, 0.5, 0.84],
                             truths = truths)

    if outname is not None: figure.savefig('{0}_triangle.png'.format(outname))

def spec_plotter(blobs, wave, truth = None, outname = None):
    pl.figure(1)
    pl.clf()
    if truth is not None:
        pl.plot(wave, truth, linewidth = 1, label ='truth')
    for b in blobs:
        pl.plot(wave, b[0][0])
    pl.legend()
    pl.xlim(1e3,2.5e4)

def walker_prob(sample_lnprob, choose = None, outname = None):
    pl.figure(1)
    pl.clf()
    nwalker = sample_lnprob.shape[0]
    if choose is not None:
        wnum = (np.random.uniform(0,1,choose) * nwalker).astype(int)
    else:
        wnum = np.arange(nwalker)
    for i in wnum: pl.plot(sample_lnprob[i,:])
    pl.ylabel('lnP')
    pl.xlabel('step #')
    if outname is not None : pl.savefig('{0}_lnPvsStep.png'.format(outname), dpi=300, bbox_inches='tight')


def prob_vs_x(sampler, indices, choose = None, outname = None):
    nwalker = sampler.chain.shape[0]
    if choose is not None:
        wnum = (np.random.uniform(0,1,choose) * nwalker).astype(int)
    else:
        wnum = np.arange(nwalker)

    for i in indices:
        pl.figure(i+1)
        pl.clf()
        for j in wnum: pl.plot(sampler.chain[j,:,i], sampler.lnprobability[j,:])
        pl.ylabel('lnP')
        pl.xlabel(r'$\theta_{0}$'.format(i))
        #pl.ylim(-2365, -2350)
        if outname is not None : pl.savefig('{0}_lnPvsTheta_{1}.png'.format(outname, i), dpi=300, bbox_inches='tight')
