import numpy as np
import matplotlib.pyplot as pl
import fsps

ztype = [False, 0, 1, 2]
zdlabel= ['ind', 'cont', 'MDF']
sps = [fsps.StellarPopulation(zcontinuous=z) for z in ztype]
tage = 1.0
p = 3.0
zmet = 3
sfig, sax = pl.subplots()
rfig, rax = pl.subplots()
sax.set_xlim(1e3, 2e4)
rax.set_xlim(1e3, 2e4)
rax.set_xscale('log')
sax.set_xscale('log')

spec = []
    
for zt, sp in zip(ztype, sps):
    sp.params['zmet'] = zmet
    sp.params['pmetals'] = p
    sp.params['logzsol'] = np.log10(sp.zlegend[sp.params['zmet']]/0.019)
    w, s = sp.get_spectrum(tage = tage, peraa=True)
    spec.append(s)
    lbl = '{}_logz{}_p{}'.format(zdlabel[zt], sp.params['logzsol'], sp.params['pmetals'])
    sax.plot(w, s, label=lbl)
    rax.plot(w, s/spec[0], label=lbl)

rax.legend()
sax.legend()
rfig.show()
sfig.show()
