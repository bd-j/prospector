from copy import deepcopy
import timeit, time, sys
import numpy as np

import fsps
from prospect.sources import FastStepBasis, CSPBasis


sps = fsps.StellarPopulation(zcontinuous=1)
libs = [l.upper() for l in sps.libraries]

def get_model(sps, **kwargs):
    try:
        # For SSPBasis derived objects
        sps.update(**kwargs)
    except(AttributeError):
        # For StellarPopulation and CSPBasis objects
        for k, v in kwargs.iteritems():
            try:
                sps.params[k] = v
            except:
                pass
    out = sps.get_spectrum(tage=sps.params['tage'])
    return out


def call_duration(sps, ntry, **params):
    # build cached SSPs without getting charged for the time.
    junk = [get_model(sps, logzsol=[z], **params) for z in np.linspace(-1, 0, 12)]
    #print('done_setup')
    ts = time.time()
    for i in range(ntry):
        _ = get_model(sps, logzsol=[np.random.uniform(-0.8, -0.2)], **params)
    dur = time.time() - ts
    return dur / ntry


def make_agebins(nbin=5, minage=7.0, **extras):
    tuniv = 13.7e9
    allages = np.linspace(minage, np.log10(tuniv), nbin)
    allages = np.insert(allages, 0, 0.)
    agebins = np.array([allages[:-1], allages[1:]]).T
    return agebins



if __name__ == "__main__":

    step_params = {'agebins':[[]],
                   'mass': [],
                   'tage': np.array([13.7]),
                   'pmetals': np.array([-99])
                   }

    csp_params = {'tage': np.array([10.0]),
                  'sfh': np.array([4.0]),
                  'mass': np.array([1.0]),
                  'pmetals': np.array([-99])
                  }

    w = ['WITHOUT', 'WITH']
    ntry = 100
    zlist = [1, 2]
    nlist = [False, True]

    print("Using {} isochrones and {} spectra.\nAsking for single ages.".format(*libs))

    # FSPS
    string = "StellarPopulation takes {:7.5f}s per call {} nebular emission and zcontinuous={}."
    params = deepcopy(csp_params)
    for zcont in zlist:
        print("\n")
        for neb in nlist:
            sps = fsps.StellarPopulation(zcontinuous=zcont)
            dur = call_duration(sps, ntry, add_neb_emission=[neb], **params)
            print(string.format(dur, w[int(neb)], zcont))

    # CSP
    string = "CSPBasis takes {:7.5f}s per call {} nebular emission and zcontinuous={}."
    params = deepcopy(csp_params)
    for zcont in zlist:
        print("\n")
        for neb in nlist:
            sps = CSPBasis(zcontinuous=zcont)
            dur = call_duration(sps, ntry, add_neb_emission=[neb], **params)
            print(string.format(dur, w[int(neb)], zcont))

    # Step SFH
    nbin = 10

    params = deepcopy(step_params)
    params['agebins'] = make_agebins(nbin)
    params['mass'] = np.ones(nbin) * 1.0
    string = "FastStepBasis ({} bins) takes {:7.5f}s per call {} nebular emission and zcontinuous={}."

    for zcont in zlist:
        print("\n")
        for neb in nlist:
            sps = FastStepBasis(zcontinuous=zcont)
            dur = call_duration(sps, ntry, add_neb_emission=[neb], **params)
            #print(sps.params, sps.ssp.params['add_neb_emission'])
            print(string.format(nbin, dur, w[int(neb)], zcont))




   # sys.exit()


    # Now time calls for random Z (which always causes dirtiness=1)
    #setup = "from __main__ import test; import numpy as np"
    #call = "out=get_model(sps, logzsol=[np.random.uniform(-1, 0)], **params)"
    #dur = timeit.timeit(call, setup=setup, number=100)

