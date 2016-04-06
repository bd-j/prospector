#compare a lookup table of spectra at ages and metallicities to
#calls to fsps.sps.get_spectrum() for different metallicities
import time, os, subprocess, re, sys
import numpy as np
#import matplotlib.pyplot as pl
import fsps
from prospect import sources as sps_basis
from prospect.models import sedmodel


def run_command(cmd):
    """
    Open a child process, and return its exit status and stdout.

    """
    child = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out = [s for s in child.stdout]
    w = child.wait()
    return os.WEXITSTATUS(w), out


# Check to make sure that the required environment variable is present.
try:
    ev = os.environ["SPS_HOME"]
except KeyError:
    raise ImportError("You need to have the SPS_HOME environment variable")

# Check the SVN revision number.
cmd = ["svnversion", ev]
stat, out = run_command(" ".join(cmd))
fsps_vers = int(re.match("^([0-9])+", out[0]).group(0))


sps = fsps.StellarPopulation(zcontinuous=True)
print('FSPS version = {}'.format(fsps_vers))
print('Zs={0}, N_lambda={1}'.format(sps.zlegend, len(sps.wavelengths)))
print('single age')

def spec_from_fsps(z, t, s):
    t0 = time.time()
    sps.params['logzsol'] = z
    sps.params['sigma_smooth'] = s
    sps.params['tage'] = t
    wave, spec  = sps.get_spectrum(peraa=True, tage = sps.params['tage'])
    #print(spec.shape)
    return time.time()-t0

def mags_from_fsps(z, t, s):
    t0 = time.time()
    sps.params['zred']=t
    sps.params['logzsol'] = z
    sps.params['sigma_smooth'] = s
    sps.params['tage'] = t
    mags  = sps.get_mags(tage = sps.params['tage'], redshift=0.0)
    #print(spec.shape)
    return time.time()-t0


def spec_from_ztinterp(z, t, s):
    t0 = time.time()
    sps.params['logzsol'] = z
    sps.params['sigma_smooth'] = s
    sps.params['tage'] = t
    sps.params['imf3'] = s
    spec, m, l  = sps.ztinterp(sps.params['logzsol'], sps.params['tage'], peraa=True)
    #print(spec.shape)
    return time.time()-t0
  

if sys.argv[1] == 'mags':
    from_fsps = mags_from_fsps
    print('timing get_mags')
    print('nbands = {}'.format(len(sps.get_mags(tage=1.0))))
elif sys.argv[1] == 'spec':
    from_fsps = spec_from_fsps
    print('timing get_spectrum')
elif sys.argv[1] == 'ztinterp':
    from_fsps = spec_from_ztinterp
    print('timing get_spectrum')
elif sys.argv[1] == 'sedpy':
    from sedpy import observate
    nbands = len(sps.get_mags(tage=1.0))
    fnames = nbands * ['sdss_r0']
    filters = observate.load_filters(fnames)
    
    def mags_from_sedpy(z, t, s):
        t0 = time.time()
        sps.params['logzsol'] = z
        sps.params['sigma_smooth'] = s
        sps.params['tage'] = t
        wave, spec  = sps.get_spectrum(peraa=True,
                                       tage = sps.params['tage'])
        mags = observate.getSED(wave, spec, filters)
        return time.time()-t0
    
    from_fsps = mags_from_sedpy
    
sps.params['add_neb_emission'] = False
sps.params['smooth_velocity'] = True
sps.params['sfh'] = 0

ntry = 30
zz = np.random.uniform(-1,0,ntry)
tt = np.random.uniform(0.1,4,ntry)
ss = np.random.uniform(1,2.5,ntry)

#make sure all z's already compiled
_ =[from_fsps(z, 1.0, 0.0) for z in [-1, -0.8, -0.6, -0.4, -0.2, 0.0]]
all_dur = []
print('no neb emission:')
dur_many = np.zeros(ntry)
for i in xrange(ntry):
    dur_many[i] = from_fsps(zz[i], tt[i], ss[i])
print('<t/call>={0}s, sigma_t={1}s'.format(dur_many.mean(), dur_many.std()))
all_dur += [dur_many]

print('no neb emission, no smooth:')
dur_many = np.zeros(ntry)
for i in xrange(ntry):
    dur_many[i] = from_fsps(zz[i], tt[i], 0.0)
print('<t/call>={0}s, sigma_t={1}s'.format(dur_many.mean(), dur_many.std()))
all_dur += [dur_many]

sps.params['add_neb_emission'] = True    
print('neb emission:')
dur_many = np.zeros(ntry)
for i in xrange(ntry):
    dur_many[i] = from_fsps(zz[i], tt[i], ss[i])
print('<t/call>={0}s, sigma_t={1}s'.format(dur_many.mean(), dur_many.std()))
all_dur += [dur_many]
    
print('neb emission, no smooth:')
dur_many = np.zeros(ntry)
for i in xrange(ntry):
    dur_many[i] = from_fsps(zz[i], tt[i], 0.0)
print('<t/call>={0}s, sigma_t={1}s'.format(dur_many.mean(), dur_many.std()))
all_dur += [dur_many]


