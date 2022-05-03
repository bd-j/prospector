"""Test prospector smoothing module in different modes and against FSPS
smoothing.
"""

# TODO: turn plots into asserts with tolerances.
# TODO: have some tests that do not require a python-fsps install
import numpy as np
import matplotlib.pyplot as pl
from sedpy.smoothing import smooth_fft, smooth_wave_fft, smooth_lsf_fft, smoothspec


def lsf(wave, wave0=5000, a=5e-5, b=1e-7, c=1.0, **extras):
    return c + a * (wave - wave0) + b * (wave-wave0)**2


def display(w, flib, fsmooth, outwave, msmooth):

    fig, axes = pl.subplots(3, 1, sharex=True, figsize=(14.89, 12.067))
    ax = axes[0]
    ax.plot(w, flib, label='Native MILES')
    ax.plot(w, fsmooth, label='FSPS smoothing')
    ax.plot(outwave, msmooth, label='Prospector smoothing')
    ax = axes[1]
    ax.plot(outwave, msmooth / fsmooth[good] - 1, label='(Pro-FSPS)/FSPS')
    ax = axes[2]
    ax.plot(outwave, sigma)
    ax.set_ylabel('$\sigma_\lambda$')

    [ax.set_xlim(wmin-100, wmax+100) for ax in axes]

    return fig, axes


def compare_lsf(lsf, wmin=3800, wmax=7100, **kwargs):

    import fsps
    sps = fsps.StellarPopulation(zcontinuous=1)
    wave = sps.wavelengths
    good = (wave > wmin) & (wave < wmax)
    outwave = wave[good]
    sigma = lsf(outwave, **kwargs)

    # native MILES
    sps.params['smooth_lsf'] = False
    w, flib = sps.get_spectrum(tage=1.0)

    # FSPS smoothing
    sps.params['smooth_lsf'] = True
    sps.set_lsf(outwave, 2.998e5 * sigma / outwave)
    w, fsmooth = sps.get_spectrum(tage=1.0)
    sps.params['smooth_lsf'] = False

    # prospector smoothing
    msmooth = smooth_lsf_fft(w, flib, outwave, lsf=lsf, **kwargs)

    display(w, flib, fsmooth, outwave, msmooth)


def compare_simple(resolution, wmin=3800, wmax=7100, vel=False, **kwargs):

    import fsps
    sps = fsps.StellarPopulation(zcontinuous=1)
    wave = sps.wavelengths
    good = (wave > wmin) & (wave < wmax)
    outwave = wave[good]

    # native MILES
    sps.params['smooth_lsf'] = False
    w, flib = sps.get_spectrum(tage=1.0)

    # FSPS smoothing
    sps.params['sigma_smooth'] = resolution
    if vel:
        sps.params['smooth_velocity'] = True
        sps.params['min_wave_smooth'] = 3700.0
        sps.params['max_wave_smooth'] = 7100.0
    else:
        sps.params['smooth_velocity'] = False
    w, fsmooth = sps.get_spectrum(tage=1.0)

    # Prospector smoothing
    if vel:
        smoothtype = 'vel'
    else:
        smoothtype = 'lambda'
    msmooth = smoothspec(w, flib, outwave=outwave, resolution=resolution,
                         smoothtype=smoothtype, fftsmooth=True, **kwargs)

    display(w, flib, fsmooth, outwave, msmooth)


def compare_fft(lsf, **kwargs):

    # Do a test with stellar spectrum
    from sedpy.reference_spectra import vega
    swave, spec = vega.T
    g = (swave < 2e4) & (swave > 2e3)

    sigma_out = 60.0

    # constant dlam
    out1 = smooth_wave_fft(swave[g], spec[g], outwave, sigma_out=sigma_out)
    # wave dependent dlam with lsf_fft
    out2 = smooth_lsf_fft(swave[g], spec[g], outwave, lsf=lsf, **kwargs)
    # use lsf_fft to do a constant dlam case
    out3 = smooth_lsf_fft(swave[g], spec[g], outwave, lsf=lsf, a=0, b=0, c=sigma_out)
    out4 = smooth_lsf_fft(swave[g], spec[g], outwave, lsf=lsf, a=0, b=0, c=sigma_out,
                          preserve_all_input_frequencies=True)
    # wave dependent dlam
    out5 = smooth_lsf_fft(swave[g], spec[g], outwave, lsf=lsf,
                          preserve_all_input_frequencies=True, **kwargs)

    fig, axes = pl.subplots(2, 1, sharex=True)
    ax = axes[0]
    ax.plot(outwave, out3/out1 - 1, label='(default - const) / const')
    ax.plot(outwave, out4/out1 - 1, label='(exact - const) / const')

    ax = axes[1]
    ax.plot(swave[g], spec[g], label='Native')
    #ax.plot(outwave, out1, label='wave_fft')
    ax.plot(outwave, out2, label='lsf_fft (default)')
    ax.legend()

    return fig, axes


if __name__ == "__main__":
    kwargs = {'a': 1e-5, 'b': 8e-7, 'c': 0.1, 'wave0': 5000.0}
    outwave = np.arange(2000, 10000, 0.5)

    if False:
        fig, axes = compare_lsf(lsf, **kwargs)

    if True:
        fig, axes = compare_fft(lsf, pix_per_sigma=20, **kwargs)
        [ax.legend(loc=0) for ax in axes[:-1]]

    if False:
        fig, axes = compare_simple(100.0, vel=True)
        [ax.legend(loc=0) for ax in axes]
        pl.show()
        fig, axes = compare_simple(1.0, vel=False)
        [ax.legend(loc=0) for ax in axes]
        pl.show()
