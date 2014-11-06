import numpy as np
from sedpy import attenuation
from bsfh import priors, elines

tophat = priors.tophat


rp = {'verbose':True,
      'filename':'data/mmt/nocal/020.B192-G242.s.fits',
      'objname': 'B192-G242',
      'outfile':'results/test',
      'wlo':3750., 'whi': 7200.,
      'ftol':0.5e-5, 'maxfev':500, 'nsamplers':1,
      'walker_factor':3, 'nthreads':1, 'nburn':3 * [10], 'niter': 10,
      'initial_disp':0.1
      }

param_template = {'name':'', 'N':1, 'isfree': False,
                  'init':0.0, 'units':'',
                  'prior_function_name': None, 'prior_args': None}

default_parlist = []

###### Distance ##########
default_parlist.append({'name': 'lumdist', 'N': 1,
                        'isfree': False,
                        'init': 0.783,
                        'units': 'Mpc',
                        'prior_function': None,
                        'prior_args': None})

###### SFH ################

default_parlist.append({'name': 'mass', 'N': 1,
                        'isfree': True,
                        'init': 10e3,
                        'units': r'M$_\odot$',
                        'prior_function': tophat,
                        'prior_args': {'mini':1e2, 'maxi': 1e6}})

default_parlist.append({'name': 'tage', 'N': 1,
                        'isfree': True,
                        'init': 0.250,
                        'units': 'Gyr',
                        'prior_function':tophat,
                        'prior_args':{'mini':0.001, 'maxi':2.5}})

default_parlist.append({'name': 'zmet', 'N': 1,
                        'isfree': True,
                        'init': -0.2,
                        'units': r'$\log (Z/Z_\odot)$',
                        'prior_function': tophat,
                        'prior_args': {'mini':-1, 'maxi':0.19}})

default_parlist.append({'name': 'sfh', 'N':1,
                        'isfree': False,
                        'init': 0,
                        'units': None})

###### DUST ##################

default_parlist.append({'name': 'dust_curve', 'N': 1,
                        'isfree': False,
                        'init': attenuation.cardelli,
                        'units': None})

default_parlist.append({'name': 'dust2', 'N': 1,
                        'isfree': True,
                        'init': 0.5,
                        'units': r'$\tau_V$',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':2.5}})

default_parlist.append({'name': 'dust1', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': r'$\tau_V$',
                        'prior_function': None,
                        'prior_args': None})

default_parlist.append({'name': 'dust_tesc', 'N': 1,
                        'isfree': False,
                        'init': 0.01,
                        'units': 'Gyr',
                        'prior_function': None,
                        'prior_args': None})

default_parlist.append({'name': 'dust_type', 'N': 1,
                        'isfree': False,
                        'init': 1,
                        'units': 'NOT USED'})

###### IMF ###################

default_parlist.append({'name': 'imf_type', 'N': 1,
                        'isfree': False,
                        'init': 2, #2 = kroupa
                        'units': None})

default_parlist.append({'name': 'imf3', 'N':1,
                        'isfree': False,
                        'init': 2.3,
                        'units': None,
                        'prior_function':tophat,
                        'prior_args':{'mini':1.3, 'maxi':3.5}})

###### WAVELENGTH SCALE ######

default_parlist.append({'name': 'zred', 'N':1,
                        'isfree': True,
                        'init': 0.00001,
                        'units': None,
                        'prior_function': tophat,
                        'prior_args': {'mini':-0.001, 'maxi':0.001}})

default_parlist.append({'name': 'sigma_smooth', 'N': 1,
                        'isfree': True,
                        'init': 2.2,
                        'units': r'$\AA$',
                        'prior_function': tophat,
                        'prior_args': {'mini':1.0, 'maxi':6.0}})

default_parlist.append({'name': 'smooth_velocity', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'units': None})

default_parlist.append({'name': 'min_wave_smooth', 'N': 1,
                        'isfree': False,
                        'init': 3700.0,
                        'units': r'$\AA$'})

default_parlist.append({'name': 'max_wave_smooth', 'N': 1,
                        'isfree': False,
                        'init': 7300.0,
                        'units': r'$\AA$'})

###### CALIBRATION ###########

polyorder = 2
polymin = [-5, -50]
polymax = [5, 50]
polyinit = [0.1, 0.1]

default_parlist.append({'name': 'poly_coeffs', 'N': polyorder,
                        'isfree': True,
                        'init': polyinit,
                        'units': None,
                        'prior_function': tophat,
                        'prior_args': {'mini':polymin, 'maxi':polymax}})
    
default_parlist.append({'name': 'spec_norm', 'N':1,
                        'isfree': True,
                        'init':1,
                        'units': None,
                        'prior_function': tophat,
                        'prior_args': {'mini':0.1, 'maxi':10}})

default_parlist.append({'name': 'gp_jitter', 'N':1,
                        'isfree': True,
                        'init': 0.001,
                        'units': 'spec units',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':0.1}})

default_parlist.append({'name': 'gp_amplitude', 'N':1,
                        'isfree': True,
                        'init': 0.04,
                        'units': 'spec units',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':0.2}})

default_parlist.append({'name': 'gp_length', 'N':1,
                        'isfree': True,
                        'init': 100.0,
                        'units': r'$\AA$',
                        'prior_function': tophat,
                        'prior_args': {'mini':20.0, 'maxi':500}})

default_parlist.append({'name': 'phot_jitter', 'N':1,
                        'isfree': True,
                        'init': 0.01,
                        'units': 'mags',
                        'prior_function': tophat,
                        'prior_args': {'mini':0.0, 'maxi':0.1}})

###### EMISSION ##############

linelist = ['CaII_K', 'NaI_5891', 'NaI_5897',
            'Ha', 'NII_6585','SII_6718','SII_6732',
            'HeI_3821','HeI_4010','HeI_4027','HeI_4145','HeI_4389',
            'HeI_4473','HeI_4923','HeI_5017'
            ]
linemin = 3 * [-100] + 4 * [0.]  + 8 * [-50.0 ]
linemax = 3 * [0.] + 4 * [100.0 ] + 8 * [50.0 ]
lineinit = 3 * [-0.1 ] + 4 * [1.0 ] + 8 * [0.1 ]

nlines = len(linelist)
ewave = [elines.wavelength[l] for l in linelist]

default_parlist.append({'name': 'emission_rest_wavelengths', 'N': nlines,
                        'isfree': False,
                        'init': ewave,
                        'line_names': linelist,
                        'units': r'$\AA$'})

default_parlist.append({'name': 'emission_luminosity', 'N': nlines,
                        'isfree': True,
                        'init': lineinit,
                        'units': r'$L_\odot$',
                        'prior_function':tophat,
                        'line_names': linelist,
                        'prior_args':{'mini': linemin, 'maxi': linemax}})

default_parlist.append({'name': 'emission_disp', 'N': 1, 'isfree': True,
                        'init': 2.2,
                        'units': r'$\AA$',
                        'prior_function':tophat,
                        'prior_args':{'mini':1.0, 'maxi':6.0}})

