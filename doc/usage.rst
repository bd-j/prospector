User Interaction
================

The primary user interaction is through  a **parameter file**,
a python file in which several functions must be defined.
These functions are described below and are used to build the ingredients for a fit (data, model, and noise model.)
During execution any supplied command line options are parsed
-- includiing any user defined custom arguments --
and the resulting set of arguments is passed to each of these functions before fitting begins.

Command line syntax calls the parameter file and is as follows for single thread execution:

.. code-block:: shell

		python parameter_file.py

Additional command line options can be given (see below) e.g.

.. code-block:: shell

		python parameter_file.py --emcee --nwalkers=128

will cause a fit to be run using emcee with 128 walkers.

A description of the available command line options can be obtained with

.. code-block:: shell

		python parameter_file.py --help

This syntax requires that the end of the parameter file have something like the following code block at the end.

.. code-block:: python

        if __name__ == "__main__":
            import time
            from prospect.fitting import fit_model
            from prospect.io import write_results as writer
		    from prospect import prospect_args

            # Get the default argument parser
            parser = prospect_args.get_parser()
            # Add custom arguments that controll the build methods
            parser.add_argument("--custom_argument_1", ...)
            # Parse the supplied arguments, convert to a dictionary, and add this file for logging purposes
            args = parser.parse_args()
            run_params = vars(args)
            run_params["param_file"] = __file__

            # Set up an output file name, build fit ingredients, and run the fit
            ts = time.strftime("%y%b%d-%H.%M", time.localtime())
            hfile = "{0}_{1}_mcmc.h5".format(args.outfile, ts)
            obs, model, sps, noise = build_all(**run_params)
            output = fit_model(obs, model, sps, noise, **run_params)

            # Write results to output file
            writer.write_hdf5(hfile, run_params, model, obs,
                              output["sampling"][0], output["optimization"][0],
                              tsample=output["sampling"][1],
                              toptimize=output["optimization"][1],
                              sps=sps)


		
Command Line Options and Custom Arguments
-----------------------------------------
A number of default command line options are included with prospector.
These options can control the output filenames and format and some details of how the model is built and run.
However, most of the default parameters control the fitting backends.

You can inspect the default set of arguments and their default values as follows:

.. code-block:: python

		from prospect import prospect_args
		parser = prospect_args.get_parser()
		parser.print_help()

In the typical **parameter file** the arguments are converted to a dictionary and passed as keyword arguments
to all of the :py:func:`build_*` methods described below.

A user can add custom arguments that will further control the behavior of the model and data building methods.
This is done by adding arguments to the parser in the executable part of the **parameter file**.
See the argparse `documentation <https://docs.python.org/2/library/argparse.html#adding-arguments>`_
for details on adding custom arguments.

Build methods
-------------------------

The required methods in a **parameter file** for building the data and model are:


1. :py:meth:`build_obs`: 
   This function will take the command line arguments dictionary as keyword arguments
   and returns on obs dictionary (see :doc:`dataformat` .)

2. :py:meth:`build_model`:
   This function will take the command line arguments dictionary dictionary as keyword arguments
   and return an instance of a :class:`ProspectorParams` subclass, containing
   information about the parameters of the model (see :doc:`models` .)

3.  :py:meth:`build_sps`:
    This function will take the command line arguments dictionary dictionary as keyword arguments
    and return an **sps** object, which must have the method
    :py:meth:`get_spectrum` defined.  This object generally includes all the
    spectral libraries and isochrones necessary to build a model, as well as much of the model
    building code and as such has a large memory footprint.

4.  :py:meth:`build_noise`:
    This function should return a :py:class:`NoiseModel` object for the spectroscopy and/or
    photometry.  Either or both can be ``None``(the default)  in which case the likelihood
    will not include covariant noise or jitter and is equivalent to basic :math:`\chi^2`.
