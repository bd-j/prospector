User Interaction
================

The primary user interaction is through  a **parameter file**, a python file in
which several functions must be defined. These functions are described below and
are used to build the ingredients for a fit (data, model, and noise model.)
During execution any supplied command line options are parsed -- includiing any
user defined custom arguments -- and the resulting set of arguments is passed to
each of these functions before fitting begins.

Command line syntax calls the parameter file and is as follows for single thread
execution:

.. code-block:: shell

		python parameter_file.py --dynesty

Additional command line options can be given (see below) e.g.

.. code-block:: shell

		python parameter_file.py --emcee --nwalkers=128

will cause a fit to be run using emcee with 128 walkers.

A description of the available command line options can be obtained with

.. code-block:: shell

		python parameter_file.py --help

All of this command line syntax requires that the end of the parameter file have
something like the following code block at the end, which reads command line
arguments, runs all the `build_` methods (see below), conducts the fit, and
writes output.

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

            # build the fit ingredients
            obs, model, sps, noise = build_all(**run_params)
            run_params["sps_libraries"] = sps.ssp.libraries

            # Set up an output file name and run the fit
            ts = time.strftime("%y%b%d-%H.%M", time.localtime())
            hfile = "{0}_{1}_mcmc.h5".format(args.outfile, ts)
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
   and returns a list of `Observation` instances (see :doc:`dataformat` .)

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
    This function, if present, should add a :py:class:`NoiseModel` object to the
    spectroscopy and/or photometry. If not present the likelihood will not
    include covariant noise or jitter and is equivalent to basic :math:`\chi^2`.



Using MPI
---------

For large galaxy samples we recommend conducting a fit for each object entirely
independently on individual CPU cores. However, for a small number of objects or
during testing it can be helpful to decrease the elapsed wall time for a single
fit. Message Passing Interface (MPI) can be used to parallelize the fit for a
single object over many CPU cores.  This will reduce the wall time required for
a single fit, but will not reduce the total CPU uptime (and when using dynesty
might actually increase the total CPU usage).

To use MPI a "pool" of cores must be made available; each core will instantiate
the fitting ingredients separately, and a single core in the pool will then
conduct the fit, distributing likelihood requests to the other cores in the
pool.  This requires changes to the final code block that instantiates and runs the fit:

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

            # Build the fit ingredients on each process
            obs, model, sps, noise = build_all(**run_params)
            run_params["sps_libraries"] = sps.ssp.libraries

            # Set up MPI communication
            try:
                import mpi4py
                from mpi4py import MPI
                from schwimmbad import MPIPool

                mpi4py.rc.threads = False
                mpi4py.rc.recv_mprobe = False

                comm = MPI.COMM_WORLD
                size = comm.Get_size()

                withmpi = comm.Get_size() > 1
            except ImportError:
                print('Failed to start MPI; are mpi4py and schwimmbad installed? Proceeding without MPI.')
                withmpi = False

        # Evaluate SPS over logzsol grid in order to get necessary data in cache/memory
        # for each MPI process. Otherwise, you risk creating a lag between the MPI tasks
        # caching data depending which can slow down the parallelization
        if (withmpi) & ('logzsol' in model.free_params):
            dummy_obs = dict(filters=None, wavelength=None)

            logzsol_prior = model.config_dict["logzsol"]['prior']
            lo, hi = logzsol_prior.range
            logzsol_grid = np.around(np.arange(lo, hi, step=0.1), decimals=2)

            sps.update(**model.params)  # make sure we are caching the correct IMF / SFH / etc
            for logzsol in logzsol_grid:
                model.params["logzsol"] = np.array([logzsol])
                _ = model.predict(model.theta, obs=dummy_obs, sps=sps)

        # ensure that each processor runs its own version of FSPS
        # this ensures no cross-over memory usage
        from prospect.fitting import lnprobfn
        from functools import partial
        lnprobfn_fixed = partial(lnprobfn, sps=sps)

        if withmpi:
            run_params["using_mpi"] = True
            with MPIPool() as pool:

                # The dependent processes will run up to this point in the code
                if not pool.is_master():
                    pool.wait()
                    sys.exit(0)
                nprocs = pool.size
                # The parent process will oversee the fitting
                output = fit_model(obs, model, sps, noise, pool=pool, queue_size=nprocs, lnprobfn=lnprobfn_fixed, **run_params)
        else:
            # without MPI we don't pass the pool
            output = fit_model(obs, model, sps, noise, lnprobfn=lnprobfn_fixed, **run_params)

        # Set up an output file and write
        ts = time.strftime("%y%b%d-%H.%M", time.localtime())
        hfile = "{0}_{1}_mcmc.h5".format(args.outfile, ts)
        writer.write_hdf5(hfile, run_params, model, obs,
                          output["sampling"][0], output["optimization"][0],
                          tsample=output["sampling"][1],
                          toptimize=output["optimization"][1],
                          sps=sps)

        try:
            hfile.close()
        except(AttributeError):
            pass

Then, to run this file using mpi it can be called from the command line with something like

.. code-block:: shell

        mpirun -np <number of processors> python parameter_file.py --emcee
        # or
        mpirun -np <number of processors> python parameter_file.py --dynesty

Note that only model evaluation is parallelizable with `dynesty`, and many
operations (e.g. new point proposal) are still done in serial. This means that
single-core fits will always be more efficient in terms of total CPU usage per
fit. Having a large ratio of (live points / processors) helps efficiency, the
scaling goes as K ln(1 + M/K), where M = number of processes and K = number of
live points.

For `emcee` efficiency is maximized when K/(M-1) is an integer >= 2, where M =
number of processes and K = number of walkers.  The wall time speedup should be
approximately the same as this integer.