#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""prospect_args.py - methods to get a default argument parser for prospector.
"""

import argparse

__all__ = ["get_parser", "show_default_args"]


def show_default_args():
    parser = get_parser()
    parser.print_help()


def get_parser(fitters=["optimize", "emcee", "dynesty"]):
    """Get a default prospector argument parser
    """

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # --- Basic ---
    parser.add_argument("--verbose", type=int, default=1,
                        help="Whether to print lots of stuff")

    parser.add_argument("--debug", dest="debug", action="store_true",
                        help=("If set, halt execution just before optimization and sampling, "
                              "but after the obs, model, and sps objects have been built."))
    parser.set_defaults(debug=False)

    parser.add_argument("--outfile", type=str, default="prospector_test_run",
                        help="Root name (including path) of the output file(s).")

    parser.add_argument("--output_pickle", action="store_true",
                        help="If set, output pickles in addition to HDF5.")

    # --- SPS parameters ---
    parser.add_argument("--zcontinuous", type=int, default=1,
                        help=("The type of metallicity parameterization to use.  "
                              "See python-FSPS documentation for details."))

    if "optimize" in fitters:
        parser = add_optimize_args(parser)

    if "emcee" in fitters:
        parser = add_emcee_args(parser)

    if "dynesty" in fitters:
        parser = add_dynesty_args(parser)

    return parser


def add_optimize_args(parser):
    # --- Optimization ---
    parser.add_argument("--optimize", action="store_true",
                        help="If set, do an optimization before sampling.")

    parser.add_argument("--min_method", type=str, default="lm",
                        help=("The scipy.optimize method to use for minimization."
                              "One of 'lm' (Levenberg-Marquardt) or 'powell' (powell line-search.)"))

    parser.add_argument("--min_opts", type=dict, default={},
                        help="Minimization parameters.  See scipy.optimize.")

    parser.add_argument("--nmin", type=int, default=1,
                        help=("Number of draws from the prior from which to start minimization."
                              "nmin > 1 can be useful to avoid local minima"))

    return parser


def add_emcee_args(parser):
    # --- emcee fitting ----
    parser.add_argument("--emcee", action="store_true",
                        help="If set, do ensemble MCMC sampling with emcee.")

    parser.add_argument("--nwalkers", type=int, default=64,
                        help="Number of `emcee` walkers.")

    parser.add_argument("--niter", type=int, default=512,
                        help="Number of iterations in the `emcee` production run")

    parser.add_argument("--nburn", type=int, nargs="*", default=[16, 32, 64],
                        help=("Specify the rounds of burn-in for `emcee` by giving the "
                              "number of iterations in each round as a list.  "
                              "After each round the walkers are reinitialized based "
                              "on the locations of the best half of the walkers."))

    parser.add_argument("--save_interval", dest="interval", type=float, default=0.2,
                        help=("Number between 0 and 1 giving the fraction of the `emcee` "
                              "production run at which to write the current chains to "
                              "disk.  Useful in case an expensive `emcee` run dies."))

    parser.add_argument("--restart_from", type=str, default="",
                        help=("If given, the name of a file that contains a previous "
                              "`emcee` run from which to try and restart emcee sampling."
                              "In this case niter gives the number of additional iterations"
                              " to run for; all other options are ignored "
                              "(they are taken from the previous run.)"))

    parser.add_argument("--ensemble_dispersion", dest="initial_disp", type=float, default=0.1,
                        help=("Initial dispersion in parameter value for the `emcee` walkers."
                              " This can be overriden for individual parameters by adding an 'init_disp' "
                              "key to the parameter specification dictionary for that parameter."))

    return parser


def add_dynesty_args(parser):
    # --- dynesty parameters ---
    parser.add_argument("--dynesty", action="store_true",
                        help="If set, do nested sampling with dynesty.")

    parser.add_argument("--nested_bound", type=str, default="multi",
                        choices=["single", "multi", "balls", "cubes"],
                        help=("Method for bounding the prior volume when drawing new points. "
                              "One of single | multi | balls | cubes"))

    parser.add_argument("--nested_sample", "--nested_method", type=str, dest="nested_sample",
                        default="slice", choices=["unif", "rwalk", "slice"],
                        help=("Method for drawing new points during sampling.  "
                              "One of unif | rwalk | slice"))

    parser.add_argument("--nested_walks", type=int, default=48,
                        help=("Number of Metropolis steps to take when "
                              "`nested_sample` is 'rwalk'"))

    parser.add_argument("--nlive_init", dest="nested_nlive_init", type=int, default=100,
                        help="Number of live points for the intial nested sampling run.")

    parser.add_argument("--nlive_batch", dest="nested_nlive_batch", type=int, default=100,
                        help="Number of live points for the dynamic nested sampling batches")

    parser.add_argument("--nested_dlogz_init", type=float, default=0.05,
                        help=("Stop the initial run when the remaining evidence is estimated "
                              "to be less than this."))

    parser.add_argument("--nested_maxcall", type=int, default=int(5e7),
                        help=("Maximum number of likelihood calls during nested sampling. "
                              "This will only be enforced after the initial pass"))

    parser.add_argument("--nested_maxiter", type=int, default=int(1e6),
                        help=("Maximum number of iterations during nested sampling. "
                              "This will only be enforced after the initial pass"))

    parser.add_argument("--nested_maxbatch", type=int, default=10,
                        help="Maximum number of dynamic batches.")

    parser.add_argument("--nested_bootstrap", type=int, default=0,
                        help=("Number of bootstrap resamplings to use when estimating "
                              "ellipsoid expansion factor."))

    parser.add_argument("--nested_posterior_thresh", type=float, default=0.05,
                        help=("Stop when the fractional scatter in the K-L divergence of the "
                              "posterior estimates reaches this value"))

    return parser


def add_data_args(parser):
    # --- data manipulation
    # logify_spectrum
    # normalize_spectrum

    return parser
