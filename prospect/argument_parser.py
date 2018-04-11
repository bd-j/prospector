import argparse


#parser.add_argument("--", type=, default=,
#                    help="")


def get_parser(fitter="dynesty"):
    """Get a default prospector argument parser
    """
    
    parser = argparse.ArgumentParser()

    # --- Basic ---
    parser.add_argument("--verbose", type=bool, default=True,
                        help="Whether to print lots of stuff")

    parser.add_argument("--debug", type=bool, default=False,
                        help="If True, halt execution just before optimization and sampling.")

    parser.add_argument("--outfile", type=str, default="prospector_test",
                        help="Root name (including path) of the output file(s).")

    parser.add_argument("--output_pickle", type=bool, default=False,
                        help="Swith to turn on/off output of pickles in addition to HDF5")

    parser.add_argument("--fit_type", type=str, default="dynesty",
                        help=("Specify the type of fit to do.  One of |dynesty|emcee|optimization"))


    # --- Optimization ---
    parser.add_argument("--optimize", type=bool, default=False,
                        help="Do an optimization before sampling.")

    parser.add_argument("--min_method", type=str, default="lm",
                        help=("The scipy.optimize method to use for minimization"))

    parser.add_argument("--min_opts", type=dict, default={},
                        help="minimization parameters")

    parser.add_argument("--nmin", type=int, default=5,
                        help="Number of draws from the prior from which to start minimization")


    # --- emcee fitting ----
    parser.add_argument("--emcee", type=bool, default=False,
                        help="Do ensemble MCMC sampling with emcee.")
    
    parser.add_argument("--nwalkers", type=int, default=64,
                        help="Number of `emcee` walkers.")

    parser.add_argument("--niter", type=int, default=512,
                        help="Number of iterations in the production run")

    parser.add_argument("--nburn", type=list, default=[16, 32, 64],
                        help=("Specify the rounds of burn-in by giving the number of "
                              "iterations in each round as a list.  After each round "
                              "the walkers are reinitialized based on the locations of "
                              "the best half of the walkers."))
 
    parser.add_argument("--interval", type=float, default=0.2,
                        help=("Number between 0 and 1 giving the fraction of the "
                              "production run at which to write the curtrent chains to "
                              "disk.  Useful in case the run dies."))

    # --- dynesty parameters ---
    parser.add_argument("--dynesty", type=bool, default=True,
                        help="Do nested sampling with dynesty.")

    parser.add_argument("--nested_bound", type=str, default="multi",
                        help="")

    parser.add_argument("--nested_method", type=str, default="unif",
                        help="")

    # nested_nlive_init
    # neste_nlive_batch
    # nested_dlogz_init
    # nested_weight_kwargs
    # nested_stop_kwargs
    # nested_bootstrap

    # --- data manipulation
    # logify_spectrum
    # normalize_spectrum
    
    # --- SPS parameters ---
    parser.add_argument("--zcontinuous", type=int, default=1,
                        help=("The type of metallicity parameterization to use.  "
                              "See python-FSPS documentation for details."))

    return parser

                              
