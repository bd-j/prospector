import inspect
import numpy as np
import time
import warnings

__all__ = ["run_nested_sampler"]


def run_nested_sampler(model,
                       likelihood_function,
                       nested_sampler="dynesty",
                       nested_nlive=1000,
                       nested_neff=1000,
                       verbose=False,
                       **kwargs):
    """We give a model -- parameter discription and prior transform -- and a
    likelihood function. We get back samples, weights, and likelihood values.

    Parameters
    ----------
    model : instance of the :py:class:`prospect.models.SpecModel`
        The model parameterization and parameter state.
    likelihood_function : callable
        Likelihood function
    nested_live : int
        Number of live points.
    nested_neff : float
        Minimum effective sample size.
    verbose : bool
        Whether to output sampler progress.

    Returns
    -------
    samples : 3-tuple of ndarrays (loc, logwt, loglike)
        Loctions, log-weights, and log-likelihoods for the samples

    obj : Object
        The sampling results object. This will depend on the nested sampler being used.
    """
    if verbose:
        print(f"running {nested_sampler} for {nested_neff} effective samples")

    go = time.time()

    # Initialize the sampler.
    if nested_sampler == 'nautilus':
        from nautilus import Sampler
        sampler_init = Sampler
        init_args = (model.prior_transform, likelihood_function)
        init_kwargs = dict(pass_dict=False, n_live=nested_nlive,
                           n_dim=model.ndim)
    elif nested_sampler == 'ultranest':
        from ultranest import ReactiveNestedSampler
        sampler_init = ReactiveNestedSampler
        init_args = (model.theta_labels(), likelihood_function,
                     model.prior_transform)
        init_kwargs = dict()
    elif nested_sampler == 'dynesty':
        from dynesty import DynamicNestedSampler
        sampler_init = DynamicNestedSampler
        init_args = (likelihood_function, model.prior_transform, model.ndim)
        init_kwargs = dict(nlive=nested_nlive)
    elif nested_sampler == 'nestle':
        import nestle
        init_kwargs = dict()
    else:
        raise ValueError(f"No nested sampler called '{nested_sampler}'.")

    if nested_sampler != 'nestle':
        sig = inspect.signature(sampler_init).bind_partial()
        sig.apply_defaults()
        for key in kwargs.keys() & init_kwargs.keys():
            warnings.warn(f"Value of key '{key}' overwritten.")
        init_kwargs = {
            **{key: kwargs[key] for key in sig.kwargs.keys() & kwargs.keys()},
            **init_kwargs}
        sampler = sampler_init(*init_args, **init_kwargs)

    # Run the sampler.
    if nested_sampler == 'nautilus':
        sampler_run = sampler.run
        run_args = ()
        run_kwargs = dict(n_eff=nested_neff, verbose=verbose)
    elif nested_sampler == 'ultranest':
        sampler_run = sampler.run
        run_args = ()
        run_kwargs = dict(
            min_ess=nested_neff, min_num_live_points=nested_nlive,
            show_status=verbose)
    elif nested_sampler == 'dynesty':
        sampler_run = sampler.run_nested
        run_args = ()
        run_kwargs = dict(n_effective=nested_neff, print_progress=verbose)
    elif nested_sampler == 'nestle':
        sampler_run = nestle.sample
        run_args = (likelihood_function, model.prior_transform, model.ndim)
        run_kwargs = dict()

    sig = inspect.signature(sampler_run).bind_partial()
    sig.apply_defaults()
    for key in kwargs.keys() & run_kwargs.keys():
        warnings.warn(f"Value of key '{key}' overwritten.")
    run_kwargs = {
        **{key: kwargs[key] for key in sig.kwargs.keys() & kwargs.keys()},
        **run_kwargs}
    run_return = sampler_run(*run_args, **run_kwargs)
    #for key in kwargs.keys() - (init_kwargs.keys() | run_kwargs.keys()):
    #    warnings.warn(f"Key '{key}' not recognized by the sampler.")

    if nested_sampler == 'nautilus':
        obj = sampler
        points, log_w, log_like = sampler.posterior()
    elif nested_sampler == 'ultranest':
        obj = run_return
        points = np.array(run_return['weighted_samples']['points'])
        log_w = np.log(np.array(run_return['weighted_samples']['weights']))
        log_like = np.array(run_return['weighted_samples']['logl'])
    elif nested_sampler == 'dynesty':
        obj = sampler
        points = sampler.results["samples"]
        log_w = sampler.results["logwt"]
        log_like = sampler.results["logl"]
    elif nested_sampler == 'nestle':
        obj = run_return
        points = run_return["samples"]
        log_w = run_return["logwt"]
        log_like = run_return["logl"]

    dur = time.time() - go

    return dict(points=points, log_weight=log_w, log_like=log_like,
                duration=dur), obj
