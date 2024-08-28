import time
import numpy as np

__all__ = ["run_nested_sampler", "parse_nested_kwargs"]


def parse_nested_kwargs(nested_sampler=None, **kwargs):

    # TODO:
    #   something like 'enlarge'
    #   something like 'bootstrap' or N_networks or?

    sampler_kwargs = {}
    run_kwargs = {}

    if nested_sampler == "dynesty":
        sampler_kwargs["bound"] = kwargs["nested_bound"]
        sampler_kwargs["sample"] = kwargs["nested_sample"]
        sampler_kwargs["walks"] = kwargs["nested_walks"]
        run_kwargs["dlogz_init"] = kwargs["nested_dlogz"]

    elif nested_sampler == "ultranest":
        #run_kwargs["dlogz"] = kwargs["nested_dlogz"]
        pass

    elif nested_sampler == "nautilus":
        pass

    else:
        # say what?
        raise ValueError(f"{nested_sampler} not a valid fitter")

    return sampler_kwargs, run_kwargs



def run_nested_sampler(model,
                       likelihood_function,
                       nested_sampler="dynesty",
                       nested_nlive=1000,
                       nested_neff=1000,
                       verbose=False,
                       nested_run_kwargs={},
                       nested_sampler_kwargs={}):
    """We give a model -- parameter discription and prior transform -- and a
    likelihood function. We get back samples, weights, and likelihood values.

    Returns
    -------
    samples : 3-tuple of ndarrays (loc, logwt, loglike)
        Loctions, log-weights, and log-likelihoods for the samples

    obj : Object
        The sampling object.  This will depend on the nested sampler being used.
    """

    go = time.time()

    # --- Nautilus ---
    if nested_sampler == "nautilus":
        from nautilus import Sampler

        sampler = Sampler(model.prior_transform,
                          likelihood_function,
                          n_dim=model.ndim,
                          pass_dict=False, # likelihood expects array, not dict
                          n_live=nested_nlive,
                          **nested_sampler_kwargs)
        sampler.run(n_eff=nested_neff,
                    verbose=verbose,
                    **nested_run_kwargs)
        obj = sampler

        points, log_w, log_like = sampler.posterior()

    # --- Ultranest ---
    if nested_sampler == "ultranest":

        from ultranest import ReactiveNestedSampler
        parameter_names = model.theta_labels()
        sampler = ReactiveNestedSampler(parameter_names,
                                        likelihood_function,
                                        model.prior_transform,
                                        **nested_sampler_kwargs)
        result = sampler.run(min_ess=nested_neff,
                             min_num_live_points=nested_nlive,
                             show_status=verbose,
                             **nested_run_kwargs)
        obj = result

        points = np.array(result['weighted_samples']['points'])
        log_w = np.log(np.array(result['weighted_samples']['weights']))
        log_like = np.array(result['weighted_samples']['logl'])

    # --- Dynesty ---
    if nested_sampler == "dynesty":
        from dynesty import DynamicNestedSampler

        sampler = DynamicNestedSampler(likelihood_function,
                                       model.prior_transform,
                                       model.ndim,
                                       nlive=nested_nlive,
                                       **nested_sampler_kwargs)
        sampler.run_nested(n_effective=nested_neff,
                           print_progress=verbose,
                           **nested_run_kwargs)
        obj = sampler

        points = sampler.results["samples"]
        log_w = sampler.results["logwt"]
        log_like = sampler.results["logl"]

    # --- Nestle ---
    if nested_sampler == "nestle":
        import nestle
        result = nestle.sample(likelihood_function,
                               model.prior_transform,
                               model.ndim,
                               **nested_sampler_kwargs)
        obj = result

        points = result["samples"]
        log_w = result["logwt"]
        log_like = result["logl"]

    dur = time.time() - go

    return dict(points=points, log_weight=log_w, log_like=log_like, duration=dur), obj


# OMG
NESTED_KWARGS = {
"dynesty_sampler_kwargs" : dict(nlive=None,
                                bound='multi',
                                sample='auto',
                                #periodic=None,
                                #reflective=None,
                                update_interval=None,
                                first_update=None,
                                npdim=None,
                                #rstate=None,
                                queue_size=None,
                                pool=None,
                                use_pool=None,
                                #logl_args=None,
                                #logl_kwargs=None,
                                #ptform_args=None,
                                #ptform_kwargs=None,
                                #gradient=None,
                                #grad_args=None,
                                #grad_kwargs=None,
                                #compute_jac=False,
                                enlarge=None,
                                bootstrap=None,
                                walks=None,
                                facc=0.5,
                                slices=None,
                                fmove=0.9,
                                max_move=100,
                                #update_func=None,
                                ncdim=None,
                                blob=False,
                                #save_history=False,
                                #history_filename=None)
                             ),
"dynesty_run_kwargs" : dict(nlive_init=None, # nlive0
                            maxiter_init=None,
                            maxcall_init=None,
                            dlogz_init=0.01,
                            logl_max_init=np.inf,
                            n_effective_init=np.inf, # deprecated
                            nlive_batch=None, #nlive0
                            wt_function=None,
                            wt_kwargs=None,
                            maxiter_batch=None,
                            maxcall_batch=None,
                            maxiter=None,
                            maxcall=None,
                            maxbatch=None,
                            n_effective=None,
                            stop_function=None,
                            stop_kwargs=None,
                            #use_stop=True,
                            #save_bounds=True, # doesn't hurt...?
                            print_progress=True,
                            #print_func=None,
                            live_points=None,
                            #resume=False,
                            #checkpoint_file=None,
                            #checkpoint_every=60)
                            ),
"ultranest_sampler_kwargs":dict(transform=None,
                                #derived_param_names=[],
                                #wrapped_params=None,
                                #resume='subfolder',
                                #run_num=None,
                                #log_dir=None,
                                #num_test_samples=2,
                                draw_multiple=True,
                                num_bootstraps=30,
                                #vectorized=False,
                                ndraw_min=128,
                                ndraw_max=65536,
                                storage_backend='hdf5',
                                warmstart_max_tau=-1,
                                ),
"ultranest_run_kwargs":dict(update_interval_volume_fraction=0.8,
                            update_interval_ncall=None,
                            #log_interval=None,
                            show_status=True,
                            #viz_callback='auto',
                            dlogz=0.5,
                            dKL=0.5,
                            frac_remain=0.01,
                            Lepsilon=0.001,
                            min_ess=400,
                            max_iters=None,
                            max_ncalls=None,
                            max_num_improvement_loops=-1,
                            min_num_live_points=400,
                            cluster_num_live_points=40,
                            insertion_test_window=10,
                            insertion_test_zscore_threshold=4,
                            region_class="MLFriends",
                            #widen_before_initial_plateau_num_warn=10000,
                            #widen_before_initial_plateau_num_max=50000,
                            ),
"nautilus_sampler_kwargs":dict(n_live=2000,
                               n_update=None,
                               enlarge_per_dim=1.1,
                               n_points_min=None,
                               split_threshold=100,
                               n_networks=4,
                               neural_network_kwargs={},
                               #prior_args=[],
                               #prior_kwargs={},
                               #likelihood_args=[],
                               #likelihood_kwargs={},
                               n_batch=None,
                               n_like_new_bound=None,
                               #vectorized=False,
                               #pass_dict=None,
                               pool=None,
                               seed=None,
                               blobs_dtype=None,
                               #filepath=None,
                               #resume=True
                               ),
"nautilus_run_kwargs":dict(f_live=0.01,
                           n_shell=1,
                           n_eff=10000,
                           n_like_max=np.inf,
                           discard_exploration=False,
                           timeout=np.inf,
                           verbose=False
                           ),
}
