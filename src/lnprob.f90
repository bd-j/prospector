SUBROUTINE LNPROB(position, obs, lnprob, extras)

  USE sps_vars; USE sps_utils, ONLY: prior_probability, model

  REAL(SP), INTENT(in), DIMENSION(:) :: position
  TYPE(OBSDAT), INTENT(in) :: obs

  REAL(SP), INTENT(out) :: lnprob
  TYPE(MEXTRAS), INTENT(out) :: extras

  REAL(SP) :: prior_prob, lnp_spec, lnp_phot, lnp_jitter
  REAL(SP), DIMENSION(obs%nspec) :: outspec
  REAL(SP), DIMENSION(obs%nband) :: mags

  CALL PRIOR_PROBABILITY(position, prior_prob)

  IF (prior_prob.gt.tiny_number) THEN
     CALL MODEL(position, outspec, mags, extras)

     lnp_spec = -0.50d * SUM( (outspec - obs%spec)**2 / obs%specerr**2 )
     lnp_phot = -0.50d * SUM( (10**(-0.4 * mags) - 10**(-0.4 * obs%mags))**2 / &
          (1.086d * obs%magerr * 10**(-0.4 * obs%mags))**2 )

     lnprob = LOG(prior_prob) + lnp_spec + lnp_phot

  ELSE
     lnprob = LOG(tiny_number) !HACK


END SUBROUTINE LNPROB
