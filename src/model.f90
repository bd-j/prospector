SUBROUTINE MODEL(position, obs, outspec, phot, extras)
  ! Given a position vector and an observation, determine the full spectral
  !   model including calibration and also the photometry
  ! 
  ! Inputs
  ! ------
  !
  ! position [float (npar)]:
  !   The postion vector for the parameters to be fit
  !
  ! obs [obsdat]:
  !   An OBSDAT structure, used to define the output wavelength array
  !
  ! Outputs
  ! -------
  !
  ! outspec [float (obs%nspec)]:
  !   The output spectrum, including spectral calibration factors, and 
  !   interpolated to the observational wavelength scale
  !
  ! phot [float (obs%nband)]:
  !   The model magnitudes through each of the bands
  ! 
  ! extras [float (nextra)]:
  !   extra parameters of the model to claculate and return

  USE sps_vars; USE sps_utils, ONLY : position_to_params, ssp_gen,&
       single_ssp, combine_ssp, smoothspec, calibration_model, getmags
  IMPLICIT NONE

  REAL(SP), INTENT(in), DIMENSION(:) :: position
  TYPE(OBSDAT), INTENT(in)  :: obs

  REAL(SP), INTENT(inout), DIMENSION(obs%nspec) :: outspec 
  REAL(SP), INTENT(inout), DIMENSION(obs%nbands) :: phot 
  REAL(SP), INTENT(inout), DIMENSION(1) :: extras=0.0 

  REAL(SP), DIMENSION(nspec) :: spec
  REAL(SP) :: ssp_dirt = 0
  
  TYPE(PARAMS) :: pset
  
  TYPE(MEXTRAS) :: extras

  !parse the state vector into parameters for the SSPs, calibration, etc.
  ! if ssp parameters have been changed, ssp_dirt will be 1
  CALL POSITION_TO_PARAMS(position, pset, ssp_dirt)

  !rebuild the SSPs  if necessary
  IF (ssp_dirt.GT.0) THEN
     DO i=1,nz
        pset%zmet = i
        CALL SSP_GEN(pset,mass_ssp_zz(i,:),&
             lbol_ssp_zz(i,:),spec_ssp_zz(i,:,:))
     ENDDO
  ENDIF

  !combine (or interpolate) the SSPs with weighting and dust attenuation
  IF (pset%sfh.EQ.0) THEN
     CALL SINGLE_SSP(pset, spec, extras)
  ELSE
     CALL COMBINE_SSP(pset, spec, extras)
  ENDIF

  !smooth the spectrum.  This should probably be moved into the 
  ! SSP_GEN loop since it won't have easy derivatives.  Then the spec_ssp_zz 
  ! can be used to easily find partial derivatives
  CALL SMOOTHSPEC(spec_lambda, spec, pset%sigma_smooth, pset%min_wave_smooth, pset%max_wave_smooth)

  !apply any spectral calibration factors and bring onto the observed wavelength scale
  ! Again, this should probably go into the SSP_GEN loop, though aspects of it might have
  ! easy derivatives
  CALL CALIBRATION_MODEL(pset, obs, spec, outspec)

  !find the photometry for this model, ignoring spectral calibration factors
  CALL GETMAGS(pset%zred, spec, phot, mag_compute)

  !that's it

END SUBROUTINE
