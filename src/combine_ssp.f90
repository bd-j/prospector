SUBROUTINE COMBINE_SSP(pset, outspec, extras)
  ! Linearly combine SSPs, including attenuation.  This assumes that
  !  the spec_ssp_zz array has been previously filled
  !  
  ! Inputs
  ! ------
  !
  ! pset [params]:
  !   The parameter set to use for the composite SPS
  !
  ! Outputs
  ! -------
  !
  ! outspec [float (nspec)]:
  !   The output spectrum including 
  !
  ! extras [outextras]:
  !   Extra outputs expressed as a structure

  USE sps_vars; USE sps_utils, ONLY : attenuate
  IMPLICIT NONE

  TYPE(PARAMS), INTENT(in) :: pset
  !REAL(SP), INTENT(in), DIMENSION(nz,ntfull,nspec) :: ssp_spec
  !REAL(SP), INTENT(in), DIMENSION(nz,ntfull) :: ssp_mass

  REAL(SP), INTENT(inout), DIMENSION(nspec) :: outspec = 0
  TYPE(MEXTRAS), INTENT(inout) :: extras

  REAL(SP), DIMENSION(nspec) :: spec_att

  !if theres more than one age at this metallicity, sum them after attenuation
  DO i = 1, pset%ncomp
     j = pset%age_index(i)
     k = pset%zindex(i)
     spec_att = ATTENUATE(pset, spec_ssp_zz(k,j,:), time_full(j))
     outspec = outspec + pset%ssp_amplitude(i) * spec_att
  END DO

END SUBROUTINE COMBINE_SSP
