SUBROUTINE SINGLE_SSP(pset, outspec, extras)
  ! Compute the spectum of a single-age, single-metallicity spectrum.
  !  This does all the interpolation in age and metallicity as well 
  !  as dust attenuation and spectrum broadening.
  !
  ! Inputs
  ! ------
  !
  ! pset [params]:
  !   The parameter set
  !
  ! ssp_spec [float (nz, ntfull, nspec):
  !   The full arrays of SSP spectra for all metalllicities
  !
  ! ssp_mass [float (nz, ntfull):
  !   The stellar mass of each SSP
  !
  !
  ! Outputs
  ! -------
  !
  ! outspec [float (nspec)]:
  !   The output spectrum including 
  !
  ! extras [outextras]:
  !   Extra outputs expressed as a structure

  USE sps_vars; USE sps_utils, ONLY : zinterp, tinterp, attenuate
  IMPLICIT NONE

  TYPE(PARAMS), INTENT(in) :: pset
  REAL(SP), INTENT(in), DIMENSION(ntfull,nspec) :: tspec
  REAL(SP), INTENT(in), DIMENSION(nz,ntfull) :: tmass

  REAL(SP), INTENT(inout), DIMENSION(nspec) :: outspec = 0
  TYPE(EXTRAS), INTENT(inout) :: extras

  !Interpolate in metallicity
  CALL ZINTERP(pset%logzsol, tspec, tmass)

  !interpolate in time
  CALL TINTERP(pset%tage, tspec, tmass, outspec, outmass)

  !attenuate by dust
  outspec = pset%ssp_amplitude(1) * ATTENUATE(pset, outspec, pset%tage)

END SUBROUTINE
