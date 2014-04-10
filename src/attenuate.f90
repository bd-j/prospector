FUNCTION ATTENUATE(pset, spectrum, age)
  ! Wrapper on ADD_DUST to take care of the age dependence of the 
  !  dust attenuation this assumes that you are inputting in an SSP 
  !  spectrum but you still want the two-component dust model with 
  !  parameters in 'pset'.
  !
  ! Inputs
  ! ------
  !
  ! pset [params]:
  !   The parameter set, containing dust_tesc, dust1, dust2, and other
  !   parameters of the dust attenuation.  It will be passed to ADD_DUST
  !
  ! spectrum [float (nspec)]:
  !   The intrinsic spectrum to attenuate
  ! 
  ! age [float]:
  !   The age (in Gyr) of the population that has `spectrum`
  ! Outputs
  ! -------
  !
  ! attenuate [double precision (ndim)]:
  !   The attenuated spectrum.

  IMPLICIT NONE
  USE sps_vars

  TYPE(PARAMS), INTENT(in) :: pset
  REAL(SP), INTENT(in), DIMENSION(nspec) :: spectrum
  REAL(SP), INTENT(in) :: age

  REAL(SP), INTENT(out), DIMENSION(nspec) :: attenuate

  REAL(SP), DIMENSION(nspec) :: spec1, spec2
  REAL(SP), :: mdust

  spec1 = 0.0
  spec2 = 0.0
  IF (age.LT.pset%dust_tesc) THEN 
     spec1 = spectrum
  ELSE
     spec2 = spectrum
  END IF
  CALL ADD_DUST(pset, spec1, spec2, attenuate, mdust)

END FUNCTION attenuate
