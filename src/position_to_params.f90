SUBROUTINE POSITION_TO_PARAMS(pos, pset, ssp_dirt)

  ! Converts elements of the state vector 'pos' into fsps parameters. If any
  ! any of the state voctors change the ssp parameters, then set the ssp_dirt flag.
  !
  ! Note: THIS ROUTINE, ALONG WITH 'CALIBRATION_MODEL' AND 'PRIORS' LARGELY 
  !       DEFINES YOUR MODEL.  CHANGE IT!!!

  ! Inputs
  ! ------
  !
  ! pos [float (npos)]:
  !   Position vector
  !
  ! Outputs
  ! ------
  !
  ! pset [params]:
  !   The parameter set, updated with new values from the position vector
  !
  ! ssp_dirt [integer]:
  !   A flag that indicates whether SSP based parameters were changed. Set
  !   to zero id they were not changed, 1 otherwise

  USE sps_vars
  IMPLICIT NONE
  
  REAL(SP), DIMENSION(:), INTENT(in) :: pos
  TYPE(PARAMS), INTENT(inout) :: pset
  !TYPE(CSP_PARAMS), INTENT(inout) :: csp_pset
  !TYPE(CAL_PARAMS), INTENT(inout) :: cal_pset

  INTEGER, INTENT(out) :: ssp_dirt=0 

  pset%tage = pos(1)
  pset%amplitudes = pos(1:10)
  
  IF (ABS(pset%imf3-pos(3)).GT.tiny_number) THEN
     pset%imf3 = pos(3)
     ssp_dirt = 1
  ENDIF

  pset%sigma_smooth = pos(4)
  pset%dust2 = pos(5)
  pset%dust1 = 0.0 !pars%dust2 * 2.0d0
  pset%mass = pos(6)
  pset%mwr = 3.1

END SUBROUTINE
