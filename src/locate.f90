FUNCTION LOCATE(xx,x)
  !Find the index of an array corresponding to a given value
  !
  ! Inputs
  ! ------
  !
  ! xx [float (nval)]:
  !   A vector of values of length nval
  !
  ! x [float]:
  !   The value for which you want to know the index
  !
  ! Outputs
  ! -------
  !
  ! locate [integer]:
  !   The index of xx corresponding to x

  USE sps_vars
  IMPLICIT NONE
  REAL(SP), DIMENSION(:), INTENT(IN) :: xx
  REAL(SP), INTENT(IN) :: x
  INTEGER :: locate
  INTEGER :: n,jl,jm,ju
  LOGICAL :: ascnd
  
  n = SIZE(xx)
  ascnd = (xx(n) >= xx(1))
  jl=0
  ju=n+1
  
  IF (x == xx(1)) THEN
     locate=1
  ELSE IF (x == xx(n)) THEN
     locate=n-1
  ELSE
     DO
        IF (ju-jl <= 1) EXIT
        jm=(ju+jl)/2
        IF (ascnd.EQV.(x >= xx(jm))) THEN
           jl=jm
        ELSE
           ju=jm
        ENDIF
     ENDDO
     locate=jl
  ENDIF

END FUNCTION LOCATE
