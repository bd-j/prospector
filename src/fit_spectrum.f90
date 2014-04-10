PROGRAM BSFH

  use sps_vars; use fit_vars

  REAL(8)    :: A,B
  integer :: num_args, ix
  character(len=12), dimension(:), allocatable :: args

  TYPE(OBSDAT) :: obs
  TYPE(MCPARAMS) :: mcpars
  REAL(SP), DIMENSION(:) :: position

  num_args = command_argument_count()
  allocate(args(num_args))  ! I've omitted checking the return status of the allocation 
  do ix = 1, num_args
     call get_command_argument(ix,args(ix))
     ! now parse the argument as you wish
  end do

  ! Load the observations into an obsdat structure
  filename = args(1)
  CALL LOAD_OBS(filename, obs)

  CALL SPS_SETUP(-1)

  !draw a a starting position from the prior distribution
  ! or simply hard-code it
  CALL INITIAL_POSITION(position, obs)

  !Use a minimization routine to find a good initial starting point

  minpars%mintype = 0 !0 for powell, 1 for BFGS, -1 for no optimization
  minpars%neval = 1000 !maximum number of likelhood evaluations 

  CALL MAXIMIZE_LNPROB(minpars, position, obs)

  !Run MCMC from this initial position.  Use either emcee or 
  ! hmc (if derivatives available)
  mcpars%mctype = 0 !0 for fmc, 1 for hmc
  mcpars%nburn = 100
  mcpars%nreburn =10
  mcpars%nwalkers = 128
  mcpars%niter = 100
  mcpars%fileout = 'chain.dat'
  mcpars%ballsize = 0.05

  CALL MCMC(mcpars, obs, position)

END PROGRAM BSFH
