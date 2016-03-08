superMIC Install
--------

docs http://www.hpc.lsu.edu/docs/guides.php?system=SuperMIC

1. In your .modules file add
   ```
   module load intel
   module load mvapich2
   module load python/2.7.10-mkl-mic
   module load hdf5
   ```
   log out and log back in to make sure the modules are loaded.
   Note also that the `/work/<username>/` directory referenced below takes an hour after first login to be created.

2. Get all the required sources.
   ```
   cd /work/<username>; mkdir apps; cd apps
   git clone https://github.com/h5py/h5py
   git clone https://github.com/cconroy20/fsps
   git clone https://github.com/dfm/python-fsps
   git clone https://github.com/dfm/emcee
   git clone https://github.com/bd-j/sedpy
   git clone https://github.com/bd-j/bsfh
   ```
   The python we are using is *very* bare bones and we don't have system install privileges,
   so we need some extra stuff installed by hand:
   ```
   wget --no-check-certificate https://pypi.python.org/packages/source/s/setuptools/setuptools-20.2.2.tar.gz#md5=bf37191cb4c1472fb61e6f933d2006b1
   tar -xvf setuptools-20.2.2.tar.gz
   wget --no-check-certificate https://pypi.python.org/packages/source/a/astropy/astropy-1.1.1.tar.gz#md5=03a6a9189fc0ce58862540c974bc17e8
   tar -xvf astropy-1.1.1.tar.gz
```

3. Add paths.  In ~/.bashrc add the lines
   ```
   export WORK=/work/<username>
   export SPS_HOME=$WORK/apps/fsps/
   export CODE=$WORK/apps
   export PYTHONPATH=$PYTHONPATH:$CODE/emcee
   export PYTHONPATH=$PYTHONPATH:$CODE/python-fsps
   export PYTHONPATH=$PYTHONPATH:$CODE/sedpy
   export PYTHONPATH=$PYTHONPATH:$CODE/bsfh
   export PYTHONPATH=$PYTHONPATH:$CODE/setuptools-20.2.2
   export PYTHONPATH=$PYTHONPATH:$CODE/astropy-1.1.1
   export PYTHONPATH=$PYTHONPATH:$CODE/h5py
   ```
   and `source ~/.bashrc`

4. Build some of the basic tools we need
   ```
   cd setuptools-20.2.2; python setup.py build
   cd ../sedpy; python setup.py build
   cd ../astropy-1.1.1; python setup.py build
   ```
   The last one will take awhile.
   I hate astropy, we really only need the cosmology calculation....

5. Build FSPS and python-FSPS.
   ```
   cd $SPS_HOME/src
   ```
   edit Makefile to have following uncommented
   ```
   F90 = ifort
   F90FLAGS = -O3 -funroll-loops -cpp -fPIC
   ```
   Then edit `sps_vars.f90` to change to MILES and whatever `time_res_inc` you want.
   Also make sure there is not a trailing empty line in `$SPS_HOME/data/FILTER_LIST`
   (I need to write some python-FSPS code to deal with this situation.)

	Then `make clean; make all`
	Then change to the python-FSPS directory:
	```
	cd ../../python-fsps/
	```
	and make the following change in setup.py at line 59, in order to compile with intel:
	```python
	- flags = '-c -I{0} --f90flags=-cpp --f90flags=-fPIC'.format(fsps_dir).split()
	+ flags = '-c -I{0} --f90flags=-cpp --f90flags=-fPIC --fcompiler=intelem'.format(fsps_dir).split()
	```
	and finally build:
	`python setup.py build`


6. Assuming you've defined `$PROJECT_DIR` ro be the place where your project parameter files, data, etc is,
	your jobscript should look something like:

		
        #!/bin/bash
        
        ###queue
        #PBS -q workq
        
        ### Requested number of nodes
        #PBS -l nodes=4:ppn=20
        
        ### Requested computing time
        #PBS -l walltime=10:00:00
        
        ### Account
        #PBS -A TG-AST150015
        
        ### Job name
        #PBS -N 'specphot'
        
        ### output and error logs
        #PBS -o specphot_$PBS_JOBID.out
        #PBS -e specphot_$PBS_JOBID.err
        
        cd $PBS_O_WORKDIR
        
        mpirun -np 80 -machinefile $PBS_NODEFILE \
        python $PROJECT_DIR/prospector.py \
		--param_file=$PROJECT_DIR/paramfile.py \
		--objname=NGC1851 \
		--outfile=$PROJECT_DIR/results/specphot_$PBS_JOBID \
		--nwalkers=158 --niter=1024 --do_powell=False --noisefactor=5.0
