Installation on Odyssey
======

Setup environments
==

* git: follow instructions [here](https://rc.fas.harvard.edu/resources/documentation/software/git-and-github-on-odyssey/) and [here](https://help.github.com/articles/generating-an-ssh-key/) to get ssh based git working

* Download dependencies. In your code directory:

  ```bash
  git clone git@github.com:joshspeagle/dynesty.git;
   git clone git@github.com:dfm/emcee.git;
  git clone git@github.com:bd-j/sedpy.git;
  git clone git@github.com:bd-j/prospector.git;
  git clone git@github.com:cconroy20/fsps.git;
  git clone git@github.com:dfm/python-fsps.git;
  ```

* Load modules we'll use and create a python environment
  ```bash
  module purge
  module load anaconda/5.0.1-fasrc01
  conda create -n pro --clone="$PYTHON_HOME"
  source activate pro
  module load gcc/4.8.2-fasrc01
  module load openmpi/1.8.8-fasrc01
  module load hdf5/1.8.16-fasrc01
  ```

* Compile FSPS-- there can be some advantage to doing this on the cores you
  plan to use. Here we will use ```conroy-intel```, but you could also use ```test```.
   - Add ``SPS_HOME`` environment variable to your .bashrc, this is just the
     path to the ```fsps/``` directory.
   - Change ``$SPS_HOME/src/Makefile`` F90 flags to include ``-fPIC``
   - get an interactive job on the partition/cores you plan to use and compile
   ```bash
   srun --pty --mem 2000 -p conroy-intel -t 0-1:00 /bin/bash
   cd $SPS_HOME/src
   make clean; make all
   source activate pro
   cd ../../python-fsps; python setup.py install
   ```

* Install MPI and HDF5, which are not in the 2.7.11 python anaconda env (for
  good reason)  Note that the ``mpi4py`` install can be skipped if it is acausing
  problems as it is only used for certain kinds of ``emcee`` sampling

  ```bash
  source activate pro
  pip install --no-binary=h5py h5py
  pip install mpi4py==1.3.1
  ```

* Install dependencies in Anaconda environment:
  ```bash
  source activate pro
  cd sedpy; python setup.py install
  cd ../emcee; python setup.py install
  cd ../dynesty; python setup.py install
  cd ../prospector; python setup.py install
  cd ../python-fsps; python setup.py install # only if not done on cores intended for use (see above)
  ```

You will want to add some of these to your .bashrc
```bash
  module load python/2.7.11-fasrc01
  module load gcc/4.8.2-fasrc01
  module load openmpi/1.8.8-fasrc01
  module load hdf5/1.8.16-fasrc01
  source activate pro #optional
```
