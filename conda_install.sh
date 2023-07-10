#!/usr/bin/bash

# change this if you want to install elsewhere;
# or, copy and run this script in the desired location
CODEDIR=$PWD
cd $CODEDIR

# Install FSPS from source (to get data files)
git clone https://github.com/cconroy20/fsps.git
export SPS_HOME="$PWD/fsps"

# Create and activate environment (named 'prospector')
git clone https://github.com/bd-j/prospector.git
cd prospector
conda env create -f environment.yml -n prospector
conda activate prospector
python -m pip install .

echo "Add 'export SPS_HOME=$SPS_HOME' to your .bashrc"