#!/usr/bin/bash

# change this if you want to install elsewhere;
# or, copy and run this script in the desired location
CODEDIR=$PWD
cd $CODEDIR

# Install FSPS from source (to get data files)
git clone git@github.com:cconroy20/fsps
export SPS_HOME="$PWD/fsps"

# Create and activate environment (named 'prospector')
git clone git@github.com:bd-j/prospector.git
cd prospector
conda env create -f environment.yml
conda activate prospector
cd ..

# Install other repos from source
cd $CODEDIR
repos=( bd-j/sedpy )
for r in "${repos[@]}"; do
    git clone git@github.com:$r
    cd ${r##*/}
    python setup.py install
    cd ..
done

cd prospector
python setup.py install

echo "Add 'export SPS_HOME=$SPS_HOME' to your .bashrc"