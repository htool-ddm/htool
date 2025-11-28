#!/bin/bash

# Initialization
MY_PATH="`dirname \"$0\"`"             
MY_PATH="`( cd \"$MY_PATH\" && pwd )`" 
cd ${MY_PATH}
cd ../
mkdir -p build & cd build
cmake ../
make Use_hmatrix
mkdir -p ../output/examples/use_hmatrix

# Arguments
outputfolder=../output/examples/use_hmatrix/

# Run
./examples/Use_hmatrix ${outputfolder}

# Display output
python3 ../tools/plot_hmatrix.py --inputfile ../output/examples/use_hmatrix/hmatrix.csv
