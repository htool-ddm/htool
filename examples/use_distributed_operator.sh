#!/bin/bash

# Initialization
MY_PATH="`dirname \"$0\"`"             
MY_PATH="`( cd \"$MY_PATH\" && pwd )`" 
cd ${MY_PATH}
cd ../
mkdir -p build & cd build
cmake ../
make Use_distributed_operator
mkdir -p ../output/examples/use_distributed_operator

# Arguments
outputpath=../output/examples/use_distributed_operator/

# Run
mpirun -np 4 ./examples/Use_distributed_operator ${outputpath}

# Display output
python3 ../tools/plot_hmatrix.py --inputfile ../output/examples/use_distributed_operator/local_hmatrix_0.csv
