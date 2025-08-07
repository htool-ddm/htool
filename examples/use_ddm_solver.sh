#!/bin/bash

# Initialization
MY_PATH="`dirname \"$0\"`"             
MY_PATH="`( cd \"$MY_PATH\" && pwd )`" 
cd ${MY_PATH}
cd ../
mkdir -p build & cd build
cmake ../
make Use_ddm_solver
mkdir -p ../output/examples/use_ddm_solver

# Arguments
outputpath=../output/examples/use_ddm_solver/

# Run
mpirun -np 4 ./examples/Use_ddm_solver ${outputpath}
