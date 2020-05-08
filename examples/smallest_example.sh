#!/bin/bash

# Initialization
MY_PATH="`dirname \"$0\"`"             
MY_PATH="`( cd \"$MY_PATH\" && pwd )`" 
cd ${MY_PATH}
cd ../
mkdir -p build & cd build
cmake ../
make Smallest_example
mkdir -p ../output/examples/smallest_example

# Arguments
outputpath=../output/examples/smallest_example/

# Run
mpirun -np 2 ./examples/smallest_example ${outputpath}



# python3 ../tools/plot_hmatrix.py --inputfile ../output/examples/smallest_example/smallest_example_plot --sizeWorld 2
