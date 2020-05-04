#!/bin/bash

# Initialization
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
