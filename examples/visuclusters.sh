#!/bin/bash

# Initialization
cd ../
mkdir -p build & cd build
cmake ../
make VisuCluster
mkdir -p ../output/examples/visucluster

# Arguments
outputpath=../output/examples/visucluster/

# Run
mpirun -np 2 ./examples/visucluster ${outputpath}


# python3 ../tools/plot_cluster.py --inputfile ../output/examples/visucluster/clustering_output.csv --depth 2