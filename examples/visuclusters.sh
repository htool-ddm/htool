#!/bin/bash

# Initialization
cd ../data/data_example
gmsh -2 disk.geo
cd ../../
mkdir -p build & cd build
cmake ../
make VisuCluster
mkdir -p ../output/examples/visucluster

# Arguments
mesh=../data/data_example/disk.msh
outputfile=../output/examples/visucluster/clusters
depths=(1 2 3)

# Run
for depth in "${depths[@]}"
do
    ./examples/VisuCluster ${depth} ${mesh} ${outputfile}_${depth}.msh
done