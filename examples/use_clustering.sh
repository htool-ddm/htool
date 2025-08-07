#!/bin/bash

# Initialization
MY_PATH="`dirname \"$0\"`"             
MY_PATH="`( cd \"$MY_PATH\" && pwd )`" 
cd ${MY_PATH}
cd ../
mkdir -p build & cd build
cmake ../
make Use_clustering
mkdir -p ../output/examples/use_clustering

# Arguments
outputpath=../output/examples/use_clustering/

# Run
./examples/Use_clustering ${outputpath}

# Display output
python3 ../tools/plot_cluster.py --inputfile ../output/examples/use_clustering/clustering_output.csv --depth 2
