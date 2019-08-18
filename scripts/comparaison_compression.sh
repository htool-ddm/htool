#!/bin/bash

# Initialization
cd ../
mkdir -p build & cd build
cmake ../
make Compression_comparison
mkdir -p ../output/compression_comparison

# Arguments
outputpath=../output/compression_comparison/
distances=(1 2 3)

# Run
for distance in "${distances[@]}"
do
    ./examples/compression_comparison ${distance} compression_comparison_${distance}.csv ${outputpath}
done
