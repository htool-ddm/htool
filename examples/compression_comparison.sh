#!/bin/bash

# Initialization
cd ../
mkdir -p build & cd build
cmake ../
make Compression_comparison
mkdir -p ../output/examples/compression_comparison

# Arguments
outputpath=../output/examples/compression_comparison/
distances=(1 2 3)

# Run
for distance in "${distances[@]}"
do
    ./examples/compression_comparison ${distance} compression_comparison_${distance}.csv ${outputpath}
done


python3 ../tools/plot_comparison_compression.py --inputfile ../output/examples/compression_comparison/compression_comparison_1.csv
