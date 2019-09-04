#!/bin/bash

# Initialization
cd ../../
mkdir -p build & cd build
cmake ../
make build-Hmat_scaling
mkdir -p ../output/tests/performance_tests/scaling

# HPC data
nodes=(2 4 8 16)
threads=(2 4 8)
procs_per_node=16

# Htool inputs
epsilon=0.01
eta=100
minclustersize=100

# Arguments
outputpath=../output/tests/performance_tests/scaling/
logpath=../log/tests/performance_tests/scaling/
distance=1
nr=100000
nc=100000
executable=./tests/performance_tests/Hmat_scaling_partialACA
time=00:30:00

# Run
for node in "${nodes[@]}"
do
    for thread in "${threads[@]}"
    do
        ntask=$((node*procs_per_node/thread))
        signature=hmat_scaling_partialACA_${node}_${nr}_${nc}
        
        outputfile=${outputpath}/${signature}.eno
        logfile=${logpath}/${signature}

        ./launch_slurm.sh ${node} ${ntask} $((ntask/node)) ${thread} ${time} ${logfile} ${executable} ${distance} ${outputfile} ${outputpath} ${epsilon} ${eta} ${minclustersize} ${nr} ${nc}
    done
done
