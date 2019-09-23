#!/bin/bash

# Initialization
mkdir -p ../../build & cd ../../build
cmake ../
make build-Hmat
cd -

# HPC data
nodes=(2 4 8 16)
threads=(2 4 8)
procs_per_node=16

# Htool inputs
epsilon=0.00001
eta=100
minclustersize=100

# Arguments
outputpath=../../output/tests/performance_tests/scaling/
logpath=../../log/tests/performance_tests/scaling/
mkdir -p ${outputpath}
mkdir -p ${logpath}
distance=0.05
nr=1000000
nc=1000000
executable=../../build/tests/performance_tests/Hmat_partialACA
time=00:30:00

# Run
for node in "${nodes[@]}"
do
    for thread in "${threads[@]}"
    do
        ntask=$((node*procs_per_node/thread))
        signature=hmat_scaling_partialACA_${node}_${thread}_${nr}_${nc}
        
        outputfile=${signature}.eno
        logfile=${logpath}/${signature}

        ./launch_slurm.sh ${node} ${ntask} $((ntask/node)) ${thread} ${time} ${logfile} ${executable} ${distance} ${outputfile} ${outputpath} ${epsilon} ${eta} ${minclustersize} ${nr} ${nc}
    done
done
