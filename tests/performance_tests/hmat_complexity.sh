#!/bin/bash

# Initialization
mkdir -p ../../build & cd ../../build
cmake ../
make build-Hmat
cd - 

# HPC data
node=8
thread=4
procs_per_node=16

# Htool inputs
epsilon=0.01
eta=100
minclustersize=100

# Arguments
outputpath=../../output/tests/performance_tests/complexity/
logpath=../../log/tests/performance_tests/complexity/
mkdir -p ${outputpath}
mkdir -p ${logpath}
distance=1
sizes=(100 1000 10000 100000)

executable=../../build/tests/performance_tests/Hmat_partialACA
time=00:30:00

# Run
for size in "${sizes[@]}"
do

    ntask=$((node*procs_per_node/thread))
    nr=${size}
    nc=${size}
    signature=hmat_complexity_partialACA_${node}_${nr}_${nc}
    
    outputfile=${outputpath}/${signature}.eno
    logfile=${logpath}/${signature}

    ./launch_slurm.sh ${node} ${ntask} $((ntask/node)) ${thread} ${time} ${logfile} ${executable} ${distance} ${outputfile} ${outputpath} ${epsilon} ${eta} ${minclustersize} ${nr} ${nc}
done
