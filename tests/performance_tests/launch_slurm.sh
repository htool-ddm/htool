#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH -J HTOOL
#SBATCH --nodes=$1
#SBATCH --ntasks=$2
##SBATCH --constraint=BDW28
#SBATCH --constraint=HSW24
#SBATCH --mem=118000M
#SBATCH --ntasks-per-node=$3
#SBATCH --cpus-per-task=$4
##SBATCH --mem-per-cpu=1000M
#SBATCH --time=${5}
#SBATCH --output ${6}log.output
#SBATCH --error ${6}log.error
module purge
source ~/.profile

export OMP_NUM_THREADS=$4
srun --mpi=pmi2 -K1 --resv-ports -n $2 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15}
EOT
