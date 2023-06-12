#!/bin/bash

#SBATCH --time=18:00:00   # walltime
#SBATCH --ntasks=40   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=4096M   # memory per CPU core
#SBATCH -J "uvo_3d"   # job name
#SBATCH --mail-user=jaringson@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

#module purge
module load python/3.6
#module load llvm/7
#module load gcc/8
#module load openmpi/3.1
#module load python-mpi4py/3.0

#cd uvo_3d
python run_monte_carlo.py
#cd ..
