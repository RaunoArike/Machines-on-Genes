#!/bin/bash

#SBATCH --job-name="nanostars_test"
#SBATCH --partition=compute
#SBATCH --account=research-as-bn
#SBATCH --time=07:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=100MB

module load 2022r2
module load openmpi
module load python
module load py-numpy
module load py-scipy
module load py-scikit-learn

srun python main.py > output.log