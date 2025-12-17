#!/bin/bash
#SBATCH --job-name=layouts_no_excl
#SBATCH --time=1:00:00
#SBATCH --ntasks=64
#SBATCH -N 1
#SBATCH --exclusive

source ~/.bash_profile
ard-env

# Macbook
# -------
NPROCS=8 # 0.83 min for 50 iters

# Kestrel
# -------
# module purge
# module load conda/2024.06.1
# module load parallel/20220522
# conda activate ard-env
NPROCS=64 # 1.5 min for 50 iters


work_dir='no_exclusion'
prefix="${work_dir}/nonuniform"
mkdir -p $work_dir

echo "Start at `date`"

#parallel -a runlist -j $NPROCS "echo 'Running opt prob {} with NO exclusions'; ./optimization_sweep.py {} &> ${prefix}{}.log"
seq 0 1023 | parallel -j $NPROCS "echo 'Running opt prob {} with NO exclusions'; ./optimization_sweep.py {} &> ${prefix}{}.log"

echo "Finish at `date`"

mv no_exclusion no_exclusion--prelim
