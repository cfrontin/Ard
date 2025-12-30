#!/bin/bash
#SBATCH --job-name=layouts
#SBATCH --time=1:00:00
#SBATCH --ntasks=64
#SBATCH -N 1
#SBATCH --exclusive

# Kestrel
# -------
# module purge
# module load conda/2024.06.1
# module load parallel/20220522
# conda activate ard-env
NPROCS=64 # 1.5 min for 50 iters

caselist='n1024_P5.txt'

parallel -a $caselist -j $NPROCS "echo 'Running opt prob {}'; ./optimization_sweep_warmstart.py {} &> warmstart{}--run0.log"

#./check_full_sweep.sh
./check_floris_logs.sh
