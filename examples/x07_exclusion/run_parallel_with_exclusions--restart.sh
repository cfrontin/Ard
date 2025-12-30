#!/bin/bash
#SBATCH --job-name=ard
#SBATCH --account=erf
#SBATCH --time=4:00:00
#SBATCH --ntasks=96
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
NPROCS=96 # 1.5 min for 50 iters

RESTART='firstpass'

for fpath in inputs/exclusions/*.yaml; do

    exclusions_yaml=${fpath##*/}
    work_dir=${exclusions_yaml%.yaml}
    prefix="${work_dir}/nonuniform"
    mkdir -p $work_dir

    jobsuffix=${exclusions_yaml##*/}
    jobsuffix=${jobsuffix%.yaml}

    # Call optimization_sweep.py with
    # - prob ID
    # - exclusion definition
    # - previous log to scrape the turbine positions from
    parallel -a n1024_P5_$jobsuffix.txt -j $NPROCS "echo 'Running opt prob {} with $exclusions_yaml'; prevlog=$RESTART/${prefix}{}.log; cp $prevlog $work_dir/; ./optimization_sweep.py {} $exclusions_yaml $prevlog 2>&1 >> ${prefix}{}.log"

done

./check_full_sweep.sh
./check_floris_logs.sh
