#!/bin/bash
#SBATCH --job-name=ard
#SBATCH --account=erf
#SBATCH --time=10:00:00
#SBATCH --ntasks=96
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --qos=high

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

prefix='nonuniform'
export prefix

starts=`seq 0 1023`
threshold_values='0.18 0.2 0.22 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.8 0.9 1'  # 25

run_opt() {
    probnum=$1
    thresh=$2

    work_dir="thresh_${thresh}"
    mkdir -p $work_dir

    outfile="${work_dir}/${prefix}${probnum}.log"

    if grep '^Optimization' $outfile &> /dev/null; then
        echo "Skipping opt prob $probnum with thresh=$thresh"
    else
        if [ -f "$outfile" ]; then
            echo "Re-running opt prob $probnum with thresh=$thresh"
            mv $outfile $outfile.last
        else
            echo "Running opt prob $probnum with thresh=$thresh"
        fi

        ./optimization_sweep_with_eco_constraint.py $probnum $thresh &> $outfile
    fi
}
export -f run_opt  # make available to subshells spawned by gnu parallel

echo "Starting optimization sweep at `date`"

parallel -j $NPROCS run_opt ::: $starts ::: $threshold_values

echo "Finished optimization sweep at `date`"

./check_full_sweep.sh
./check_floris_logs.sh
