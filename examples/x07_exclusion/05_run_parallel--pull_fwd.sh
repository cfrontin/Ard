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

caselist='n1024_P5--run1.txt'
oldsuffix='run0'
newsuffix='run1'

for dname in {no_exclusion,thresh_0p?,thresh_0p??}; do
    mv -v $dname "$dname--$oldsuffix"
done

# run optimization_sweep_warmstart.py with the final layout from the previous run {1} at the specified threshold {2}
parallel -a $caselist --colsep ' ' -j $NPROCS "echo 'Re-running opt prob {1}'; ./optimization_sweep_warmstart.py {1} {2}--$oldsuffix &> warmstart{1}--$newsuffix.log"

#./check_full_sweep.sh
./check_floris_logs.sh
