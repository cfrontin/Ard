#!/bin/bash
NPROCS=8

alias generate_seeds=`seq 0 511`

for fpath in inputs/exclusions/thresh_*.yaml; do

    exclusions_yaml=${fpath##*/}
    work_dir=${exclusions_yaml%.yaml}
    prefix="${work_dir}/nonuniform"
    mkdir -p $work_dir

    generate_seeds | parallel -j $NPROCS "echo 'Running opt prob {} with $exclusions_yaml'; ./optimization_sweep.py {} $exclusions_yaml &> ${prefix}{}.log"

done

work_dir='no_exclusion'
prefix="${work_dir}/nonuniform"
mkdir -p $work_dir
generate_seeds | parallel -j $NPROCS "echo 'Running opt prob {} with NO exclusions'; ./optimization_sweep.py {} &> ${prefix}{}.log"

./check_full_sweep.sh
./check_floris_logs.sh
