#!/bin/bash
NPROCS=8

for fpath in inputs/exclusions/*.yaml; do

    exclusions_yaml=${fpath##*/}
    work_dir=${exclusions_yaml%.yaml}
    prefix="${work_dir}/nonuniform"
    mkdir -p $work_dir

    seq 0 511 | parallel -j $NPROCS "echo 'Running opt prob {} with $exclusions_yaml'; ./optimization_sweep.py {} $exclusions_yaml &> ${prefix}{}.log"

done

./check_full_sweep.sh
./check_floris_logs.sh
