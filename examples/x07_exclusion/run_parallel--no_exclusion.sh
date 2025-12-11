#!/bin/bash
NPROCS=8

fpath='inputs/exclusions/no_exclusion.yaml'
exclusions_yaml=${fpath##*/}
work_dir=${exclusions_yaml%.yaml}
prefix="${work_dir}/nonuniform"
mkdir -p $work_dir

seq 512 1023 | parallel -j $NPROCS "echo 'Running opt prob {} with $exclusions_yaml'; ./optimization_sweep.py {} $exclusions_yaml &> ${prefix}{}.log"

./check_full_sweep.sh
./check_floris_logs.sh
