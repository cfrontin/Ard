#!/bin/bash
NPROCS=8

if [ -n "$1" ]; then
    parallel -a $1 -j $NPROCS "echo 'Running opt prob {} with $exclusions_yaml'; ./optimization_sweep_warmstart.py {} &> warmstart{}.log"
else
    echo 'specify text file with list of starting realizations'
fi
