#!/usr/bin/env python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from helpers import postproc_workdir

thresholds = [0.33, 0.4, 0.48, 0.6, 0.71, 0.8, 0.88, 1.0]
thresholds = thresholds[::-1]
caselist = 'n1024_P5.txt'
suffix = '--run0' # suffix appended to logs from the last run, and
                  # the suffix that will be added to the workdir names by 05_run_parallel--pull_fwd.sh
newsuffix = '--run1'

probnum = np.loadtxt(caselist).astype(int)
if suffix == '':
    logfiles = [f'warmstart{prob}.log' for prob in probnum]
else:
    logfiles = [f'warmstart{prob}{suffix}.log' for prob in probnum]
assert all([os.path.isfile(log) for log in logfiles])

# scraping from log files is faster
selected_layouts = []
for i,log in enumerate(logfiles):
    lcoe_vals = []
    with open(log,'r') as f:
        for line in f:
            if line.startswith(" 'LCOE_val'"):
                lcoe_vals.append(float(line.strip().rstrip(',').split()[-1]))
    assert len(lcoe_vals) == len(thresholds)
    imin = np.argmin(lcoe_vals)
    if imin==0:
        selected_layouts.append(f'no_exclusion')
    else:
        thresh = f'{thresholds[imin]:g}'.replace('.','p')
        selected_layouts.append(f'thresh_{thresh}')
    print(i,log,imin,lcoe_vals[imin],selected_layouts[-1])

fname = os.path.splitext(caselist)[0]
fname = f'{fname}{newsuffix}.txt'
with open(fname,'w') as f:
    for prob, sel in zip(probnum,selected_layouts):
        f.write(f'{prob} {sel}\n')
