#!/usr/bin/env python
import os
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt

from helpers import postproc_workdir

casedir = 'no_exclusion--prelim'
prefix = 'nonuniform'
cutoff = 5. # percentile

#lcoe, _, _, _ = postproc_workdir(casedir,prefix)

# scraping from log files is faster
logfiles = glob.glob(f'{casedir}/{prefix}*.log')
lcoe = []
probnum = []
for log in logfiles:
    probstr = os.path.split(log)[1]
    prob = int(probstr[len(prefix):-4])
    probnum.append(prob)
    with open(log,'r') as f:
        for line in f:
            if line.startswith(" 'LCOE_val'"):
                lcoe_val = float(line.strip().rstrip(',').split()[-1])
                lcoe.append(lcoe_val)
    print(log,lcoe[-1])
lcoe = np.array(lcoe)

fig,ax = plt.subplots()
ax.hist(lcoe)
fig.savefig('lcoe_histogram--prelim.png',bbox_inches='tight')

P5 = np.percentile(lcoe, cutoff)
print(f'{cutoff}th percentile LCOE =',P5,'USD/MWh')

indices = np.where(lcoe <= P5)
selected_layouts = np.array([probnum[i] for i in indices[0]])
selected_layouts.sort()
print('Selected layouts:', len(selected_layouts), selected_layouts)

with open(f'n{len(lcoe)}_P{cutoff:g}.txt','w') as f:
    selected_layouts.tofile(f,sep='\n')
