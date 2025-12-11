#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

from helpers import postproc_workdir

casedir = 'no_exclusion--prelim'
prefix = 'nonuniform'

lcoe, xturb, yturb, _ = postproc_workdir(casedir,prefix)

fig,ax = plt.subplots()
ax.hist(lcoe)
fig.savefig('lcoe_histogram--prelim.png',bbox_inches='tight')

P5 = np.percentile(lcoe, 5)
print('5th percentile LCOE =',P5,'USD/MWh')

selected_layouts = np.where(lcoe <= P5)[0]
print('Selected layouts:', len(selected_layouts), selected_layouts)

with open(f'n{len(lcoe)}_P5.txt','w') as f:
    selected_layouts.tofile(f,sep='\n')
