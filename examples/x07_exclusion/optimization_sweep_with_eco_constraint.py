#!/usr/bin/env python
# coding: utf-8

from pathlib import Path  # optional, for nice path specifications

import pprint as pp  # optional, for nice printing
import numpy as np  # numerics library
import pandas as pd
import matplotlib.pyplot as plt  # plotting capabilities

import ard  # technically we only really need this
from ard.utils.io import load_yaml  # we grab a yaml loader here
from ard.api import set_up_ard_model  # the secret sauce
from ard.viz.layout import plot_layout  # a plotting tool!

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import copy

from helpers import scrape_log

# load input
path_inputs = Path.cwd().absolute() / "inputs"
input_dict = load_yaml(path_inputs / "ard_system.yaml")

wio = input_dict['modeling_options']['windIO_plant']

boundary_poly = wio['site']['boundaries']['polygons'][0]
boundary = Polygon(list(zip(boundary_poly['x'], boundary_poly['y'])))

#==============================================================================
# modify this section

Nturb = 7
rotorD = 127.
init_min_spacing = 1.0 * rotorD

maxiter0 = 50 # first pass only
maxiter = 200
maxiter = 250
maxiter = 500

# prefix for each realization, within each exclusion condition directory
name_prefix = 'nonuniform'

# pre-generated initial layouts with latin hypercube sampling
#init_layout_file = 'structured_layout_samples.csv'

init_layout_file = 'random_from_grid.csv'
init_layout_offset = 0

eagle_scaler = 1.0

#==============================================================================

input_dict['analysis_options']['driver']['options']['opt_settings']['maxiter'] = maxiter0
input_dict['modeling_options']['layout']['N_turbines'] = Nturb
assert wio['wind_farm']['turbine']['rotor_diameter'] == rotorD

def run_opt_with_random_turbine_locs(seed, workdir, name,
                                     threshold_value=1.0,
                                     prev_log=None):
    print('********************************************')
    print('********************************************')
    print('********************************************')
    print(f'    NEW RUN (seed={seed}): {name}')
    print(f'    threshold value: {threshold_value}')
    print('********************************************')
    print('********************************************')
    print('********************************************')

    # modify input dict
    my_input_dict = copy.deepcopy(input_dict)
    my_input_dict['analysis_options']['constraints']['eagle_normalized_density']['upper'] = threshold_value
    my_input_dict['analysis_options']['constraints']['eagle_normalized_density']['scaler'] = eagle_scaler

    # get initial layout
    if prev_log is None:
# read from Cory's csv
#        init_layouts = pd.read_csv('structured_layout_samples.csv')
#        xturbstr = init['x_turbines']
#        yturbstr = init['y_turbines']
#        assert xturbstr[0]=='[' and xturbstr[-1]==']'
#        assert yturbstr[0]=='[' and yturbstr[-1]==']'
#        xturb0 = list(map(float,xturbstr[1:-1].split()))
#        yturb0 = list(map(float,yturbstr[1:-1].split()))
        init_layouts = pd.read_csv(init_layout_file,header=[0,1],index_col=0)
        init = init_layouts.iloc[seed]
        init = init.unstack().transpose()
        xturb0 = init['x']
        yturb0 = init['y']
    else:
        _, xturb, yturb = scrape_log(prev_log)        
        xturb0 = xturb[0]
        yturb0 = yturb[0]
        my_input_dict['analysis_options']['driver']['options']['opt_settings']['maxiter'] = maxiter

    #=========================#
    # create and setup system #
    #=========================#
    prob = set_up_ard_model(
        input_dict=my_input_dict,
        root_data_path=path_inputs,
        work_dir=work_dir,
        name=name,
    )

    prob.model.set_input_defaults("x_turbines", xturb0, units="m")
    prob.model.set_input_defaults("y_turbines", yturb0, units="m")

    #========================================#
    # Now, we can optimize the same problem! #
    #========================================#
    prob.run_driver()

    # collapse the test result data
    test_data = {
        "AEP_val": float(prob.get_val("AEP_farm", units="GW*h")[0]),
        "CapEx_val": float(prob.get_val("tcc.tcc", units="MUSD")[0]),
        "BOS_val": float(prob.get_val("landbosse.total_capex", units="MUSD")[0]),
        "OpEx_val": float(prob.get_val("opex.opex", units="MUSD/yr")[0]),
        "LCOE_val": float(prob.get_val("financese.lcoe", units="USD/MW/h")[0]),
        "coll_length": float(
            prob.get_val("collection.total_length_cables", units="km")[0]
        ),
        "turbine_spacing": float(
            np.min(prob.get_val("spacing_constraint.turbine_spacing", units="km"))
        ),
    }

    # clean up the recorder
    prob.cleanup()

    # print the results
    print("\n\nRESULTS (opt):\n")
    pp.pprint(test_data)
    print("\n\n")

    try:
        print(prob.get_val("exclusions.exclusion_distances", units="km"))
    except KeyError:
        pass

#    ax = plot_layout(
#        prob,
#        input_dict=my_input_dict,
#        show_image=False,
#        include_cable_routing=False,
#    )

    xturb = prob.get_val("x_turbines", units="m")
    yturb = prob.get_val("y_turbines", units="m")

    fig,ax = plt.subplots()
    ax.plot(*boundary.exterior.xy, 'k--')
    if threshold_value < 1:
        pres_map = input_dict['modeling_options']['ssrs']['presence_density_map']
        ax.contour(pres_map['x'], pres_map['y'],
                   np.array(pres_map['normalized_presence_density']).T,
                   levels=[threshold_value])
        ax.set_title(f'threshold: {threshold_value:g}')
    else:
        ax.set_title('no area exclusion')
    for xt,yt in zip (xturb0,yturb0):
        ax.plot(xt, yt, '.')#, label='random initial layout')
    for xt,yt in zip (xturb,yturb):
        ax.plot(xt, yt, 'y*', alpha=0.4, markersize=12)#, label='optimal layout')
    #ax.legend(loc='best')
    ax.axis('equal')
    fig.savefig(prob.get_outputs_dir() / 'layout.png')

#==============================================================================
#==============================================================================
#==============================================================================
if __name__ == '__main__':
    import sys, os
    try:
        problist = [int(sys.argv[1])]
    except IndexError:
        problist = range(10)

    try:
        threshold_value = float(sys.argv[2])
    except IndexError:
        threshold_value = 1.0
    work_dir = f'thresh_{threshold_value:g}'

    try:
        prev_log = sys.argv[3]
    except IndexError:
        prev_log = None
    else:
        assert os.path.isfile(prev_log)

    for probnum in problist:
        name = f'{name_prefix}{probnum:02d}'
        seed = probnum + init_layout_offset

        run_opt_with_random_turbine_locs(seed, work_dir, name, threshold_value, prev_log)
