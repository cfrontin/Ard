#!/usr/bin/env python
# coding: utf-8

from pathlib import Path  # optional, for nice path specifications
import glob

import pprint as pp  # optional, for nice printing
import numpy as np  # numerics library
import matplotlib.pyplot as plt  # plotting capabilities

import ard  # technically we only really need this
from ard.utils.io import load_yaml  # we grab a yaml loader here
from ard.api import set_up_ard_model  # the secret sauce
from ard.viz.layout import plot_layout  # a plotting tool!
import openmdao.api as om

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import copy

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

maxiter = 250

# prefix for each realization, within each exclusion condition directory
name_prefix = 'nonuniform'

# list of threshold polygons, with decreasing threshold values (i.e.,
# increasing area restriction)
threshold_defs = [
    'thresh_0p88.yaml',
    'thresh_0p8.yaml',
    'thresh_0p71.yaml',
    'thresh_0p6.yaml',
    'thresh_0p48.yaml',
    'thresh_0p4.yaml',
    'thresh_0p33.yaml',
]

#==============================================================================

input_dict['analysis_options']['driver']['options']['opt_settings']['maxiter'] = maxiter
input_dict['modeling_options']['layout']['N_turbines'] = Nturb
assert wio['wind_farm']['turbine']['rotor_diameter'] == rotorD

def run_opt_with_random_turbine_locs(workdir, name, exclusions_yaml=None,
                                     seed=12345, x_turbine=None, y_turbine=None):
    print('********************************************')
    print('********************************************')
    print('********************************************')
    print(f'    NEW RUN (seed={seed}): {name}')
    print(f'    exclusion polygons: {exclusions_yaml}')
    print('********************************************')
    print('********************************************')
    print('********************************************')

    # modify input dict
    my_input_dict = copy.deepcopy(input_dict)
    excluded_polygons = {}
    if exclusions_yaml is not None:
        exclusion_def = load_yaml(path_inputs / "exclusions" / exclusions_yaml)
        # for plotting
        excluded_polygons = exclusion_def['polygons'][0]
        exclusion_area = Polygon(list(zip(excluded_polygons['x'], excluded_polygons['y'])))
    if excluded_polygons:
        my_input_dict['modeling_options']['windIO_plant']['site']['exclusions'] = exclusion_def
    else:
        del my_input_dict['modeling_options']['windIO_plant']['site']['exclusions']
        del my_input_dict['system']['systems']['exclusions']
        del my_input_dict['analysis_options']['constraints']['exclusion_distances']

    # random initial points
    xturb0, yturb0 = [], []
    if x_turbine is None and y_turbine is None:
        rng = np.random.default_rng(seed=seed)

        min_x, min_y, max_x, max_y = boundary.bounds
        while len(xturb0) < Nturb:
            random_x = rng.uniform(min_x, max_x)
            random_y = rng.uniform(min_y, max_y)
            if Point(random_x, random_y).within(boundary):
                too_close = False
                if len(xturb0) > 0:
                    for xt,yt in zip(xturb0,yturb0):
                        dist2 = (random_x - xt)**2 + (random_y - yt)**2
                        if dist2 < init_min_spacing**2:
                            too_close = True
                            break
                if not too_close:
                    xturb0.append(random_x)
                    yturb0.append(random_y)
        print('RANDOM INIT TURBINE POSITIONS')
    else:
        xturb0 = x_turbine
        yturb0 = y_turbine
        print('PREV TURBINE POSITIONS')
    for iturb,(xt,yt) in enumerate(zip(xturb0,yturb0)):
        print(f'{iturb} : {xt:g} {yt:g}')

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
    if excluded_polygons:
        ax.fill(*exclusion_area.exterior.xy, color='k', facecolor='none', hatch='///',
                label='exclusion zone')
        ax.set_title(exclusions_yaml)
    else:
        ax.set_title('no area exclusion')
    for xt,yt in zip (xturb0,yturb0):
        ax.plot(xt, yt, '.')#, label='random initial layout')
    for xt,yt in zip (xturb,yturb):
        ax.plot(xt, yt, 'y*', alpha=0.4, markersize=12)#, label='optimal layout')
    #ax.legend(loc='best')
    ax.axis('equal')
    fig.savefig(prob.get_outputs_dir() / 'layout.png')

    return xturb, yturb


def get_turb_locs(work_dir, name):
    fname = input_dict["analysis_options"]["recorder"]["filepath"]
    print(f'{work_dir}/{name}_out/{fname}')
    recfiles = glob.glob(f'{work_dir}/{name}_out/{fname}')
    print(recfiles[0])
    assert len(recfiles) == 1

    # Extract the driver cases
    cr = om.CaseReader(recfiles[0])
    cases = cr.get_cases("driver")

    xturb = cases[-1].get_val('x_turbines', units='m')
    yturb = cases[-1].get_val('y_turbines', units='m')

    return xturb, yturb


#==============================================================================
#==============================================================================
#==============================================================================
if __name__ == '__main__':
    import sys, os
    #seed = 42
    try:
        probnum = int(sys.argv[1])
    except IndexError:
        probnum = 0
    #seed += probnum

    name = f'{name_prefix}{probnum:02d}'

    # read specified 
    work_dir = 'no_exclusion'
    #xturb,yturb = run_opt_with_random_turbine_locs(work_dir, name, seed=seed)
    xturb,yturb = get_turb_locs(work_dir, name)

    for exclusions_yaml in threshold_defs:
        work_dir = os.path.splitext(exclusions_yaml)[0]
        xturb,yturb = run_opt_with_random_turbine_locs(work_dir, name,
                                                       exclusions_yaml=exclusions_yaml,
                                                       x_turbine=xturb,
                                                       y_turbine=yturb)
