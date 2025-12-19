from pathlib import Path  # optional, for nice path specifications

import pprint as pp  # optional, for nice printing
import numpy as np  # numerics library
import matplotlib.pyplot as plt  # plotting capabilities
import pandas as pd
import seaborn as sns

import wisdem.optimization_drivers as opt_drivers

import ard  # technically we only really need this
from ard.utils.io import load_yaml  # we grab a yaml loader here
from ard.api import set_up_ard_model  # the secret sauce
from ard.viz.layout import plot_layout  # a plotting tool!

import openmdao.api as om  # for N2 diagrams from the OpenMDAO backend

# %matplotlib inline

# load input
path_inputs = Path.cwd().absolute() / "inputs"
input_dict = load_yaml(path_inputs / "ard_system.yaml")

# create and setup system

prob = set_up_ard_model(input_dict=input_dict, root_data_path=path_inputs)
prob.model.set_input_defaults('x_turbines', val=input_dict["modeling_options"]["windIO_plant"]["wind_farm"]["layouts"]["coordinates"]["x"])
prob.model.set_input_defaults('y_turbines', val=input_dict["modeling_options"]["windIO_plant"]["wind_farm"]["layouts"]["coordinates"]["y"])

if False:
    # visualize model
    om.n2(prob)

# run the model
prob.run_model()

# collapse the test result data
test_data = {
    "AEP_val": float(prob.get_val("AEP_farm", units="GW*h")[0]),
    # "CapEx_val": float(prob.get_val("tcc.tcc", units="MUSD")[0]),
    # "BOS_val": float(prob.get_val("landbosse.total_capex", units="MUSD")[0]),
    # "OpEx_val": float(prob.get_val("opex.opex", units="MUSD/yr")[0]),
    # "LCOE_val": float(prob.get_val("financese.lcoe", units="USD/MW/h")[0]),
    "coll_length": float(prob.get_val("collection.total_length_cables", units="km")[0]),
    "turbine_spacing": float(
        np.min(prob.get_val("spacing_constraint.turbine_spacing", units="km"))
    ),
    "blade_root_load": float(prob.get_val("aepFLORIS.blade_root_DEL", units="kN*m")[0]),
    "shaft_load": float(prob.get_val("aepFLORIS.shaft_DEL", units="kN*m")[0]),
    "tower_base_load": float(prob.get_val("aepFLORIS.tower_base_DEL", units="kN*m")[0]),
    "yaw_bearings_load": float(prob.get_val("aepFLORIS.yaw_bearings_DEL", units="kN*m")[0]),
}

print("\n\nRESULTS:\n")
pp.pprint(test_data)
print("\n\n")

optimize = True  # set to False to skip optimization
if optimize:
    # run the optimization
    prob.run_driver()
    prob.cleanup()

    # collapse the test result data
    test_data = {
        "AEP_val": float(prob.get_val("AEP_farm", units="GW*h")[0]),
        # "CapEx_val": float(prob.get_val("tcc.tcc", units="MUSD")[0]),
        # "BOS_val": float(prob.get_val("landbosse.total_capex", units="MUSD")[0]),
        # "OpEx_val": float(prob.get_val("opex.opex", units="MUSD/yr")[0]),
        # "LCOE_val": float(prob.get_val("financese.lcoe", units="USD/MW/h")[0]),
        "area_tight": float(prob.get_val("landuse.area_tight", units="km**2")[0]),
        "coll_length": float(
            prob.get_val("collection.total_length_cables", units="km")[0]
        ),
        "turbine_spacing": float(
            np.min(prob.get_val("spacing_constraint.turbine_spacing", units="km"))
        ),
        "blade_root_load": float(prob.get_val("aepFLORIS.blade_root_DEL", units="kN*m")[0]),
        "shaft_load": float(prob.get_val("aepFLORIS.shaft_DEL", units="kN*m")[0]),
        "tower_base_load": float(prob.get_val("aepFLORIS.tower_base_DEL", units="kN*m")[0]),
        "yaw_bearings_load": float(prob.get_val("aepFLORIS.yaw_bearings_DEL", units="kN*m")[0]),
    }

    # clean up the recorder
    prob.cleanup()

    # print the results
    print("\n\nRESULTS (opt):\n")
    pp.pprint(test_data)
    print("\n\n")

plot_layout(
    prob,
    input_dict=input_dict,
    show_image=False,
    include_cable_routing=True,
)
plt.savefig('layout.png')
plt.close()

# Access the recorder data
case_reader = om.CaseReader(prob.get_outputs_dir() / "cases.sql")

# Get all driver cases
driver_cases = case_reader.list_cases("driver", out_stream=None)
# problem_cases = case_reader.list_cases("problem")
# problem_vars = case_reader.list_source_vars('problem')
# print(print(problem_vars['outputs']))

# print(f"Number of cases: {len(system_cases)}")

# for case_num, case_id in enumerate(system_cases):
#     case = case_reader.get_case(case_id)

#     # Get the names of all the outputs to the objective component
#     outputs = case.outputs.keys()

#     # Get the rounded value of each output (as a python list)
#     values = [(name, case[name].round(10).tolist()) for name in outputs]

#     # Print the output values for this case
#     print(values)
#     break

# Extract data from all cases
driver_results = []
for case_id in driver_cases:

    case = case_reader.get_case(case_id)
    # print(case)

    # Extract specific variables you're interested in
    result = {
        "case_id": case_id,
        "AEP": float(case.get_val("AEP_farm", units="GW*h")[0]),
        "area_tight": float(case.get_val("landuse.area_tight", units="km**2")[0]),
        "blade_root_DEL": float(case.get_val("aepFLORIS.blade_root_DEL", units="kN*m")[0]),
        "shaft_DEL": float(case.get_val("aepFLORIS.shaft_DEL", units="kN*m")[0]),
        "tower_base_DEL": float(case.get_val("aepFLORIS.tower_base_DEL", units="kN*m")[0]),
        "yaw_bearings_DEL": float(case.get_val("aepFLORIS.yaw_bearings_DEL", units="kN*m")[0]),
        # "total_length_cables": float(case.get_val("collection.total_length_cables", units="km")[0]),
    }
    driver_results.append(result)

# floris_cases = case_reader.list_cases("root.aepFLORIS", recurse=False)
# floris_results = []
# for case_id in floris_cases:

#     case = case_reader.get_case(case_id)
#     # print(case)

#     # Extract specific variables you're interested in
#     result = {
#         "case_id": case_id,
#         "blade_root_DEL": float(case.get_val("aepFLORIS.blade_root_DEL", units="kN*m")[0]),
#         "shaft_DEL": float(case.get_val("aepFLORIS.shaft_DEL", units="kN*m")[0]),
#         "tower_base_DEL": float(case.get_val("aepFLORIS.tower_base_DEL", units="kN*m")[0]),
#         "yaw_bearings_DEL": float(case.get_val("aepFLORIS.yaw_bearings_DEL", units="kN*m")[0]),
#     }
#     floris_results.append(result)

# collection_cases = case_reader.list_cases("root.collection", recurse=False)
# collection_results = []
# for case_id in collection_cases:

#     case = case_reader.get_case(case_id)
#     # print(case)

#     # Extract specific variables you're interested in
#     result = {
#         "case_id": case_id,
#         "total_length_cables": float(case.get_val("collection.total_length_cables", units="km")[0]),
#     }
#     collection_results.append(result)

# Convert to arrays for plotting/analysis
case_id_history = np.array([int(r["case_id"].split('|')[-1]) for r in driver_results])
aep_history = np.array([r["AEP"] for r in driver_results])
area_history = np.array([r["area_tight"] for r in driver_results])
blade_root_DEL_history = np.array([r["blade_root_DEL"] for r in driver_results])
shaft_DEL_history = np.array([r["shaft_DEL"] for r in driver_results])
tower_base_DEL_history = np.array([r["tower_base_DEL"] for r in driver_results])
yaw_bearings_DEL_history = np.array([r["yaw_bearings_DEL"] for r in driver_results])
# total_length_cables_history = np.array([r["total_length_cables"] for r in driver_results])

# Create a correlation matrix
# obj_data = pd.DataFrame({
#     'AEP': aep_history,
#     'DEL': DEL_history,
#     'Cable Length': total_length_cables_history,
# })
obj_data = pd.DataFrame({
    'AEP': aep_history,
    "Area": area_history,
    'Blade Root DEL': blade_root_DEL_history,
    'Shaft DEL': shaft_DEL_history,
    'Tower Base DEL': tower_base_DEL_history,
    'Yaw Bearings DEL': yaw_bearings_DEL_history,
    # 'Cable Length': total_length_cables_history
})
obj_data["pareto_rank"] = None

idx_pareto = opt_drivers.nsga2.fast_nondom_sort.fast_nondom_sort(
    np.vstack([
        -aep_history,
        area_history,
        # blade_root_DEL_history,
        # shaft_DEL_history,
        # tower_base_DEL_history,
        # yaw_bearings_DEL_history,
        # total_length_cables_history
    ]).T
)
for pareto_rank, indices in enumerate(idx_pareto):
    for index in indices:
        obj_data.loc[index, "pareto_rank"] = pareto_rank
obj_data.sort_values(
    [
        "pareto_rank",
        "AEP",
        "Area",
        "Blade Root DEL",
        "Shaft DEL",
        "Tower Base DEL",
        "Yaw Bearings DEL",
        # "Cable Length"
    ],
    ascending=False,
    inplace=True,
)
obj_data["is_pareto"] = (obj_data["pareto_rank"] == 0)

print(obj_data)

sns.pairplot(
    data=obj_data,
    vars=[
        "AEP",
        "Area",
        "Blade Root DEL",
        "Shaft DEL",
        "Tower Base DEL",
        "Yaw Bearings DEL",
        # "Cable Length"
    ],
    hue="is_pareto",
)

plt.savefig('aep_vs_area.png')
plt.close()
# plt.show()

sns.pairplot(
    data=obj_data[obj_data["is_pareto"]],
    vars=[
        "AEP",
        "Area",
        "Blade Root DEL",
        "Shaft DEL",
        "Tower Base DEL",
        "Yaw Bearings DEL",
        # "Cable Length"
    ],
    hue="is_pareto",
)

plt.savefig('aep_vs_area_pareto.png')
plt.close()
# plt.show()
