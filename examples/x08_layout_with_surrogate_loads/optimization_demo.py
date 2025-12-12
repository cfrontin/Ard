from pathlib import Path  # optional, for nice path specifications

import pprint as pp  # optional, for nice printing
import numpy as np  # numerics library
import matplotlib.pyplot as plt  # plotting capabilities
import pandas as pd
import seaborn as sns

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
    "CapEx_val": float(prob.get_val("tcc.tcc", units="MUSD")[0]),
    "BOS_val": float(prob.get_val("landbosse.total_capex", units="MUSD")[0]),
    "OpEx_val": float(prob.get_val("opex.opex", units="MUSD/yr")[0]),
    "LCOE_val": float(prob.get_val("financese.lcoe", units="USD/MW/h")[0]),
    "coll_length": float(prob.get_val("collection.total_length_cables", units="km")[0]),
    "turbine_spacing": float(
        np.min(prob.get_val("spacing_constraint.turbine_spacing", units="km"))
    ),
    "blade_root_DEL": float(prob.get_val("aepFLORIS.blade_root_DEL", units="kN*m")[0]),
    "shaft_DEL": float(prob.get_val("aepFLORIS.shaft_DEL", units="kN*m")[0]),
    "tower_base_DEL": float(prob.get_val("aepFLORIS.tower_base_DEL", units="kN*m")[0]),
    "yaw_bearings_DEL": float(prob.get_val("aepFLORIS.yaw_bearings_DEL", units="kN*m")[0]),
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
        "blade_root_DEL": float(prob.get_val("aepFLORIS.blade_root_DEL", units="kN*m")[0]),
        "shaft_DEL": float(prob.get_val("aepFLORIS.shaft_DEL", units="kN*m")[0]),
        "tower_base_DEL": float(prob.get_val("aepFLORIS.tower_base_DEL", units="kN*m")[0]),
        "yaw_bearings_DEL": float(prob.get_val("aepFLORIS.yaw_bearings_DEL", units="kN*m")[0]),
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
    show_image=True,
    include_cable_routing=True,
)

# Access the recorder data
case_reader = om.CaseReader(prob.get_outputs_dir() / "cases.sql")

# Get all driver cases
driver_cases = case_reader.list_cases("driver", out_stream=None)

# Extract data from all cases
results = []
for case_id in driver_cases:

    case = case_reader.get_case(case_id)

    # Extract specific variables you're interested in
    result = {
        "case_id": case_id,
        "AEP": float(case.get_val("AEP_farm", units="GW*h")[0]),
        "DEL": float(case.get_val("aepFLORIS.tower_base_load", units="kN*m")[0]),
        "total_length_cables": float(case.get_val("collection.total_length_cables", units="km")[0]),
    }
    results.append(result)

# Convert to arrays for plotting/analysis
case_id_history = np.array([int(r["case_id"].split('|')[-1]) for r in results])
aep_history = np.array([r["AEP"] for r in results])
DEL_history = np.array([r["DEL"] for r in results])
total_length_cables_history = np.array([r["total_length_cables"] for r in results])

# Create a correlation matrix
sns.pairplot(
    data = pd.DataFrame({
        'AEP': aep_history,
        'DEL': DEL_history,
        'Cable Length': total_length_cables_history
    }),

)
plt.show()

obj_nd = prob.driver.obj_nd.copy()
obj_nd = obj_nd[obj_nd[:, 0].argsort()]  # Sort rows by the first column
print(obj_nd)
plt.plot(*(obj_nd.T))
plt.show()
