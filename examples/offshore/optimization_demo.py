from pathlib import Path

import pprint as pp

import numpy as np
import matplotlib.pyplot as plt

import floris
import openmdao.api as om

from wisdem.optimization_drivers.nlopt_driver import NLoptDriver

import optiwindnet.plotting
import ard
import ard.api.prototype_temp as glue
import ard.layout.spacing

# layout type
layout_type = "gridfarm"

# create the wind query
wind_rose_wrg = floris.wind_data.WindRoseWRG(
    Path(ard.__file__).parents[1] / "examples" / "data" / "wrg_example.wrg"
)
wind_rose_wrg.set_wd_step(90.0)
wind_rose_wrg.set_wind_speeds(np.array([5.0, 10.0, 15.0, 20.0]))
wind_rose = wind_rose_wrg.get_wind_rose_at_point(0.0, 0.0)
wind_query = ard.wind_query.WindQuery.from_FLORIS_WindData(wind_rose)

# specify the configuration/specification files to use
filename_turbine_spec = (
    Path(ard.__file__).parents[1]
    / "examples"
    / "data"
    / "turbine_spec_IEA-22-284-RWT.yaml"
)  # toolset generalized turbine specification
data_turbine_spec = ard.utils.io.load_turbine_spec(filename_turbine_spec)

# set up the modeling options
modeling_options = {
    "farm": {
        "N_turbines": 25,
        "N_substations": 1,
        "spacing_primary": 7.0,
        "spacing_secondary": 7.0,
        "angle_orientation": 0.0,
        "angle_skew": 0.0,
        "x_substations": 0.1,
        "y_substations": 0.1,
    },
    "turbine": data_turbine_spec,
    "wind_rose": {
        "file": "wrg_example.wrg",
        "wd_step": 90.0,
        "wind_speeds": [5.0, 10.0, 15.0, 20.0],
        "point": [0.0, 0.0],
    },
    "offshore": True,
    "floating": True,
    "platform": {
        "N_anchors": 3,
        "min_mooring_line_length_m": 500.0,
        "N_anchor_dimensions": 2,
    },
    "site_depth": 50.0,
    "collection": {
        "max_turbines_per_string": 8,
        "solver_name": "appsi_highs",
        "solver_options": dict(
            time_limit=60,
            mip_rel_gap=0.005,  # TODO ???
        ),
    },
}

# create the OpenMDAO model
model = om.Group()
group_layout2aep = om.Group()

# first the layout
if layout_type == "gridfarm":
    group_layout2aep.add_subsystem(  # layout component
        "layout",
        ard.layout.gridfarm.GridFarmLayout(modeling_options=modeling_options),
        promotes=["*"],
    )
    layout_global_input_promotes = [
        "angle_orientation",
        "angle_skew",
        "spacing_primary",
        "spacing_secondary",
    ]
elif layout_type == "sunflower":
    group_layout2aep.add_subsystem(  # layout component
        "layout",
        ard.layout.sunflower.SunflowerFarmLayout(modeling_options=modeling_options),
        promotes=["*"],
    )
    layout_global_input_promotes = ["spacing_target"]
else:
    raise KeyError("you shouldn't be able to get here.")
layout_global_output_promotes = [
    "spacing_effective_primary",
    "spacing_effective_secondary",
]  # all layouts have this

# group_layout2aep.add_subsystem(  # FLORIS AEP component
#     "aepPlaceholder",
#     ard.farm_aero.placeholder.PlaceholderAEP(
#         modeling_options=modeling_options,
#         wind_rose=wind_rose,
#     ),
#     # promotes=["AEP_farm"],
#     promotes=["x_turbines", "y_turbines", "AEP_farm"],
# )
group_layout2aep.add_subsystem(  # FLORIS AEP component
    "aepFLORIS",
    ard.farm_aero.floris.FLORISAEP(
        modeling_options=modeling_options,
        case_title="letsgo",
        data_path="inputs",
    ),
    # promotes=["AEP_farm"],
    promotes=["x_turbines", "y_turbines", "AEP_farm"],
)
farmaero_global_output_promotes = ["AEP_farm"]

group_layout2aep.approx_totals(
    method="fd", step=1e-3, form="central", step_calc="rel_avg"
)
model.add_subsystem(
    "layout2aep",
    group_layout2aep,
    promotes_inputs=[
        *layout_global_input_promotes,
    ],
    promotes_outputs=[
        *layout_global_output_promotes,
        *farmaero_global_output_promotes,
    ],
)

if layout_type == "gridfarm":
    model.add_subsystem(  # landuse component
        "landuse",
        ard.layout.gridfarm.GridFarmLanduse(modeling_options=modeling_options),
        promotes_inputs=layout_global_input_promotes,
    )
elif layout_type == "sunflower":
    model.add_subsystem(  # landuse component
        "landuse",
        ard.layout.sunflower.SunflowerFarmLanduse(modeling_options=modeling_options),
    )
    model.connect("layout2aep.x_turbines", "landuse.x_turbines")
    model.connect("layout2aep.y_turbines", "landuse.y_turbines")
else:
    raise KeyError("you shouldn't be able to get here.")

model.add_subsystem(  # collection component
    "optiwindnet_coll",
    ard.collection.optiwindnetCollection(
        modeling_options=modeling_options,
    ),
)
model.connect("layout2aep.x_turbines", "optiwindnet_coll.x_turbines")
model.connect("layout2aep.y_turbines", "optiwindnet_coll.y_turbines")

model.add_subsystem(  # mooring system design
    "mooring_design",
    ard.offshore.mooring_design_constant_depth.ConstantDepthMooringDesign(
        modeling_options=modeling_options,
        wind_query=None,
    ),
    promotes_inputs=["phi_platform"],
)
model.connect("layout2aep.x_turbines", "mooring_design.x_turbines")
model.connect("layout2aep.y_turbines", "mooring_design.y_turbines")

model.add_subsystem(  # regulatory constraints for mooring
    "mooring_constraint",
    ard.offshore.mooring_constraint.MooringConstraint(
        modeling_options=modeling_options,
    ),
)
model.connect("layout2aep.x_turbines", "mooring_constraint.x_turbines")
model.connect("layout2aep.y_turbines", "mooring_constraint.y_turbines")
model.connect("mooring_design.x_anchors", "mooring_constraint.x_anchors")
model.connect("mooring_design.y_anchors", "mooring_constraint.y_anchors")

model.add_subsystem(  # constraints for turbine proximity
    "spacing_constraint",
    ard.layout.spacing.TurbineSpacing(
        modeling_options=modeling_options,
    ),
)
model.connect("layout2aep.x_turbines", "spacing_constraint.x_turbines")
model.connect("layout2aep.y_turbines", "spacing_constraint.y_turbines")

model.add_subsystem(  # turbine capital costs component
    "tcc",
    ard.cost.wisdem_wrap.TurbineCapitalCosts(),
    promotes_inputs=[
        "turbine_number",
        "machine_rating",
        "tcc_per_kW",
        "offset_tcc_per_kW",
    ],
)
if modeling_options["offshore"]:
    model.add_subsystem(  # Orbit component
        "orbit",
        ard.cost.wisdem_wrap.ORBIT(floating=True),
    )
    model.connect(  # effective primary spacing for BOS
        "spacing_effective_primary", "orbit.plant_turbine_spacing"
    )
    model.connect(  # effective secondary spacing for BOS
        "spacing_effective_secondary", "orbit.plant_row_spacing"
    )
else:
    model.add_subsystem(  # LandBOSSE component
        "landbosse",
        ard.cost.wisdem_wrap.LandBOSSEArdComp(),
    )
    model.connect(  # effective primary spacing for BOS
        "spacing_effective_primary",
        "landbosse.turbine_spacing_rotor_diameters",
    )
    model.connect(  # effective secondary spacing for BOS
        "spacing_effective_secondary",
        "landbosse.row_spacing_rotor_diameters",
    )

model.add_subsystem(  # operational expenditures component
    "opex",
    ard.cost.wisdem_wrap.OperatingExpenses(),
    promotes_inputs=[
        "turbine_number",
        "machine_rating",
        "opex_per_kW",
    ],
)

model.add_subsystem(  # cost metrics component
    "financese",
    ard.cost.wisdem_wrap.PlantFinance(),
    promotes_inputs=[
        "turbine_number",
        "machine_rating",
        "tcc_per_kW",
        "offset_tcc_per_kW",
        "opex_per_kW",
    ],
)
model.connect("AEP_farm", "financese.plant_aep_in")
if modeling_options["offshore"]:
    model.connect("orbit.total_capex_kW", "financese.bos_per_kW")
else:
    model.connect("landbosse.total_capex_kW", "financese.bos_per_kW")

# build out the problem based on this model
prob = om.Problem(model)
prob.setup()

# setup the latent variables for LandBOSSE/ORBIT and FinanceSE
ard.cost.wisdem_wrap.ORBIT_setup_latents(prob, modeling_options)
# ard.cost.wisdem_wrap.LandBOSSE_setup_latents(prob, modeling_options)
ard.cost.wisdem_wrap.FinanceSE_setup_latents(prob, modeling_options)

# run the model
prob.run_model()

# derivs = prob.compute_totals()
# print(derivs)

# # visualize model
# om.n2(prob, "auld")

# collapse the test result data
test_data = {
    "AEP_val": float(prob.get_val("AEP_farm", units="GW*h")[0]),
    "CapEx_val": float(prob.get_val("tcc.tcc", units="MUSD")[0]),
    "BOS_val": float(prob.get_val("orbit.total_capex", units="MUSD")[0]),
    # "BOS_val": float(prob.get_val("landbosse.total_capex", units="MUSD")[0]),
    "OpEx_val": float(prob.get_val("opex.opex", units="MUSD/yr")[0]),
    "LCOE_val": float(prob.get_val("financese.lcoe", units="USD/MW/h")[0]),
    "area_tight": float(prob.get_val("landuse.area_tight", units="km**2")[0]),
    "coll_length": float(
        prob.get_val("optiwindnet_coll.total_length_cables", units="km")[0]
    ),
    "mooring_spacing": float(
        np.min(prob.get_val("mooring_constraint.mooring_spacing", units="km"))
    ),
    "turbine_spacing": float(
        np.min(prob.get_val("spacing_constraint.turbine_spacing", units="km"))
    ),

}

print("\n\nRESULTS:\n")
pp.pprint(test_data)
print("\n\n")

optimize = True
if optimize:
    # now set up an optimization driver

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"

    prob.model.add_design_var("spacing_primary", lower=3.0, upper=10.0)
    prob.model.add_design_var("spacing_secondary", lower=3.0, upper=10.0)
    prob.model.add_design_var("angle_orientation", lower=-180.0, upper=180.0)
    prob.model.add_design_var("angle_skew", lower=-75.0, upper=75.0)
    prob.model.add_design_var("phi_platform", lower=-30.0, upper=30.0)
    prob.model.add_constraint(
        "mooring_constraint.mooring_spacing", units="m", lower=50.0
    )
    prob.model.add_constraint(
        "spacing_constraint.turbine_spacing", units="m", lower=284.0 * 3.0
    )
    # prob.model.add_constraint("landuse.area_tight", units="km**2", lower=50.0)
    prob.model.add_objective("optiwindnet_coll.total_length_cables")

    # create a recorder
    recorder = om.SqliteRecorder("opt_results.sql")

    # add the recorder to the problem
    prob.add_recorder(recorder)
    # add the recorder to the driver
    prob.driver.add_recorder(recorder)

    # set up the problem
    prob.setup()

    # setup the latent variables for LandBOSSE/ORBIT and FinanceSE
    ard.cost.wisdem_wrap.ORBIT_setup_latents(prob, modeling_options)
    # ard.cost.wisdem_wrap.LandBOSSE_setup_latents(prob, modeling_options)
    ard.cost.wisdem_wrap.FinanceSE_setup_latents(prob, modeling_options)

    prob.run_model()
    # prob.check_totals(compact_print=True)
    # from openmdao.utils.assert_utils import assert_check_partials
    # data = prob.check_partials(out_stream=None)
    # # print(data)
    # try:
    #     assert_check_partials(data, atol=1.e-4, rtol=1.e-4)
    # except ValueError as err:
    #     print(str(err))
    # run the optimization
    prob.run_driver()
    prob.check_totals(compact_print=True, show_only_incorrect=True)

    # collapse the test result data
    test_data = {
        "AEP_val": float(prob.get_val("AEP_farm", units="GW*h")[0]),
        "CapEx_val": float(prob.get_val("tcc.tcc", units="MUSD")[0]),
        "BOS_val": float(prob.get_val("orbit.total_capex", units="MUSD")[0]),
        # "BOS_val": float(prob.get_val("landbosse.total_capex", units="MUSD")[0]),
        "OpEx_val": float(prob.get_val("opex.opex", units="MUSD/yr")[0]),
        "LCOE_val": float(prob.get_val("financese.lcoe", units="USD/MW/h")[0]),
        "area_tight": float(prob.get_val("landuse.area_tight", units="km**2")[0]),
        "coll_length": float(
            prob.get_val("optiwindnet_coll.total_length_cables", units="km")[0]
        ),
        "mooring_spacing": float(
            np.min(prob.get_val("mooring_constraint.mooring_spacing", units="km"))
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

optiwindnet.plotting.gplot(prob.model.optiwindnet_coll.graph)
for idx in range(modeling_options["farm"]["N_turbines"]):
    plt.plot(
        prob.get_val("mooring_design.x_anchors", units="m")[idx, :],
        prob.get_val("mooring_design.y_anchors", units="m")[idx, :],
        ".w",
    )
plt.show()


# RESULTS:

# {'AEP_val': 4818.0,
#  'BOS_val': 2127.5924853696597,
#  'CapEx_val': 768.4437570425,
#  'LCOE_val': 57.63858824842508,
#  'OpEx_val': 60.50000000000001,
#  'area_tight': 63.234304,
#  'coll_length': 47.761107521256534,
#  'mooring_spacing': 1.1208759839268934}


# RESULTS (opt):

# {'AEP_val': 4818.0,
#  'BOS_val': 2113.9686662769814,
#  'CapEx_val': 768.4437570425,
#  'LCOE_val': 57.42651136342073,
#  'OpEx_val': 60.50000000000001,
#  'area_tight': 11.614464,
#  'coll_length': 20.49594468696311,
#  'mooring_spacing': 0.0582385263254008,
#  'turbine_spacing': 0.8519999999999998}
