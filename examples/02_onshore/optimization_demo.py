from pathlib import Path

import pprint as pp

import numpy as np
import matplotlib.pyplot as plt

import optiwindnet.plotting
from ard.utils.io import load_yaml
from ard.glue.prototype_jt import set_up_system_recursive
from ard.cost.wisdem_wrap import LandBOSSE_setup_latents, FinanceSE_setup_latents
import openmdao.api as om

def run_example():
        
    # load input    
    system_spec = load_yaml("./inputs/ard/ard_system.yaml")

    # set up system
    prob = set_up_system_recursive(system_spec["plant"])

    prob.setup()

    # Visualize model
    # om.n2(prob)

    LandBOSSE_setup_latents(prob, system_spec["modeling_options"])
    FinanceSE_setup_latents(prob, system_spec["modeling_options"])

    # set up the working/design variables
    prob.set_val("top_level.spacing_primary", 7.0)
    prob.set_val("top_level.spacing_secondary", 7.0)
    prob.set_val("top_level.angle_orientation", 0.0)

    prob.set_val("top_level.optiwindnet_coll.x_substations", [100.0])
    prob.set_val("top_level.optiwindnet_coll.y_substations", [100.0])

    # run the model
    prob.run_model()

    # # collapse the test result data
    # test_data = {
    #     "AEP_val": float(prob.get_val("AEP_farm", units="GW*h")[0]),
    #     "CapEx_val": float(prob.get_val("tcc.tcc", units="MUSD")[0]),
    #     "BOS_val": float(prob.get_val("landbosse.total_capex", units="MUSD")[0]),
    #     # "BOS_val": float(prob.get_val("landbosse.total_capex", units="MUSD")[0]),
    #     "OpEx_val": float(prob.get_val("opex.opex", units="MUSD/yr")[0]),
    #     "LCOE_val": float(prob.get_val("financese.lcoe", units="USD/MW/h")[0]),
    #     "area_tight": float(prob.get_val("landuse.area_tight", units="km**2")[0]),
    #     "coll_length": float(
    #         prob.get_val("optiwindnet_coll.total_length_cables", units="km")[0]
    #     ),
    # }

    # print("\n\nRESULTS:\n")
    # pp.pprint(test_data)
    # print("\n\n")

    # optimize = True  # set to False to skip optimization

    # if optimize:
    #     # now set up an optimization driver

    #     prob.driver = om.ScipyOptimizeDriver()
    #     prob.driver.options["optimizer"] = "SLSQP"

    #     prob.model.add_design_var("spacing_primary", lower=3.0, upper=10.0)
    #     prob.model.add_design_var("spacing_secondary", lower=3.0, upper=10.0)
    #     prob.model.add_design_var("angle_orientation", lower=-180.0, upper=180.0)
    #     prob.model.add_design_var("angle_skew", lower=-75.0, upper=75.0)
    #     prob.model.add_constraint(
    #         "spacing_constraint.turbine_spacing", units="m", lower=284.0 * 3.0
    #     )
    #     # prob.model.add_constraint("landuse.area_tight", units="km**2", lower=50.0)
    #     prob.model.add_objective("optiwindnet_coll.total_length_cables")

    #     # create a recorder
    #     recorder = om.SqliteRecorder("opt_results.sql")

    #     # add the recorder to the problem
    #     prob.add_recorder(recorder)
    #     # add the recorder to the driver
    #     prob.driver.add_recorder(recorder)

    #     # set up the problem
    #     prob.setup()

    #     # ard.cost.wisdem_wrap.LandBOSSE_setup_latents(prob, modeling_options)
    #     # ard.cost.wisdem_wrap.FinanceSE_setup_latents(prob, modeling_options)

    #     # set up the working/design variables initial conditions
    #     prob.set_val("spacing_primary", 7.0)
    #     prob.set_val("spacing_secondary", 7.0)
    #     prob.set_val("angle_orientation", 0.0)
    #     prob.set_val("angle_skew", 0.0)

    #     prob.set_val("optiwindnet_coll.x_substations", [100.0])
    #     prob.set_val("optiwindnet_coll.y_substations", [100.0])

    #     # run the optimization
    #     prob.run_driver()

    #     # collapse the test result data
    #     test_data = {
    #         "AEP_val": float(prob.get_val("AEP_farm", units="GW*h")[0]),
    #         "CapEx_val": float(prob.get_val("tcc.tcc", units="MUSD")[0]),
    #         "BOS_val": float(prob.get_val("landbosse.total_capex", units="MUSD")[0]),
    #         # "BOS_val": float(prob.get_val("landbosse.total_capex", units="MUSD")[0]),
    #         "OpEx_val": float(prob.get_val("opex.opex", units="MUSD/yr")[0]),
    #         "LCOE_val": float(prob.get_val("financese.lcoe", units="USD/MW/h")[0]),
    #         "area_tight": float(prob.get_val("landuse.area_tight", units="km**2")[0]),
    #         "coll_length": float(
    #             prob.get_val("optiwindnet_coll.total_length_cables", units="km")[0]
    #         ),
    #         "turbine_spacing": float(
    #             np.min(prob.get_val("spacing_constraint.turbine_spacing", units="km"))
    #         ),
    #     }

    #     # clean up the recorder
    #     prob.cleanup()

    #     # print the results
    #     print("\n\nRESULTS (opt):\n")
    #     pp.pprint(test_data)
    #     print("\n\n")

    # optiwindnet.plotting.gplot(prob.model.optiwindnet_coll.graph)

    # plt.show()



if __name__ == "__main__":

    run_example()
