import pprint as pp

import numpy as np
import matplotlib.pyplot as plt

import optiwindnet.plotting
from ard.utils.io import load_yaml
from ard.api import set_up_ard_model
import openmdao.api as om


# get plot limits based on the farm boundaries
def get_limits(windIOdict, lim_buffer=0.05):
    x_lim = [
        np.min(windIOdict["site"]["boundaries"]["polygons"][0]["x"])
        - lim_buffer * np.ptp(windIOdict["site"]["boundaries"]["polygons"][0]["x"]),
        np.max(windIOdict["site"]["boundaries"]["polygons"][0]["x"])
        + lim_buffer * np.ptp(windIOdict["site"]["boundaries"]["polygons"][0]["x"]),
    ]
    y_lim = [
        np.min(windIOdict["site"]["boundaries"]["polygons"][0]["y"])
        - lim_buffer * np.ptp(windIOdict["site"]["boundaries"]["polygons"][0]["y"]),
        np.max(windIOdict["site"]["boundaries"]["polygons"][0]["y"])
        + lim_buffer * np.ptp(windIOdict["site"]["boundaries"]["polygons"][0]["y"]),
    ]
    return x_lim, y_lim

def run_example():

    # load input
    input_dict = load_yaml("./inputs/ard_system.yaml")
    
    # set up system
    prob = set_up_ard_model(input_dict=input_dict, root_data_path="inputs")

    if False:
        # visualize model
        om.n2(prob)

    # run the model
    prob.run_model()

    # collapse the test result data
    test_data = {
        "AEP_val": float(prob.get_val("AEP_farm", units="GW*h")[0]),
        "CapEx_val": float(prob.get_val("tcc.tcc", units="MUSD")[0]),
        "BOS_val": float(prob.get_val("total_capex", units="MUSD")[0]),
        "OpEx_val": float(prob.get_val("opex.opex", units="MUSD/yr")[0]),
        "LCOE_val": float(prob.get_val("financese.lcoe", units="USD/MW/h")[0]),
        "area_tight": float(prob.get_val("landuse.area_tight", units="km**2")[0]),
        "coll_length": float(
            prob.get_val("optiwindnet_coll.total_length_cables", units="km")[0]
        ),
        "turbine_spacing": float(
            np.min(prob.get_val("spacing_constraint.turbine_spacing", units="km"))
        ),
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
            "BOS_val": float(prob.get_val("total_capex", units="MUSD")[0]),
            "OpEx_val": float(prob.get_val("opex.opex", units="MUSD/yr")[0]),
            "LCOE_val": float(prob.get_val("financese.lcoe", units="USD/MW/h")[0]),
            "area_tight": float(prob.get_val("landuse.area_tight", units="km**2")[0]),
            "coll_length": float(
                prob.get_val("optiwindnet_coll.total_length_cables", units="km")[0]
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


    # get the turbine locations to plot
    x_turbines = prob.get_val("x_turbines", units="km")
    y_turbines = prob.get_val("y_turbines", units="km")

    # make a plot
    fig, ax = plt.subplots()
    windIO_dict = input_dict["modeling_options"]["windIO_plant"]
    ax.fill(
        windIO_dict["site"]["boundaries"]["polygons"][0]["x"],
        windIO_dict["site"]["boundaries"]["polygons"][0]["y"],
        linestyle="--",
        alpha=0.5,
        fill=False,
    )
    ax.plot(x_turbines, y_turbines, "ok")
    x_lim, y_lim = get_limits(windIO_dict)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    plt.show()


if __name__ == "__main__":

    run_example()
