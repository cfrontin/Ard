import pprint as pp

import numpy as np
import matplotlib.pyplot as plt

import openmdao.api as om

import optiwindnet.plotting
from ard.utils.io import load_yaml
from ard.api import set_up_ard_model
from ard.viz.layout import plot_layout


def run_example():

    # load input
    input_dict = load_yaml("./inputs/ard_system.yaml")

    # set up system
    prob = set_up_ard_model(
        input_dict=input_dict,
        root_data_path="inputs",
    )

    prob.model.set_input_defaults(
        "x_turbines",
        input_dict["modeling_options"]["windIO_plant"]["wind_farm"]["layouts"][
            "coordinates"
        ]["x"],
        units="m",
    )
    prob.model.set_input_defaults(
        "y_turbines",
        input_dict["modeling_options"]["windIO_plant"]["wind_farm"]["layouts"][
            "coordinates"
        ]["y"],
        units="m",
    )

    if False:
        # visualize model
        om.n2(prob)

    # run the model
    prob.run_model()

    # collapse the test result data
    test_data = {
        "AEP_val": float(prob.get_val("AEP_farm", units="GW*h")[0]),
        "CapEx_val": float(prob.get_val("tcc.tcc", units="MUSD")[0]),
        "BOS_val": float(prob.get_val("orbit.total_capex", units="MUSD")[0]),
        "OpEx_val": float(prob.get_val("opex.opex", units="MUSD/yr")[0]),
        "LCOE_val": float(prob.get_val("financese.lcoe", units="USD/MW/h")[0]),
        "area_tight": float(prob.get_val("landuse.area_tight", units="km**2")[0]),
        "coll_length": float(
            prob.get_val("collection.total_length_cables", units="km")[0]
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

    optimize = True  # set to False to skip optimization

    if optimize:

        # run the optimization
        prob.run_driver()
        prob.cleanup()

        # collapse the test result data
        test_data = {
            "x_turbines": prob.get_val("x_turbines", units="km"),
            "y_turbines": prob.get_val("y_turbines", units="km"),
            # "spacing_primary": float(prob.get_val("spacing_primary")[0]),
            # "spacing_secondary": float(prob.get_val("spacing_secondary")[0]),
            # "angle_orientation": float(prob.get_val("angle_orientation")[0]),
            # "angle_skew": float(prob.get_val("angle_skew")[0]),
            "AEP_val": float(prob.get_val("AEP_farm", units="GW*h")[0]),
            "CapEx_val": float(prob.get_val("tcc.tcc", units="MUSD")[0]),
            "BOS_val": float(prob.get_val("orbit.total_capex", units="MUSD")[0]),
            "OpEx_val": float(prob.get_val("opex.opex", units="MUSD/yr")[0]),
            "LCOE_val": float(prob.get_val("financese.lcoe", units="USD/MW/h")[0]),
            "area_tight": float(prob.get_val("landuse.area_tight", units="km**2")[0]),
            "coll_length": float(
                prob.get_val("collection.total_length_cables", units="km")[0]
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

        # plot convergence
        ## read cases
        cr = om.CaseReader(
            prob.get_outputs_dir()
            / input_dict["analysis_options"]["recorder"]["filepath"]
        )

        # Extract the driver cases
        cases = cr.get_cases("driver")

        # Initialize lists to store iteration data
        iterations = []
        objective_values = []

        # Loop through the cases and extract iteration number and objective value
        for i, case in enumerate(cases):
            iterations.append(i)
            objective_values.append(
                case.get_objectives()[
                    input_dict["analysis_options"]["objective"]["name"]
                ]
            )

        # Plot the convergence
        plt.figure(figsize=(8, 6))
        plt.plot(
            iterations,
            np.array(objective_values)
            * input_dict["analysis_options"]["objective"]
            .get("options", {})
            .get("scaler", 1.0),
            marker="o",
            label=f"objective ({input_dict['analysis_options']['objective'].get('name')})",
        )
        plt.xlabel("Iteration number (-)")
        plt.ylabel(
            f"Objective value ({input_dict['analysis_options']['objective'].get('name')})"
        )
        plt.legend()
        plt.grid()
        plt.show()

    plot_layout(
        prob,
        input_dict=input_dict,
        show_image=True,
        include_cable_routing=False,
        include_mooring_system=True,
    )


if __name__ == "__main__":

    run_example()
