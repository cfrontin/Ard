from pathlib import Path

import pprint as pp

import numpy as np
import matplotlib.pyplot as plt

import optiwindnet.plotting
from ard.utils.io import load_yaml
from ard.glue.prototype_jt import set_up_system_recursive
import openmdao.api as om


def run_example():

    # load input
    system_spec = load_yaml("./inputs/ard/ard_system.yaml")

    # set up system
    prob = set_up_system_recursive(
        system_spec["plant"],
        system_name="top_level",
        modeling_options=system_spec["modeling_options"],
        analysis_options=system_spec["analysis_options"]
    )

    # set up the working/design variables
    prob.set_val("spacing_primary", 7.0)
    prob.set_val("spacing_secondary", 7.0)
    prob.set_val("angle_orientation", 0.0)

    prob.set_val("optiwindnet_coll.x_substations", [100.0])
    prob.set_val("optiwindnet_coll.y_substations", [100.0])

    # run the model
    prob.run_model()

    # Visualize model
    # om.n2(prob)

    # collapse the test result data
    test_data = {
        "AEP_val": float(prob.get_val("AEP_farm", units="GW*h")[0]),
        "CapEx_val": float(prob.get_val("tcc.tcc", units="MUSD")[0]),
        "BOS_val": float(prob.get_val("landbosse.total_capex", units="MUSD")[0]),
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

    # RESULTS current:

    # {'AEP_val': 738.0300000000001,
    #  'BOS_val': 43.11521020118212,
    #  'CapEx_val': 0.0,
    #  'LCOE_val': 20.534416981814644,
    #  'OpEx_val': 0.0,
    #  'area_tight': 13.2496,
    #  'coll_length': 21.89865877023397}

    # RESULTS original:

    # {'AEP_val': 738.0300000000001,
    #  'BOS_val': 43.11521020118212,
    #  'CapEx_val': 109.52499999999999,
    #  'LCOE_val': 20.534416981814644,
    #  'OpEx_val': 3.7070000000000007,
    #  'area_tight': 13.2496,
    #  'coll_length': 21.89865877023397,
    #  'turbine_spacing': 0.91}

    # RESULTS original (opt):

    # {'AEP_val': 738.0300000000001,
    #  'BOS_val': 42.79961619700176,
    #  'CapEx_val': 109.52499999999999,
    #  'LCOE_val': 20.50234572412386,
    #  'OpEx_val': 3.7070000000000007,
    #  'area_tight': 11.561725802729931,
    #  'coll_length': 20.49436456231002,
    #  'turbine_spacing': 851.9998683944751}


    optimize = True  # set to False to skip optimization

    if optimize:
        # now set up an optimization driver
        
        # run the optimization
        prob.run_driver()
        import pdb; pdb.set_trace()
        prob.cleanup()
        cr = om.CaseReader(prob.get_outputs_dir() / "cases.sql")
        driver_cases = cr.list_cases('driver')
        # collapse the test result data
        test_data = {
            "AEP_val": float(prob.get_val("AEP_farm", units="GW*h")[0]),
            "CapEx_val": float(prob.get_val("tcc.tcc", units="MUSD")[0]),
            "BOS_val": float(prob.get_val("landbosse.total_capex", units="MUSD")[0]),
            # "BOS_val": float(prob.get_val("landbosse.total_capex", units="MUSD")[0]),
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

    optiwindnet.plotting.gplot(prob.model.optiwindnet_coll.graph)

    plt.show()


if __name__ == "__main__":

    run_example()
