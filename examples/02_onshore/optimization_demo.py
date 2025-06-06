from pathlib import Path

import pprint as pp

import numpy as np
import matplotlib.pyplot as plt

import floris
import openmdao.api as om

import optiwindnet.plotting
import ard
import ard.layout.spacing
import ard.layout.gridfarm
import ard.farm_aero
import ard.utils.io
from ard.cost.approximate_turbine_spacing import LandBOSSEWithSpacingApproximations


def run_example():
        
    

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
