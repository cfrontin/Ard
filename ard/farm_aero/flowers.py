import os

import numpy as np
import pandas as pd

from flowers import FlowersModel

import ard.farm_aero.templates as templates


class FLOWERSAEP(templates.FarmAEPTemplate):
    """
    Component class for computing an AEP analysis using FLOWERS.

    A component class that evaluates a series of farm power and associated
    quantities using FLOWERS with a wind rose to make an AEP estimate. Inherits
    the interface from `templates.FarmAEPTemplate`.

    Options
    -------
    modeling_options : dict
        a modeling options dictionary (inherited via
        `templates.FarmAEPTemplate`)
    data_path: str
        not used for FLOWERS (inherited via `templates.FarmAEPTemplate`)

    Inputs
    ------
    x_turbines : np.ndarray
        a 1D numpy array indicating the x-dimension locations of the turbines,
        with length `N_turbines` (inherited via `templates.FarmAEPTemplate`)
    y_turbines : np.ndarray
        a 1D numpy array indicating the y-dimension locations of the turbines,
        with length `N_turbines` (inherited via `templates.FarmAEPTemplate`)
    yaw_turbines : np.ndarray
        not used for FLOWERS (inherited via `templates.FarmAEPTemplate`)

    Outputs
    -------
    AEP_farm : float
        the AEP of the farm given by the analysis (inherited from
        `templates.FarmAEPTemplate`)
    power_farm : np.ndarray
        an array of the farm power for each of the wind conditions that have
        been queried (inherited from `templates.FarmAEPTemplate`)
    # power_turbines : np.ndarray
    #     an array of the farm power for each of the turbines in the farm across
    #     all of the conditions that have been queried on the wind rose
    #     (`N_turbines`, `N_wind_conditions`) (inherited from
    #     `templates.FarmAEPTemplate`)
    # thrust_turbines : np.ndarray
    #     an array of the wind turbine thrust for each of the turbines in the farm
    #     across all of the conditions that have been queried on the wind rose
    #     (`N_turbines`, `N_wind_conditions`) (inherited from
    #     `templates.FarmAEPTemplate`)
    """

    def initialize(self):
        super().initialize()  # run super class script first!

    def setup(self):

        # override template to allow for input omission

        # load modeling options
        self.modeling_options = self.options["modeling_options"]
        self.windIO = self.modeling_options["windIO_plant"]
        self.N_turbines = self.modeling_options["layout"]["N_turbines"]

        # set up inputs and outputs for farm layout
        self.add_input("x_turbines", np.zeros((self.N_turbines,)), units="m")
        self.add_input("y_turbines", np.zeros((self.N_turbines,)), units="m")
        self.add_output("AEP_farm", 0.0, units="W*h")

        # grab the windrose from the windIO data
        windrose_floris = templates.create_windresource_from_windIO(
            self.windIO, resource_type="probability",
        )
        # extract to a dataframe
        self.wind_data = pd.DataFrame({
            "wd": windrose_floris.wd_flat,
            "ws": windrose_floris.ws_flat,
            "freq_val": windrose_floris.freq_table_flat,
        })

    def setup_partials(self):
        super().setup_partials()

    def compute(self, inputs, outputs):

        # extract the key inputs
        layout_x = inputs["x_turbines"]
        layout_y = inputs["y_turbines"]
        num_terms = self.modeling_options["flowers"]["num_terms"]
        k_wake_expansion = self.modeling_options["flowers"]["k"]
        # turbine_type = self.modeling_options["flowers"]["turbine"]  # TODO: handle better
        turbine_type = {
            "D": self.windIO["wind_farm"]["turbine"]["rotor_diameter"],
            "U": self.windIO["wind_farm"]["turbine"]["performance"].get("cutout_wind_speed", 25.0),
            "ct": self.windIO["wind_farm"]["turbine"]["performance"]["Ct_curve"]["Ct_values"],
            "u_ct": self.windIO["wind_farm"]["turbine"]["performance"]["Ct_curve"]["Ct_wind_speeds"],
        }
        if "Cp_curve" not in self.windIO["wind_farm"]["turbine"]["performance"]:
            raise NotImplementedError("power to coefficient tranform not programmed yet...")
        turbine_type["cp"] = self.windIO["wind_farm"]["turbine"]["performance"]["Cp_curve"]["Cp_values"]
        turbine_type["u_cp"] = self.windIO["wind_farm"]["turbine"]["performance"]["Cp_curve"]["Cp_wind_speeds"]

        # create the flowers model
        self.flowers_model = FlowersModel(
            self.wind_data,
            layout_x,
            layout_y,
            num_terms,
            k_wake_expansion,
            turbine_type,
        )

        # FLOWERS computes the powers
        outputs["AEP_farm"] = self.flowers_model.calculate_aep(self)
        # outputs["power_farm"] = FLOWERSFarmComponent.get_power_farm(self)
        # outputs["power_turbines"] = FLOWERSFarmComponent.get_power_turbines(self)
        # outputs["thrust_turbines"] = FLOWERSFarmComponent.get_thrust_turbines(self)

    def compute_partials(self, inputs, partials):
        raise NotImplementedError("FLOWERS PARTIALS ARE NOT IMPLEMENTED YET!")

    def setup_partials(self):
        super().setup_partials()
