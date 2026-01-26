from pathlib import Path
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import openmdao.api as om

import floris
import flowers
import ard
import ard.wind_query as wq
import ard.layout.gridfarm as gridfarm
import ard.farm_aero.flowers as farmaero_flowers
import ard.farm_aero.floris as farmaero_floris
import ard.utils.io


class TestFLOWERSIndepMatch:

    def setup_method(self):

        self.path_csv_HKW = (
            Path(flowers.__file__).parents[1]
            / "examples"
            / "flowers"
            / "data"
            / "HKW_wind_rose.csv"
        )
        self.df_HKW = pd.read_csv(
            self.path_csv_HKW,
        )
        self.df_HKW["freq_val"] = np.maximum(
            self.df_HKW["freq_val"], 1.0e-6
        )  # prevent divide by zero

        N_turb = 256
        seed_val = int.from_bytes(b"0xAR")  # lock the seed for consistent results

        boundaries = [
            (-5000.0, 0.0),
            (0.0, 5000.0),
            (5000.0, 0.0),
            (0.0, -5000.0),
        ]  # a big diamond grid

        # generate a random layout
        xx, yy = self.layout = flowers.tools.random_layout(
            boundaries, N_turb, seed_val=seed_val
        )

        # grab the turbine_yaml and put it into a barebones windIO
        turbine_yaml = ard.utils.io.load_yaml(
            Path(__file__).parents[4]
            / "examples"
            / "ard"
            / "data"
            / "windIO-plant_turbine_IEA-3.4MW-130m-RWT.yaml"
        )
        df_pivot = self.df_HKW.pivot(index="wd", columns="ws", values="freq_val")
        self.windIO_plant = {
            "wind_farm": {
                "turbine": turbine_yaml,
            },
            "site": {
                "energy_resource": {
                    "wind_resource": {
                        "wind_direction": df_pivot.index,
                        "wind_speed": df_pivot.columns,
                        "probability": {
                            "data": df_pivot.values,
                        },
                        "turbulence_intensity": {
                            "data": 0.06 * np.ones_like(df_pivot.values),
                        },
                    },
                },
            },
        }

        # roll up the data into a modeling_options
        self.modeling_options = {
            "windIO_plant": self.windIO_plant,
            "layout": {
                "N_turbines": N_turb,
            },
            "flowers": {
                "num_terms": 0,
                "k": 0.05,
            },
        }

        # use the floris turbine as an intermediary to cover off windIO variants
        self.floris_turbine = farmaero_floris.create_FLORIS_turbine_from_windIO(
            self.windIO_plant
        )

        # get the data that raw FLOWERS needs
        rho_density_air = self.floris_turbine["power_thrust_table"][
            "ref_air_density"
        ]  # kg/m^3
        area_rotor = np.pi / 4 * self.floris_turbine["rotor_diameter"] ** 2  # m^2
        V_table = np.array(self.floris_turbine["power_thrust_table"]["wind_speed"])
        P_table = 1.0e3 * np.array(self.floris_turbine["power_thrust_table"]["power"])
        CT_table = np.array(
            self.floris_turbine["power_thrust_table"]["thrust_coefficient"]
        )
        CP_table = np.where(
            V_table == 0.0,
            0.0,
            P_table / (0.5 * rho_density_air * area_rotor * V_table**3),
        )

        # extract the turbine from the windIO for FLOWERS
        turbine_type = {
            "D": self.windIO_plant["wind_farm"]["turbine"]["rotor_diameter"],
            "U": self.windIO_plant["wind_farm"]["turbine"]["performance"].get(
                "cutout_wind_speed", 25.0
            ),
            "ct": CT_table,
            "u_ct": V_table,
            "cp": CP_table,
            "u_cp": V_table,
        }

        # create a flowers model
        self.fm = flowers.FlowersModel(
            wind_rose=self.df_HKW,
            layout_x=xx,
            layout_y=yy,
            num_terms=self.modeling_options["flowers"]["num_terms"],
            k=self.modeling_options["flowers"]["k"],
            turbine=turbine_type,
        )

        # create a problem instance
        prob = self.prob = om.Problem()
        model = prob.model
        model.add_subsystem(
            "flowers",
            ard.farm_aero.flowers.FLOWERSAEP(
                modeling_options=self.modeling_options,
            ),
        )
        prob.setup()
        prob.set_val("flowers.x_turbines", xx, units="m")
        prob.set_val("flowers.y_turbines", yy, units="m")

    def test_standalone_match(self, subtests):

        # run the standalone flowers as a reference value
        AEP_standalone_flowers = self.fm.calculate_aep() / 1.0e9

        # run Ard flowers to make sure it matches
        self.prob.run_model()
        AEP_ard_flowers = self.prob.get_val("flowers.AEP_farm", units="GW*h")[0]

        # make sure standalone FLOWERS and Ard FLOWERS match
        assert np.isclose(AEP_ard_flowers, AEP_standalone_flowers, rtol=0.005)
        # loose tolerance for roundoff in wind rose... not sure where it comes from


# class TestFLOWERSAEP:
#
#     def setup_method(self):
#
#         # create the farm layout specification
#         farm_spec = {}
#         farm_spec["xD_farm"], farm_spec["yD_farm"] = [
#             5 * v.flatten()
#             for v in np.meshgrid(np.linspace(-2, 2, 2), np.linspace(-2, 2, 2))
#         ]
#
#         # set up the modeling options
#         path_turbine = (
#             Path(ard.__file__).parents[1]
#             / "examples"
#             / "ard"
#             / "data"
#             / "windIO-plant_turbine_IEA-3.4MW-130m-RWT.yaml"
#         )
#         with open(path_turbine) as f_yaml:
#             data_turbine_yaml = yaml.safe_load(f_yaml)
#         # set up the modeling options
#         path_wind_resource = (
#             Path(ard.__file__).parents[1]
#             / "examples"
#             / "ard"
#             / "data"
#             / "windIO-plant_wind-resource_wrg-example.yaml"
#         )
#         with open(path_wind_resource) as f_yaml:
#             data_wind_resource_yaml = yaml.safe_load(f_yaml)
#         modeling_options = self.modeling_options = {
#             "windIO_plant": {
#                 "wind_farm": {
#                     "name": "unit test farm",
#                     "turbine": data_turbine_yaml,
#                     "layouts": {
#                         "coordinates": {
#                             "x": farm_spec["xD_farm"]
#                             * data_turbine_yaml["rotor_diameter"],
#                             "y": farm_spec["yD_farm"]
#                             * data_turbine_yaml["rotor_diameter"],
#                         }
#                     },
#                 },
#                 "site": {
#                     "energy_resource": {
#                         "wind_resource": data_wind_resource_yaml,
#                     },
#                 },
#             },
#             "wind_rose": {
#                 "windrose_resample": {
#                     "wd_step": 2.5,
#                     "ws_step": 1.0,
#                 },
#             },
#             "layout": {
#                 "N_turbines": len(farm_spec["xD_farm"]),
#                 "spacing_primary": 7.0,
#                 "spacing_secondary": 5.0,
#                 "angle_orientation": 15.0,
#                 "angle_skew": 10.0,
#             },
#             "aero": {
#                 "return_turbine_output": True,
#             },
#             # "floris": {
#             #     "peak_shaving_fraction": 0.4,
#             #     "peak_shaving_TI_threshold": 0.0,
#             # },
#             "flowers": {
#                 "num_terms": 0,
#                 "k": 0.05,
#             },
#         }
#
#         # create the OpenMDAO model
#         model = om.Group()
#         self.gf = model.add_subsystem(
#             "gridfarm",
#             gridfarm.GridFarmLayout(
#                 modeling_options=self.modeling_options,
#             ),
#             promotes=["*"],
#         )
#         self.FLOWERS = model.add_subsystem(
#             "batchFLOWERS",
#             farmaero_flowers.FLOWERSAEP(
#                 modeling_options=modeling_options,
#             ),
#             promotes=["x_turbines", "y_turbines"],
#         )
#         self.FLORIS = model.add_subsystem(
#             "batchFLORIS",
#             farmaero_floris.FLORISAEP(
#                 modeling_options=modeling_options,
#                 case_title="FLOWERS_test",
#             ),
#             promotes=["x_turbines", "y_turbines"],
#         )
#
#         self.prob = om.Problem(model)
#         self.prob.setup()
#
#         self.prob.set_val(
#             "x_turbines",
#             modeling_options["windIO_plant"]["wind_farm"]["layouts"]["coordinates"][
#                 "x"
#             ],
#             units="m",
#         )
#         self.prob.set_val(
#             "y_turbines",
#             modeling_options["windIO_plant"]["wind_farm"]["layouts"]["coordinates"][
#                 "y"
#             ],
#             units="m",
#         )
#
#     def test_dummy(self):
#
#         self.prob.run_model()
#
#         angle_orientation_vec = np.arange(0.0, 360.0, 2.5)
#         AEP_flowers_vec = np.zeros_like(angle_orientation_vec)
#         AEP_floris_vec = np.zeros_like(angle_orientation_vec)
#
#         print(f"angle_orientation_vec shape: {angle_orientation_vec.shape}")
#         print(f"AEP_flowers_vec shape: {AEP_flowers_vec.shape}")
#         print(f"AEP_floris_vec shape: {AEP_floris_vec.shape}")
#
#         for idx, angle_orientation in enumerate(angle_orientation_vec):
#
#             self.prob.set_val("angle_orientation", angle_orientation)
#             self.prob.run_model()
#
#             AEP_flowers = float(
#                 self.prob.get_val("batchFLOWERS.AEP_farm", units="GW*h")[0]
#             )
#             AEP_floris = float(
#                 self.prob.get_val("batchFLORIS.AEP_farm", units="GW*h")[0]
#             )
#
#             AEP_flowers_vec[idx] = AEP_flowers
#             AEP_floris_vec[idx] = AEP_floris
#
#         plt.plot(angle_orientation_vec, AEP_flowers_vec, label="flowers")
#         plt.plot(angle_orientation_vec, AEP_floris_vec, label="floris")
#         plt.xticks(np.arange(360.0, 90.0))
#         plt.legend()
#         plt.show()
#
#         assert False
