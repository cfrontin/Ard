from pathlib import Path

import numpy as np
import pandas as pd

import flowers
import ard

import pytest


class TestFlowersModel:
    def setup_method(self):

        self.wind_rose = pd.read_csv(
            Path(flowers.__file__).parents[1]
            / "examples"
            / "flowers"
            / "data"
            / "HKW_wind_rose.csv"
        )

        self.turbine_dict = {
            "D": 130.0,
            # "U": None,
            # "ct": None,
            # "u_ct": None,
            # "cp": None,
            # "u_cp": None,
        }

        self.layout_x, self.layout_y = [
            (self.turbine_dict["D"] * v).flat
            for v in np.meshgrid(
                5 * np.arange(-3, 3.001, 1),
                5 * np.arange(-3, 3.001, 1),
            )
        ]

        self.flowers_model = flowers.FlowersModel(
            self.wind_rose,
            layout_x=self.layout_x,
            layout_y=self.layout_y,
            num_terms=50,
            k=0.05,
            turbine="nrel_5MW",
        )

    def test_reference_NREL5MW(self, subtests):

        with subtests.test("initialization of internal values: k"):
            assert self.flowers_model.k == 0.05
        with subtests.test("initialization of internal values: num_modes"):
            assert self.flowers_model.get_num_modes() <= 50

        # compute the AEP and compare it to a pyrite-standard value
        with subtests.test("pyrite AEP value"):
            AEP_ref = 904.2681563494169  # GWh
            AEP_calculated = self.flowers_model.calculate_aep() / 1e9  # GWh
            assert AEP_calculated == AEP_ref

    def test_reinit_reference_NREL5MW(self, subtests):

        rotate_angle = np.radians(15.0)  # angle to rotate layout
        layout_x = (
            np.cos(rotate_angle) * self.layout_x - np.sin(rotate_angle) * self.layout_y
        )
        layout_y = (
            np.sin(rotate_angle) * self.layout_x + np.cos(rotate_angle) * self.layout_y
        )

        self.flowers_model.reinitialize(
            self.wind_rose,
            layout_x=layout_x,
            layout_y=layout_y,
            num_terms=25,
            k=0.075,
        )

        with subtests.test("initialization of internal values: k"):
            assert self.flowers_model.k == 0.075
        with subtests.test("initialization of internal values: num_modes"):
            assert self.flowers_model.get_num_modes() <= 25

        # compute the AEP and compare it to a pyrite-standard value
        with subtests.test("pyrite AEP value"):
            AEP_ref = 976.2494933660288  # GWh
            AEP_calculated = self.flowers_model.calculate_aep() / 1e9  # GWh
            assert AEP_ref == AEP_calculated

    def test_raise_Error_reference_IEA22MW(self, subtests):
        with pytest.raises(NotImplementedError):
            busted_flowers_model = flowers.FlowersModel(
                self.wind_rose,
                layout_x=self.layout_x,
                layout_y=self.layout_y,
                num_terms=50,
                k=0.05,
                turbine="iea_22MW",
            )


class TestCustomFlowersModel:
    def setup_method(self):
        self.IEA_3p4_data = pd.read_csv(
            Path(ard.__file__).parents[1]
            / "test"
            / "ard"
            / "data"
            / "power_thrust_table_ccblade_IEA-3p4-130-RWT.csv",
            names=["u", "cp", "ct"],
        )

        self.wind_rose = pd.read_csv(
            Path(flowers.__file__).parents[1]
            / "examples"
            / "flowers"
            / "data"
            / "HKW_wind_rose.csv"
        )

        self.turbine_dict = {
            "D": 130.0,
            "U": 25.0,
            "ct": self.IEA_3p4_data["ct"].to_numpy(),
            "u_ct": self.IEA_3p4_data["u"].to_numpy(),
            "cp": self.IEA_3p4_data["cp"].to_numpy(),
            "u_cp": self.IEA_3p4_data["u"].to_numpy(),
        }

        self.layout_x, self.layout_y = [
            (self.turbine_dict["D"] * v).flat
            for v in np.meshgrid(
                5 * np.arange(-3, 3.001, 1),
                5 * np.arange(-3, 3.001, 1),
            )
        ]

        self.flowers_model = flowers.FlowersModel(
            self.wind_rose,
            layout_x=self.layout_x,
            layout_y=self.layout_y,
            num_terms=50,
            k=0.05,
            turbine=self.turbine_dict,
        )

    def test_custom_IEA3p4(self, subtests):

        # get the computed estimate
        AEP_calculated = self.flowers_model.calculate_aep() / 1e9  # GWh

        # compute the AEP and compare it to a pyrite-standard value
        with subtests.test("matching pyrite AEP value"):
            AEP_ref = 922.0285639699603  # GWh
            assert AEP_ref == AEP_calculated

        # compute the capacity factor and make sure it's plausible
        with subtests.test("plausible capacity factor"):
            rated_production = 3.4 * 8760 * len(self.layout_x) / 1000.0  # to GWh
            capacity_factor = AEP_calculated / rated_production
            is_plausible_cf = (capacity_factor > 0.25) & (capacity_factor < 0.75)
            assert is_plausible_cf

        AEP_calculated2, dAEP_calculated = [
            v / 1.0e9 for v in self.flowers_model.calculate_aep(gradient=True)
        ]

        # compute the AEP and compare it to a pyrite-standard value
        with subtests.test("matching pyrite AEP value on derivative calc"):
            AEP_ref = 922.0285639699603  # GWh
            assert AEP_calculated2 == AEP_ref

        Nt, Nd = dAEP_calculated.shape
        eps_val = 1.0e-6

        print(f"Nd, Nt: {Nd}, {Nt}")

        # break out and copy the layout variables
        layout_x_orig, layout_y_orig = self.flowers_model.get_layout()
        layout_x_orig = layout_x_orig.copy()
        layout_y_orig = layout_y_orig.copy()

        for i_t in range(Nt):
            with subtests.test(f"match derivative on turbine {i_t}"):
                for i_d in range(Nd):
                    layout_x = layout_x_orig.copy()
                    layout_y = layout_y_orig.copy()
                    if i_d:
                        layout_y[i_t] += eps_val * np.mean(np.abs(layout_y_orig))
                    else:
                        layout_x[i_t] += eps_val * np.mean(np.abs(layout_x_orig))
                    self.flowers_model.reinitialize(
                        layout_x=layout_x, layout_y=layout_y
                    )
                    AEP_plus = self.flowers_model.calculate_aep() / 1e9
                    dAEP = AEP_plus - AEP_calculated
                    dXY = (
                        (layout_y[i_t] - layout_y_orig[i_t])
                        if i_d
                        else (layout_x[i_t] - layout_x_orig[i_t])
                    )
                    dAEPdXY = dAEP / dXY

                    assert np.isclose(
                        dAEP_calculated[i_t, i_d],
                        dAEPdXY,
                        rtol=1.0e-4,
                        atol=1.0e-4,
                    )
