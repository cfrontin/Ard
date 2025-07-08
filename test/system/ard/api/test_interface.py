import pytest
from pathlib import Path
from ard.api import set_up_ard_model, replace_key_value, set_up_system_recursive
import numpy as np


class TestSetUpArdModel:
    def setup_method(self):

        input_dict_path = str(
            Path(__file__).parent.absolute() / "inputs" / "ard_system.yaml"
        )

        self.prob = set_up_ard_model(input_dict=input_dict_path)

        self.prob.run_model()

    def test_onshore_default_system_aep(self, subtests):

        with subtests.test("AEP_farm"):
            assert self.prob.get_val("AEP_farm", units="GW*h")[0] == pytest.approx(
                340.823649
            )
        with subtests.test("tcc.tcc"):
            assert self.prob.get_val("tcc.tcc", units="MUSD")[0] == pytest.approx(
                109.525
            )
        with subtests.test("BOS capex (landbosse.total_capex)"):
            assert self.prob.get_val("landbosse.total_capex", units="MUSD")[
                0
            ] == pytest.approx(41.57835529)
        with subtests.test("opex.opex"):
            assert self.prob.get_val("opex.opex", units="MUSD/yr")[0] == pytest.approx(
                3.707
            )
        with subtests.test("financese.lcoe"):
            assert self.prob.get_val("financese.lcoe", units="USD/MW/h")[
                0
            ] == pytest.approx(44.127664)
