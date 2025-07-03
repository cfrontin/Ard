import pytest
import openmdao.api as om
from ard.utils.io import load_yaml
from pathlib import Path
from ard.api import set_up_ard_model, replace_key_value, set_up_system_recursive

class TestSetUpArdModel:
    def setup_method(self):
        
        input_dict_path = str(Path(__file__).parent.absolute() / "inputs" / "ard_system.yaml")

        self.prob = set_up_ard_model(input_dict=input_dict_path)

        # set up the working/design variables
        self.prob.set_val("spacing_primary", 7.0)
        self.prob.set_val("spacing_secondary", 7.0)
        self.prob.set_val("angle_orientation", 0.0)

        self.prob.set_val("optiwindnet_coll.x_substations", [100.0])
        self.prob.set_val("optiwindnet_coll.y_substations", [100.0])

    def test_onshore_default_system_run_model(self):

        self.prob.run_model()
