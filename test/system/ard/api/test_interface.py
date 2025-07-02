import pytest
import openmdao.api as om
from ard.utils.io import load_yaml
from pathlib import Path
from ard.api import set_up_ard_model, replace_key_value, set_up_system_recursive

class TestSetUpArdModel:
    def setup_method(self):
        
        input_dict = load_yaml(
            Path(__file__).parent / "inputs" / "ard_system.yaml"
        )
        
        self.prob = set_up_ard_model(input_dict=input_dict)

    def test_onshore_default_system_run_model(self):

        try:
            self.prob.run_model()
        except:
            assert False
