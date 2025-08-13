from pathlib import Path

from pprint import pprint

import matplotlib.pyplot as plt

import windIO

import floris

import ard.farm_aero.templates as ard_aero
from ard.utils.io import load_yaml
from ard.api import set_up_ard_model
import ard.wind_query as wq

path_windIO = Path(__file__).parent / "inputs" / "windio.yaml"

windIO.validate(input=path_windIO, schema_type="plant/wind_energy_system")
windIOdict = windIO.load_yaml(path_windIO)

# load input
path_ard_system = Path(__file__).parent / "inputs" / "ard_system.yaml"
input_dict = load_yaml(path_ard_system)

# build the Ard model
prob = set_up_ard_model(input_dict=input_dict, root_data_path="inputs")

# run the model
prob.run_model()

# print the AEP result
print(f"AEP result: {prob.get_val('AEP_farm', units='GW*h')}")
