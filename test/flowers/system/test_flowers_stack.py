import os
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

import flowers
from flowers.flowers_model import FlowersModel
import flowers.tools
import flowers.visualization

import pytest


def test_flowers_stack_nrel_5MW():
    N_turb = 256
    seed_val = int.from_bytes(b'0xCF')
    show_plots = False

    boundaries = [
        (-5000.0,     0.0),
        (    0.0,  5000.0),
        ( 5000.0,     0.0),
        (    0.0, -5000.0),
    ]

    # change the cwd
    CWD = os.getcwd()
    path_wd = Path(__file__).parent
    os.chdir(path_wd)

    # load and show the wind rose
    df_wr = flowers.tools.load_wind_rose(4)
    flowers.visualization.plot_wind_rose(df_wr)
    plt.show() if show_plots else plt.close('all')

    # change the cwd back
    os.chdir(CWD)

    discrete_layout = flowers.tools.discrete_layout(n_turb=N_turb, seed_val=seed_val)
    flowers.visualization.plot_layout(discrete_layout[0], discrete_layout[1])

    flowers_model_discrete = FlowersModel(
        wind_rose=df_wr,
        layout_x=discrete_layout[0],
        layout_y=discrete_layout[1],
        turbine="nrel_5MW",
    )
    AEP_d, dAEP_d = [v / 1.0e9 for v in flowers_model_discrete.calculate_aep(gradient=True)]
    print(f"discrete farm AEP: {AEP_d}")
    plt.show() if show_plots else plt.close('all')

    random_layout = flowers.tools.random_layout(boundaries=boundaries, n_turb=N_turb, seed_val=seed_val)
    flowers.visualization.plot_layout(random_layout[0], random_layout[1])

    flowers_model_random = FlowersModel(
        wind_rose=df_wr,
        layout_x=random_layout[0],
        layout_y=random_layout[1],
        turbine="nrel_5MW",
    )
    AEP_r, dAEP_r = [v / 1.0e9 for v in flowers_model_random.calculate_aep(gradient=True)]
    print(f"random farm AEP: {AEP_r}")
    plt.show() if show_plots else plt.close('all')
