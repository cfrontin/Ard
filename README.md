
# Ard

**Dig into wind farm design.**

<!-- The (aspirationally) foolproof tool for preparing wind farm layouts. -->

[An ard is a type of simple and lightweight plow](https://en.wikipedia.org/wiki/Ard_\(plough\)), used through the single-digit centuries to prepare a farm for planting.
The intent of `Ard` is to be a modular, full-stack multi-disciplinary optimization tool for wind farms.

The problem with wind farms is that they are complicated, multi-disciplinary objects.
They are aerodynamic machines, with complicated control systems, power electronic devices, social and political objects, and the core value (and cost) of complicated financial instruments.
Moreover, the design of *one* of these aspects affects all the rest!

`Ard` seeks to make plant-level design choices that can incorporate these different aspects _and their interactions_ to make wind energy projects more successful.

## Installation instructions

<!-- `Ard` can be installed locally from the source code with `pip` or through a package manager from PyPI with `pip` or conda-forge with `conda`. -->
<!-- For Windows systems, `conda` is required due to constraints in the WISDEM installation system. -->
<!-- For macOS and Linux, any option is available. -->
`Ard` is currently in pre-release and is only available as a source-code installation.
The source can be cloned from github using the following command in your preferred location:
```shell
git clone git@github.com:WISDEM/Ard.git
```
Once downloaded, you can enter the `Ard` root directory using
```shell
cd Ard
```

At this point, although not strictly required, we recommend creating a dedicated conda environment with `pip`, `python=3.12`, and `mamba` in it:
```shell
conda create --name ard-env
conda activate ard-env
conda install python=3.12 pip mamba -y
```

From here, installation can be handled by `pip`. 

For a basic and static installation, type:
```shell
pip install .
```

For development (and really for everyone during pre-release), we recommend a full development installation:
```shell
pip install -e .[dev,docs]
```
which will install in "editable mode" (`-e`), such that changes made to the source will not require re-installation, and with additional optional packages for development and documentation (`[dev,docs]`).

There can be some hardware-software mis-specification issues with WISDEM installation from `pip` for MacOS 12 and 13 on machines with Apple Silicon.
In the event of issues, WISDEM can be installed manually or using `conda` without issues, then `pip` installation can proceed.

```shell
mamba install wisdem=3.18.1 -y
pip install -e .[dev,docs]
```

To test the installation, from the `Ard` folder run unit and regression tests:
```shell
source test/run_local_test_unit.sh
source test/run_local_test_system.sh
```

For user information, in pre-release, we are using some co-developed changes to the `FLORIS` library.

If the installation fails, please open a new issue [here](https://github.com/WISDEM/Ard/issues).

## Current capabilities

For the alpha pre-release of `Ard`, we have concentrated on optimization of wind plants, starting from a structured layout and optimizing it to minimize the levelized cost of energy, or LCOE.
This capability is demonstrated for a land-based (LB) wind farm in `examples/LCOE_LB_stack` and tested in an abridged form in `test/system/LCOE_stack/test_LCOE_LB_stack.py`.
In this example, the wind farm layout is parametrized with two angles, named orientation and skewed, and turbine distancing for rows and columns.
In the alpha pre-release stage, the constituent subcomponents of these problems are known to work and fully tested; any capabilities not touched in the layout-to-LCOE stack should be treated as experimental.

These cases start from a four parameter farm layout, compute landuse area, make FLORIS AEP estimates, compute turbine capital costs, balance-of-station (BOS), and operational costs using WISDEM components, and finally give summary estimates of plant finance figures.
The components that achieve this can be assembled to either run a single top-down analysis run, or run an optimization.

A second example is in progress to reoptimize the layout of two offshore wind farms, one fixed bottom (OFB) and one floating (OFL).
Both wind farms are made of the [22 MW reference wind turbine](https://github.com/IEAWindSystems/IEA-22-280-RWT).
In this example, BOS costs are estimated using the tool [Orbit](https://github.com/WISDEM/ORBIT).

## Roadmap to future capabilities

The future development of `Ard` is centered around two user cases:
1) systems energy researchers who are focusing on one specific subdiscipline (e.g. layout strategies, social impacts, or aerodynamic modeling) but want to be able to easily keep track of how it impacts the entire value chain down to production, cost, and/or value of energy or even optimize with respect to it, and
2) private industry researchers who are interested in how public-sector research results change when proprietary analysis tools are dropped in and coupled the other tools in a systems-level simulation.

`Ard` is being developed as a modular tool to enable these types of research queries.
This starts from our research goals, which are that `Ard` should be:
1) principled: fully documented, and adhering to best-practices for code development
2) modular and extensible: choose the parts you want, skip the ones you don't, build yourself the ones we don't have
3) effective: fully tested and testable at the unit and system level, and built with a derivative-forward approach

This, then, allows us to attempt to accomplish the technical goals of `Ard`, to:
1) allow optimization of wind farm layouts for specific wind resource profiles
2) target wholistic and complex system-level optimization objectives like LCOE and beyond-LCOE metrics
3) naturally incorporate multi-fidelity analyses to efficiently integrate physics-resolving simulation

---

Released as open-source software by the National Renewable Energy Laboratory under NREL software record number SWR-25-18.

Copyright &copy; 2024, Alliance for Sustainable Energy, LLC.
