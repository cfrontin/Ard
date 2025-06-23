
# Introduction

**Dig in to wind farm design.**

<!-- The (aspirationally) foolproof tool for preparing wind farm layouts. -->

[An ard is a type of simple and lightweight plow](https://en.wikipedia.org/wiki/Ard_\(plough\)), used through the single-digit centuries to prepare a farm for planting.
The intent of `Ard` is to be a modular, full-stack multi-disciplinary optimization tool for wind farms.

The problem with wind farms is that they are complicated, multi-disciplinary objects.
They are aerodynamic machines, with complicated control systems, power electronic devices, social and political objects, and the core value (and cost) of complicated financial instruments.
Moreover, the design of *one* of these aspects affects all the rest!

`Ard` seeks to make plant-level design choices that can incorporate these different aspects _and their interactions_ to make wind energy projects more successful.

## Documentation
The documentation organized as follows: 
- **Getting Started**: helpful information to get you started using Ard, including installation instructions
- **Tutorials**: example problems with solutions
- **How-to Guides**: instructions on solving particular problems and using specific features of Ard
- **Reference**: dry, but hopefully useful, reference material including the API reference manual
- **Explanation**: discursive content on the whys, hows, and the wherefores of Ard for those that are interested


## Current capabilities

For the alpha pre-release of `Ard`, we have concentrated on optimization of wind plants, starting from a structured layout and optimizing to minimize the levelized cost of energy, or LCOE.
This capability is demonstrated in `examples/LCOE_stack` and tested in an abridged form in `test/system/LCOE_stack/test_LCOE_stack.py`.
In the alpha pre-release stage, the constituent subcomponents of these problems are known to work and fully tested; any capabilities not touched in the layout-to-LCOE stack should be treated as experimental.

These cases start from a four parameter farm layout, compute landuse area, make FLORIS AEP estimates, compute turbine capital costs, balance-of-station (BOS), and operational costs using WISDEM components, and finally give summary estimates of plant finance figures.
The components that achieve this can be assembled to either run a single top-down analysis run, or run an optimization.

## Roadmap to future capabilities

The future development of `Ard` is centered around two user cases:
1) systems energy researchers who are focusing on one specific subdiscipline (e.g. layout strategies, social impacts, or aerodynamic modeling) but want to be able to easily keep track of how it impacts the entire value chain down to production, cost, and/or value of energy or even optimize with respect to it, and
2) private industry researchers who are interested in how public-sector research results change when proprietary analysis tools are dropped in and coupled the other tools in a systems-level simulation.

`Ard` is being developed as a modular tool to enable these types of research queries.
This starts from our research goals, which are that `Ard` should be:
1) principled: fully documented, and adhering to best-practices for code development
1) modular and extensible: choose the parts you want, skip the ones you don't, build yourself the ones we don't have
1) effective: fully tested and testable at the unit and system level, and built with a derivative-forward approach

This, then, allows us to attempt to accomplish the technical goals of `Ard`, to:
1) allow optimization of wind farm layouts for specific wind resource profiles
1) target wholistic and complex system-level optimization objectives like LCOE and beyond-LCOE metrics
1) naturally incorporate multi-fidelity analyses to efficiently integrate physics-resolving simulation

---

Copyright &copy; 2024, Alliance for Sustainable Energy, LLC
