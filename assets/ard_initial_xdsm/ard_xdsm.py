# ########################################################################### #
#                                                                             #
# an XDSM diagram builder for the baseline Ard architecture               #
#                                                                             #
# ########################################################################### #

from pyxdsm.XDSM import XDSM, OPT, SOLVER, FUNC, LEFT, RIGHT

# create the xdsm model
xmodel = XDSM(use_sfmath=False)

# some internal settings
optimizer_on = True
compute_financials = False
aero_modules = ["floris"]  # ["floris", "windse"]
is_land_based = True


### CREATE THE XDSM MODEL

if optimizer_on:
    # set up the outer level of the optimizer if requested
    xmodel.add_system(
        "optimizer",
        OPT,
        (r"\mathrm{constrained}", r"\mathrm{optimizer}"),
    )
# set up the fundamental system inputs and outputs
xmodel.add_system("layout", SOLVER, r"\mathrm{layout}")  # layout to location
xmodel.add_system("landuse", SOLVER, r"\mathrm{landuse}")  # layout to land use
xmodel.add_system(
    "plex",
    SOLVER,
    (r"\mathrm{mfAEP.demux}",),
)  # add a "demultiplexer" to plan power evaluations based on wind rose & tools
# turn on the aerodynamics modules that will be used
if "floris" in aero_modules:
    xmodel.add_system("floris", SOLVER, r"\mathrm{FLORIS}")
if "windse" in aero_modules:
    xmodel.add_system("windse", SOLVER, r"\mathrm{WindSE}")
# take the power evaluations and recombine them in an integrator
xmodel.add_system("aep", FUNC, r"\mathrm{mfAEP.BQ}")
if compute_financials:
    # add computations for annualized capital and operational expenses
    xmodel.add_system("capex", SOLVER, r"\mathrm{CapEx estimator}")
    xmodel.add_system("opex", SOLVER, r"\mathrm{OpEx estimator}")
# add the right BOS cost computer for land-based/offshore computations
if is_land_based:
    xmodel.add_system("bos", SOLVER, r"\mathrm{LandBOSSE}")
else:
    xmodel.add_system("bos", SOLVER, r"\mathrm{ORBIT}")
# recombine costs and production to get to an LCOE measure
xmodel.add_system("lcoe", FUNC, r"\mathrm{LCOE}")


### MAKE XDSM CONNECTIONS

# optimizer drives the layout variables
if optimizer_on:
    xmodel.connect("optimizer", "layout", r"\theta, L_1, \phi, L_2")
    xmodel.connect("optimizer", "landuse", r"\theta, L_1, \phi, L_2")
# layout and wind conditions get piped to the aero solvers
if "floris" in aero_modules:
    xmodel.connect("layout", "floris", r"\{(x, y)_t\}_t", stack=True)
    xmodel.connect(
        "plex",
        "floris",
        r"\{(\psi_{w_{\mathrm{lo}}}, "
        + r"V_{w_{\mathrm{lo}}})_{w_{\mathrm{lo}}}\}_{w_{\mathrm{lo}}}",
        stack=True,
    )
if "windse" in aero_modules:
    xmodel.connect("layout", "windse", r"\{(x, y)_t\}_t", stack=True)
    xmodel.connect(
        "plex",
        "windse",
        r"\{(\psi_{w_{\mathrm{hi}}}, "
        + r"V_{w_{\mathrm{hi}}})_{w_{\mathrm{hi}}}\}_{w_{\mathrm{hi}}}",
        stack=True,
    )
# in the multiplexing case, the aep needs to get the weights from the demux
# otherwise they're just based on the level at hand
if len(aero_modules) > 1:
    xmodel.connect(
        "plex",
        "aep",
        (
            r"\{\omega_{w_{\mathrm{lo}}}\}_{w_{\mathrm{lo}}}",
            r"\{\omega_{w_{\mathrm{hi}}}\}_{w_{\mathrm{hi}}}",
        ),
    )
elif "floris" in aero_modules:
    xmodel.connect("plex", "aep", r"\{\omega_{w_{\mathrm{lo}}}\}_{w_{\mathrm{lo}}}")
elif "windse" in aero_modules:
    xmodel.connect("plex", "aep", r"\{\omega_{w_{\mathrm{hi}}}\}_{w_{\mathrm{hi}}}")
# connect the powers out of the aero model to the AEP computations
if "floris" in aero_modules:
    xmodel.connect(
        "floris",
        "aep",
        r"\{P_{t,w_{\mathrm{lo}}}^\mathrm{FLORIS}\}_{t,w_{\mathrm{lo}}}",
        stack=True,
    )
if "windse" in aero_modules:
    xmodel.connect(
        "windse",
        "aep",
        r"\{P_{t,w_{\mathrm{hi}}}^\mathrm{WindSE}\}_{t,w_{\mathrm{hi}}}",
        stack=True,
    )
# BOS cost model is dependent on layout
if optimizer_on:
    xmodel.connect("optimizer", "bos", r"\theta, L_1, \phi, L_2")
# connect the upstream data to the LCOE estimation
xmodel.connect("aep", "lcoe", r"\mathrm{AEP}")
xmodel.connect("bos", "lcoe", r"\mathrm{BOS}")
# finally send the constraint and objective back to the optimizer
if optimizer_on:
    xmodel.connect("landuse", "optimizer", r"A_{\mathrm{landuse}}")
    xmodel.connect("lcoe", "optimizer", r"\mathrm{LCOE}")

### CONFIGURE THE INPUTS AND OUTPUTS

# if the optimizer is off, the DVs are just inputs
if not optimizer_on:
    xmodel.add_input("layout", r"\theta, L_1, \phi, L_2")
    xmodel.add_input("landuse", r"\theta, L_1, \phi, L_2")
# add the always-on inputs
xmodel.add_input("plex", r"\{\psi_w,V_w,p(\psi_w,V_w)\}_w", stack=True)
xmodel.add_input("aep", r"\{\psi_w,V_w,p(\psi_w,V_w)\}_w", stack=True)
if not compute_financials:
    xmodel.add_input("lcoe", r"\mathrm{CapEx},\mathrm{OpEx}")

# add tracking for the objective, constraint, and other QoIs
xmodel.add_output("landuse", r"A_{\mathrm{landuse}}", side=RIGHT)
xmodel.add_output("aep", r"\mathrm{AEP}", side=RIGHT)
xmodel.add_output("lcoe", r"\mathrm{LCOE}", side=RIGHT)

# output the result!
xmodel.write("ard_xdsm")

### FIN!
