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
is_land_based = False


### CREATE THE XDSM MODEL

if optimizer_on:
    # set up the outer level of the optimizer if requested
    xmodel.add_system(
        "optimizer",
        OPT,
        (r"\textrm{constrained}", r"\textrm{optimizer}"),
    )
# set up the fundamental system inputs and outputs
xmodel.add_system("layout", SOLVER, r"\texttt{layout}")  # layout to location
if is_land_based:
    xmodel.add_system("landuse", SOLVER, r"\texttt{landuse}")  # layout to land use
# turn on the aerodynamics modules that will be used
xmodel.add_system("farm_aero", SOLVER, r"\texttt{farm\_aero}")
if compute_financials:
    # add computations for annualized capital and operational expenses
    xmodel.add_system("capex", SOLVER, r"\textrm{CapEx estimator}")
    xmodel.add_system("opex", SOLVER, r"\textrm{OpEx estimator}")
# add mooring design component
xmodel.add_system("mooring", SOLVER, (r"\textrm{mooring}", r"\textrm{design}"))
# add the right BOS cost computer for land-based/offshore computations
xmodel.add_system("mooring_constr", FUNC, (r"\textrm{mooring}", r"\textrm{constraint}"))
if is_land_based:
    xmodel.add_system("bos", SOLVER, r"\textrm{LandBOSSE}")
else:
    xmodel.add_system("bos", SOLVER, r"\textrm{ORBIT}")
# recombine costs and production to get to an LCOE measure
xmodel.add_system("lcoe", FUNC, r"\textrm{LCOE}")


### MAKE XDSM CONNECTIONS

# optimizer drives the layout variables
if optimizer_on:
    xmodel.connect("optimizer", "layout", r"\theta_{\mathrm{layout}}")
    if is_land_based:
        xmodel.connect("optimizer", "landuse", r"\theta_{\mathrm{layout}}")
    xmodel.connect("optimizer", "mooring", r"\phi_{\mathrm{ptfm}}")
# layout and wind conditions get piped to the aero solvers
xmodel.connect("layout", "farm_aero", r"\{(x, y)_t\}_t", stack=True)
xmodel.connect("layout", "mooring", r"\{(x, y)_t\}_t", stack=True)
xmodel.connect("layout", "mooring_constr", r"\{(x, y)_t\}_t", stack=True)
xmodel.connect("layout", "bos", r"\{(x, y)_t\}_t", stack=True)
xmodel.connect("farm_aero", "mooring", r"\{T_{t,w}\}_{t,w}", stack=True)
xmodel.connect("mooring", "mooring_constr", r"\{(x_A, y_A)_{t,a}\}_{t,a}", stack=True)
xmodel.connect("mooring", "bos", r"\mathcal{C}_{\mathrm{mooring}}")
# BOS cost model is dependent on layout
if optimizer_on:
    xmodel.connect("optimizer", "bos", r"\theta_{\mathrm{layout}}")
# connect the upstream data to the LCOE estimation
xmodel.connect("farm_aero", "lcoe", r"\textrm{AEP}")
xmodel.connect("bos", "lcoe", r"\textrm{BOS}")
# finally send the constraint and objective back to the optimizer
if optimizer_on:
    if is_land_based:
        xmodel.connect("landuse", "optimizer", r"A_{\textrm{landuse}}")
    xmodel.connect("mooring_constr", "optimizer", r"d_{\textrm{mc}}")
    xmodel.connect("lcoe", "optimizer", r"\textrm{LCOE}")

### CONFIGURE THE INPUTS AND OUTPUTS

# if the optimizer is off, the DVs are just inputs
if not optimizer_on:
    xmodel.add_input("layout", r"\theta_{\mathrm{layout}}")
    if is_land_based:
        xmodel.add_input("landuse", r"\theta_{\mathrm{layout}}")
# add the always-on inputs
xmodel.add_input("farm_aero", r"\{\psi_w,V_w,p(\psi_w,V_w)\}_w", stack=True)
# add the bathymetry data for mooring design
xmodel.add_input("mooring", r"\{x_b, y_b, z_b\}_b", stack=True)
if not compute_financials:
    xmodel.add_input("lcoe", r"\textrm{CapEx},\textrm{OpEx}")

# add tracking for the objective, constraint, and other QoIs
if is_land_based:
    xmodel.add_output("landuse", r"A_{\textrm{landuse}}", side=RIGHT)
xmodel.add_output("farm_aero", r"\textrm{AEP}", side=RIGHT)
xmodel.add_output("mooring_constr", r"d_{\mathrm{mc}}", side=RIGHT)  # DEBUG!!!!!
xmodel.add_output("lcoe", r"\textrm{LCOE}", side=RIGHT)

# output the result!
xmodel.write("ard_xdsm")

### FIN!
