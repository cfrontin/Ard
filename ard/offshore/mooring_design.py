import numpy as np

import openmdao.api as om


class BathymetryData:
    """
    A draft class to hold and represent Bathymetry data for a mooring system.

    This class should be modified by whoever is implementing it in order to
    improve it! I just made it a boilerplate version of what I anticipated would
    be in it, even though I'm an idiot! -cfrontin
    """

    x_mesh = np.atleast_2d([0.0])  # x location in km
    y_mesh = np.atleast_2d([0.0])  # y location in km
    depth_mesh = np.atleast_2d([0.0])  # depth in m? km?

    material_mesh = np.atleast_2d(["sand"])  # DRAFT

    cost_dictionary = {
        "sand": 10.0,  # DRAFT
        "rock": 100.0,  # DRAFT
    }  # cost of anchoring in a given material per relevant metric

    def get_cost(self):
        """Get the cost of anchor building at a given location."""
        raise NotImplementedError("Bathymetry data must be implemented still.")


class MooringDesign(om.ExplicitComponent):
    """
    A class to create a mooring design for an floating offshore wind farm.

    This is a class that should be used to generate a floating offshore wind
    farm's collective mooring system.

    Options
    -------
    modeling_options : dict
        a modeling options dictionary (inherited from `FarmAeroTemplate`)
    wind_query : floris.wind_data.WindRose
        a WindQuery objects that specifies the wind conditions that are to be
        computed
    bathymetry_data : BathymetryData
        a BathymetryData object to specify the bathymetry mesh/sampling

    Inputs
    ------
    phi_platform : np.ndarray
        a 1D numpy array indicating the cardinal direction angle of the mooring
        orientation, with length `N_turbines`
    x_turbines : np.ndarray
        a 1D numpy array indicating the x-dimension locations of the turbines,
        with length `N_turbines` (mirrored w.r.t. `FarmAeroTemplate`)
    y_turbines : np.ndarray
        a 1D numpy array indicating the y-dimension locations of the turbines,
        with length `N_turbines` (mirrored w.r.t. `FarmAeroTemplate`)
    thrust_turbines : np.ndarray
        an array of the wind turbine thrust for each of the turbines in the farm
        across all of the conditions that have been queried on the wind rose
        (`N_turbines`, `N_wind_conditions`)

    Outputs
    -------
    x_anchors : np.ndarray
        a 1D numpy array indicating the x-dimension locations of the mooring
        system anchors, with shape `N_turbines` x `N_anchors`
    y_anchors : np.ndarray
        a 1D numpy array indicating the y-dimension locations of the mooring
        system anchors, with shape `N_turbines` x `N_anchors`

    """

    def initialize(self):
        """Initialization of the OpenMDAO component."""
        self.options.declare("modeling_options")

        # farm power wind conditions query (not necessarily a full wind rose)
        self.options.declare("wind_query")

        # currently I'm thinking of sea bed conditions as a class, see above
        self.options.declare("bathymetry_data")  # BatyhmetryData object

    def setup(self):
        """Setup of the OpenMDAO component."""

        # load modeling options
        self.modeling_options = self.options["modeling_options"]
        self.N_turbines = self.modeling_options["farm"]["N_turbines"]
        self.N_anchors = self.modeling_options["platform"]["N_anchors"]
        # get the number of wind conditions (for thrust measurements)
        self.N_wind_conditions = self.options["wind_query"].N_conditions
        # MANAGE ADDITIONAL LATENT VARIABLES HERE!!!!!

        # set up inputs and outputs for mooring system
        self.add_input(
            "phi_platform", np.zeros((self.N_turbines,)), units="deg"
        )  # cardinal direction of the mooring platform orientation
        self.add_input(
            "x_turbines", np.zeros((self.N_turbines,)), units="km"
        )  # x location of the mooring platform in km w.r.t. reference coordinates
        self.add_input(
            "y_turbines", np.zeros((self.N_turbines,)), units="km"
        )  # y location of the mooring platform in km w.r.t. reference coordinates
        self.add_input(
            "thrust_turbines",
            np.zeros((self.N_turbines, self.N_wind_conditions)),
            units="kN",
        )  # turbine thrust coming from each wind direction
        # ADD ADDITIONAL (DESIGN VARIABLE) INPUTS HERE!!!!!

        self.add_output(
            "x_anchors",
            np.zeros((self.N_turbines, self.N_anchors)),
            units="km",
        )  # x location of the mooring platform in km w.r.t. reference coordinates
        self.add_output(
            "y_anchors",
            np.zeros((self.N_turbines, self.N_anchors)),
            units="km",
        )  # y location of the mooring platform in km w.r.t. reference coordinates
        self.add_output(
            "cost_mooring_turbine",
            np.zeros((self.N_turbines,)),
            units="MUSD",
        )  # cost of the mooring system for each turbine
        self.add_output(
            "cost_mooring_farm",
            0.0,
            units="MUSD",
        )  # cost of the mooring system across all turbines
        # ADD ADDITIONAL (DESIGN VARIABLE) OUTPUTS HERE!!!!!

    def setup_partials(self):
        """Derivative setup for the OpenMDAO component."""
        # the default (but not preferred!) derivatives are FDM
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """Computation for the OpenMDAO component."""

        # unpack the working variables
        phi_platform = inputs["phi_platform"]
        x_turbines = inputs["x_turbines"]
        y_turbines = inputs["y_turbines"]
        thrust_turbines = inputs["thrust_turbines"]  #

        ########################################################################
        #
        # this is where the magic will happen!
        #
        # here, the implementor will take the variables above and map them into
        # the machinery for computing a mooring design for each of the turbines.
        #
        ########################################################################

        raise NotImplementedError("This component is awaiting implementation!")

        # replace the below with the final anchor locations...
        outputs["x_anchors"] = None
        outputs["y_anchors"] = None
        # ... and the final costs
        outputs["cost_mooring_turbine"] = None
        outputs["cost_mooring_farm"] = None
