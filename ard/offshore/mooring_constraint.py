import numpy as np


class MooringConstraint(om.ExplicitComponent):
    """
    A class to reduce complex mooring constraints into a simple violation lengthscale

    TODO

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
    x_turbines : np.ndarray
        a 1D numpy array indicating the x-dimension locations of the turbines,
        with length `N_turbines` (mirrored w.r.t. `FarmAeroTemplate`)
    y_turbines : np.ndarray
        a 1D numpy array indicating the y-dimension locations of the turbines,
        with length `N_turbines` (mirrored w.r.t. `FarmAeroTemplate`)
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

    def setup(self):
        """Setup of the OpenMDAO component."""

        # load modeling options
        self.modeling_options = self.options["modeling_options"]
        self.N_turbines = self.modeling_options["farm"]["N_turbines"]
        self.N_anchors = self.modeling_options["platform"]["N_anchors"]
        # MANAGE ADDITIONAL LATENT VARIABLES HERE!!!!!

        # set up inputs and outputs for mooring system
        self.add_input(
            "x_turbines", np.zeros((self.N_turbines,)), units="km"
        )  # x location of the mooring platform in km w.r.t. reference coordinates
        self.add_input(
            "y_turbines", np.zeros((self.N_turbines,)), units="km"
        )  # y location of the mooring platform in km w.r.t. reference coordinates
        self.add_input(
            "x_anchors",
            np.zeros((self.N_turbines, self.N_anchors)),
            units="km",
        )  # x location of the mooring platform in km w.r.t. reference coordinates
        self.add_input(
            "y_anchors",
            np.zeros((self.N_turbines, self.N_anchors)),
            units="km",
        )  # y location of the mooring platform in km w.r.t. reference coordinates
        # ADD ADDITIONAL (DESIGN VARIABLE) INPUTS HERE!!!!!

        self.add_output(
            "violation_distance",
            0.0,
            units="km",
        )  # consolidated violation length
        # ADD ADDITIONAL (DESIGN VARIABLE) OUTPUTS HERE!!!!!

    def setup_partials(self):
        """Derivative setup for the OpenMDAO component."""
        # the default (but not preferred!) derivatives are FDM
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """Computation for the OpenMDAO component."""

        # unpack the working variables
        x_turbines = inputs["x_turbines"]
        y_turbines = inputs["y_turbines"]
        x_anchors = inputs["x_anchors"]
        y_anchors = inputs["y_anchors"]

        ########################################################################
        #
        # this is where the magic will happen!
        #
        # here, the implementor will take the variables above and map them into
        # the machinery for computing a mooring design for each of the turbines.
        #
        ########################################################################

        raise NotImplementedError("This component is awaiting implementation!")

        # replace the below with the final values
        outputs["violation_distance"] = None
