import importlib
import openmdao.api as om
from ard.cost.wisdem_wrap import LandBOSSE, ORBIT, PlantFinance
from ard.cost.wisdem_wrap import (
    LandBOSSE_setup_latents,
    ORBIT_setup_latents,
    FinanceSE_setup_latents,
)


def set_up_system(input_dict):

    prob = om.Problem()

    for group_key in input_dict["groups"].keys():

        group = prob.model.add_subsystem(
            name=group_key,
            subsys=om.Group(),
            promotes=input_dict["groups"][group_key]["promotes"],
        )

        for subsystem_key in input_dict["groups"][group_key]["systems"]:

            module_name = input_dict["groups"][group_key]["systems"][subsystem_key][
                "name"
            ]
            object_name = input_dict["groups"][group_key]["systems"][subsystem_key][
                "object"
            ]

            Module = importlib.import_module(f"ard.{module_name}")
            SubSystem = getattr(Module, object_name)

            subsystem_type = input_dict["groups"][group_key]["systems"][subsystem_key][
                "type"
            ]

            # subsystem_input_dict = input_dict[subsystem_type][subsystem_key]
            group.add_subsystem(
                name=subsystem_key,
                subsys=SubSystem(input_dict=input_dict, name=subsystem_key),
                promotes=input_dict[subsystem_type][subsystem_key]["promotes"],
            )

    prob.setup()

    return prob


def set_up_system_recursive(
    input_dict: dict,
    system_name: str="top_level",
    parent_group=None,
    modeling_options: dict=None,
    analysis_options: dict=None,
    _depth: int=0,
):
    """
    Recursively sets up an OpenMDAO system based on the input dictionary.

    Args:
        input_dict (dict): Dictionary defining the system hierarchy.
        parent_group (om.Group, optional): The parent group to which subsystems are added.
                                           Defaults to None, which initializes the top-level problem.

    Returns:
        om.Problem: The OpenMDAO problem with the defined system hierarchy.
    """
    # Initialize the top-level problem if no parent group is provided
    if parent_group is None:
        prob = om.Problem()
        parent_group = prob.model
    else:
        prob = None

    # Add subsystems directly from the input dictionary
    if hasattr(parent_group, "name"):
        print(f"Adding {system_name} to {parent_group.name}")
    else:
        print(f"Adding {system_name}")
    if "systems" in input_dict:  # Recursively add nested subsystems]
        if _depth > 0:
            group = parent_group.add_subsystem(
                name=system_name,
                subsys=om.Group(),
                promotes=input_dict.get("promotes", None),
            )
        else:
            group = parent_group
        for subsystem_key, subsystem_data in input_dict["systems"].items():
            set_up_system_recursive(
                subsystem_data,
                parent_group=group,
                system_name=subsystem_key,
                modeling_options=modeling_options,
                analysis_options=None,
                _depth=_depth + 1,
            )

    else:
        subsystem_data = input_dict

        if "object" not in subsystem_data:
            raise ValueError(f"Ard subsystem '{system_name}' missing 'object' spec.")
        if "promotes" not in subsystem_data:
            raise ValueError(f"Ard subsystem '{system_name}' missing 'promotes' spec.")

        # Dynamically import the module and get the subsystem class
        Module = importlib.import_module(subsystem_data["module"])
        SubSystem = getattr(Module, subsystem_data["object"])

        # Add the subsystem to the parent group with kwargs
        parent_group.add_subsystem(
            name=system_name,
            subsys=SubSystem(**subsystem_data.get("kwargs", {})),
            promotes=subsystem_data["promotes"],
        )

    # Handle connections within the parent group
    if "connections" in input_dict:
        for connection in input_dict["connections"]:
            src, tgt = connection  # Unpack the connection as [src, tgt]
            parent_group.connect(src, tgt)

    # Set up the problem if this is the top-level call
    if prob is not None:

        if analysis_options:
            # set up driver
            if "driver" in analysis_options:
                Driver = getattr(om, analysis_options["driver"]["name"])
                prob.driver = Driver()
                if "options" in analysis_options["driver"]:
                    for option, value in analysis_options["driver"]["options"].items():
                        prob.driver.options[option] = value

                    # set design variables
            if "design_variables" in analysis_options:
                for var_name, var_data in analysis_options["design_variables"].items():
                    prob.model.add_design_var(var_name, **var_data)

            # set constraints
            if "constraints" in analysis_options:
                for constraint_name, constraint_data in analysis_options["constraints"].items():
                    prob.model.add_constraint(constraint_name, **constraint_data)

            # set objective
            if "objective" in analysis_options:
                prob.model.add_objective(analysis_options["objective"]["name"], **analysis_options["objective"]["options"])

            # Set up the recorder if specified in the input dictionary
            if "recorder" in analysis_options:
                recorder_filepath = analysis_options["recorder"].get("filepath")
                if recorder_filepath:
                    recorder = om.SqliteRecorder(recorder_filepath)
                    prob.add_recorder(recorder)
                    prob.driver.add_recorder(recorder)

        prob.setup()

    if _depth == 0:
        # setup the latent variables for LandBOSSE/ORBIT and FinanceSE
        if any("orbit" in var[0] for var in prob.model.list_vars(val=False, out_stream=None)):
            ORBIT_setup_latents(prob, modeling_options)
        if any("landbosse" in var[0] for var in prob.model.list_vars(val=False, out_stream=None)):
            LandBOSSE_setup_latents(prob, modeling_options)
        if any("financese" in var[0] for var in prob.model.list_vars(val=False, out_stream=None)):
            FinanceSE_setup_latents(prob, modeling_options)

    return prob