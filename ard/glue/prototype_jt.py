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
    input_dict,
    system_name="top_level",
    parent_group=None,
    modeling_options={},
    _depth=0,
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
    if "systems" in input_dict:  # Recursively add nested subsystems
        group = parent_group.add_subsystem(
            name=system_name,
            subsys=om.Group(),
            promotes=input_dict.get("promotes", None),
        )
        for subsystem_key, subsystem_data in input_dict["systems"].items():
            set_up_system_recursive(
                subsystem_data,
                parent_group=group,
                system_name=subsystem_key,
                modeling_options=modeling_options,
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

        # Handle defaults for WISDEM wrappers
        # needs_latents = [ORBIT, PlantFinance]  # , LandBOSSE]
        # latents_setters = [
        #     ORBIT_setup_latents,
        #     FinanceSE_setup_latents,
        # ]  # , LandBOSSE_setup_latents]
        # for obj_type, latent_setter in zip(needs_latents, latents_setters):
        #     if isinstance(SubSystem, obj_type):
        #         latent_setter(prob, modeling_options)

    # Handle connections within the parent group
    if "connections" in input_dict:
        for connection in input_dict["connections"]:
            src, tgt = connection  # Unpack the connection as [src, tgt]
            parent_group.connect(src, tgt)

    # Set up the problem if this is the top-level call
    if prob is not None:
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
