import importlib
import openmdao.api as om 

def set_up_system(input_dict):

    prob = om.Problem()

    for group_key in input_dict["groups"].keys():

        group = prob.model.add_subsystem(name=group_key, subsys=om.Group(), promotes=input_dict["groups"][group_key]["promotes"])

        for subsystem_key in input_dict["groups"][group_key]["systems"]:

            module_name = input_dict["groups"][group_key]["systems"][subsystem_key]["name"]
            object_name = input_dict["groups"][group_key]["systems"][subsystem_key]["object"]
            
            Module = importlib.import_module(f"ard.{module_name}")
            SubSystem = getattr(Module, object_name)

            subsystem_type = input_dict["groups"][group_key]["systems"][subsystem_key]["type"]

            # subsystem_input_dict = input_dict[subsystem_type][subsystem_key]
            group.add_subsystem(name=subsystem_key, subsys=SubSystem(input_dict=input_dict, name=subsystem_key), promotes=input_dict[subsystem_type][subsystem_key]["promotes"])

    prob.setup()

    return prob


def set_up_system_recursive(input_dict, parent_group=None):
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

    # Iterate over groups in the input dictionary
    for group_key, group_data in input_dict["groups"].items():
        # Add the group to the parent group
        group = parent_group.add_subsystem(
            name=group_key,
            subsys=om.Group(),
            promotes=group_data.get("promotes", []),
        )

        # Iterate over systems in the group
        for subsystem_key, subsystem_data in group_data["systems"].items():
            module_name = subsystem_data["name"]
            object_name = subsystem_data["object"]

            # Dynamically import the module and get the subsystem class
            Module = importlib.import_module(f"ard.{module_name}")
            SubSystem = getattr(Module, object_name)

            # Extract kwargs for the subsystem if specified
            subsystem_kwargs = subsystem_data.get("kwargs", {})

            # Check if the subsystem has nested systems (i.e., is a group)
            if "systems" in subsystem_data:
                # Recursively add nested subsystems
                set_up_system_recursive(subsystem_data, parent_group=group)
            else:
                # Add the subsystem to the group with kwargs
                group.add_subsystem(
                    name=subsystem_key,
                    subsys=SubSystem(input_dict=input_dict, name=subsystem_key, **subsystem_kwargs),
                    promotes=subsystem_data.get("promotes", []),
                )

        # Handle connections within the group
        if "connections" in group_data:
            for connection in group_data["connections"]:
                src, tgt = connection  # Unpack the connection as [src, tgt]
                group.connect(src, tgt)

    # Set up the problem if this is the top-level call
    if prob is not None:
        prob.setup()

    return prob