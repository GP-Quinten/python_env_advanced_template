import itertools



# settings

def compose_setting_id(setting: dict) -> str:
    """Compose the id for the preparing setting.
    
    Args:
        setting (dict): setting for preparing the registries.
        
    Returns:
        str: id of the setting.
    """

    # initialize id elements
    id_elements = []

    # iterate over parameters
    for parameter_name, parameter_value in setting.items():
        if parameter_value is True:
            id_elements.append(parameter_name)
        elif parameter_value is False or parameter_value == 0:
            pass
        elif isinstance(parameter_value, int):
            id_elements.append(f"{parameter_name}{parameter_value}")
        else:
            raise ValueError
    
    return "_".join(id_elements)

def generate_settings_grid(settings_space: dict) -> list[dict]:
    """Generate a grid of settings from the settings space.

    Args:
        settings_space (dict): settings space.
    
    Returns:
        list[dict]: grid of settings.
    """
    return [
        {
            parameter_name: parameter_value
            for parameter_name, parameter_value in zip(settings_space.keys(), parameters_values)
        }
        for parameters_values in itertools.product(*settings_space.values())
    ]

# Preparing

def prepare_registry(raw: dict[str, object], setting: dict) -> dict[str, object]:
    """Prepare a registry.

    Args:
        raw (dict[str, object]): raw registry to prepare.
        settings (dict): settings for preparing the registry.

    Returns:
        dict[str, object]: registry.
    """

    # initialize prepared registry
    prepared = {"registry_id": raw["registry_id"]}

    # iterate over parameters
    for parameter_name, parameter_value in setting.items():
        parameter_name = f"registry_{parameter_name}"
        if parameter_value is True:
            prepared[parameter_name] = raw[parameter_name]
        elif parameter_value is False or parameter_value == 0:
            pass
        elif isinstance(parameter_value, int):
            occurences = [occurence for _, occurence in raw[parameter_name].items()]
            occurences.sort(reverse=True)
            threshold = occurences[:parameter_value][-1] if len(occurences[:parameter_value]) > 0 else 0
            prepared[parameter_name] = [key for key, value in raw[parameter_name].items() if value >= threshold]
        else:
            raise ValueError

    return prepared

def prepare_registries(registries: list[dict[str, object]], setting: dict) -> list[dict[str, object]]:
    """Prepare registries.

    Args:
        registries (list[dict[str, object]]): registries to prepare.
        setting (dict): setting for preparing the registries.

    Returns:
        list[dict[str, object]]: registries.
    """
    return [prepare_registry(registry, setting=setting) for registry in registries]
