import json



# Queries

def read_queries(path_data_queries: str) -> list[dict[str, object]]:
    """Read queries.

    Args:
        path_data_queries (str): path to the queries data.

    Returns:
        list[dict[str, object]]: queries.
    """
    with open(path_data_queries, "r") as file:
        return json.load(file)

def write_queries(queries: list[dict[str, object]], path_data_queries: str) -> None:
    """Write queries.

    Args:
        queries (list[dict[str, object]]): queries to write.
        path_data_queries (str): path to the queries data.
    """
    with open(path_data_queries, "w") as file:
        json.dump(queries, file)

# Registries

def read_registries(path_data_registries: str) -> list[dict[str, object]]:
    """Read registries.

    Args:
        path_data_registries (str): path to the registries data.

    Returns:
        list[dict[str, object]]: registries.
    """
    with open(path_data_registries, "r") as file:
        return json.load(file)

def write_registries(registries: list[dict[str, object]], path_data_registries: str) -> None:
    """Write registries.

    Args:
        registries (list[dict[str, object]]): registries to write.
        path_data_registries (str): path to the registries data.
    """
    with open(path_data_registries, "w") as file:
        json.dump(registries, file)

# Annotations

def read_annotations(path_data_annotations: str) -> list[dict[str, object]]:
    """Read annotations.

    Args:
        path_data_annotations (str): path to the annotations data.

    Returns:
        list[dict[str, object]]: annotations.
    """
    with open(path_data_annotations, "r") as file:
        return json.load(file)

def write_annotations(annotations: list[dict[str, object]], path_data_annotations: str) -> None:
    """Write annotations.

    Args:
        annotations (list[dict[str, object]]): annotations to write.
        path_data_annotations (str): path to the annotations data.
    """
    with open(path_data_annotations, "w") as file:
        json.dump(annotations, file)

# Searches

def read_searches(path_data_searches: str, setting_id: str) -> dict[str, object]:
    """Read searches.

    Args:
        path_data_searches (str): path to the searches data.
        setting_id (str): id of the setting.

    Returns:
        dict[str, object]: searches.
    """
    with open(f"{path_data_searches}/{setting_id}.json", "r") as file:
        return json.load(file)

def write_searches(searches: dict[str, object], path_data_searches: str, setting_id: str) -> None:
    """Write searches.

    Args:
        searches (dict[str, object]): searches to write.
        path_data_searches (str): path to the searches data.
        setting_id (str): id of the setting.
    """
    with open(f"{path_data_searches}/{setting_id}.json", "w") as file:
        json.dump(searches, file)
