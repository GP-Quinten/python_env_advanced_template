import enum
import numpy as np
import weaviate
import itertools
import time



# setting

def compose_setting_id(setting: dict) -> str:
    """Compose the id for the searching setting.
    
    Args:
        setting (dict): setting for searching the registries.
        
    Returns:
        str: id of the setting.
    """

    # initialize id elements
    id_elements = []

    # iterate over parameters
    for parameter_name, parameter_value in setting.items():
        if parameter_name == "method":
            id_elements.append(parameter_value)
        elif isinstance(parameter_value, enum.Enum):
            id_elements.append(f"{parameter_name}{parameter_value.value.upper()}")
        else:
            id_elements.append(f"{parameter_name}{parameter_value}")
    
    return "_".join(id_elements).replace("-", "").replace(".", "")

def generate_settings_grid(settings_space: dict) -> list[dict]:
    """Generate a grid of settings from the settings space.

    Args:
        settings_space (dict): settings space.

    Returns:
        list[dict]: grid of settings.
    """
    return [
        {
            "method": method, 
            **{
                parameter_name: parameter_value
                for parameter_name, parameter_value in zip(method_settings_space.keys(), parameters_values)
            }
        }
        for method, method_settings_space in settings_space.items()
        for parameters_values in itertools.product(*method_settings_space.values())
    ]

# Searching

def search_with_vector(query: dict[str, object], collection: weaviate.Collection, threshold_rank: int|None=None, threshold_distance: float|None=None) -> list:
    """Search with vector method.

    Args:
        query (dict[str, object]): query to search with.
        collection (weaviate.Collection): collection of registries to search in.
        threshold_rank (int|None, optional): number of registries to return. Defaults to None.

    Returns:
        list: search results.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return collection.query.near_text(
                query["query_text"],
                limit=threshold_rank, distance=threshold_distance,
                return_metadata=weaviate.classes.query.MetadataQuery(
                    score=True,
                    distance=True,
                    explain_score=True,
                    is_consistent=True,
                ),
            ).objects
        except weaviate.exceptions.WeaviateQueryError as e:
            if "Rate limit exceeded" in str(e):
                time.sleep(2 ** attempt)
                continue
            raise
    # final attempt
    return collection.query.near_text(
        query["query_text"],
        limit=threshold_rank, distance=threshold_distance,
        return_metadata=weaviate.classes.query.MetadataQuery(
            score=True,
            distance=True,
            explain_score=True,
            is_consistent=True,
        ),
    ).objects

def search_with_keyword(query: dict[str, object], collection: weaviate.Collection, threshold_rank: int|None=None) -> list:
    """Search with keyword method.

    Args:
        query (dict[str, object]): query to search with.
        collection (weaviate.Collection): collection of registries to search in.
        threshold_rank (int|None, optional): number of registries to return. Defaults to None.

    Returns:
        list: search results.
    """
    return collection.query.bm25(
        query["query_text"], 
        limit=threshold_rank, 
        return_metadata=weaviate.classes.query.MetadataQuery(
            score=True, 
            distance=True, 
            explain_score=True, 
            is_consistent=True, 
        ), 
    ).objects

def search_with_hybrid(query: dict[str, object], collection: weaviate.Collection, alpha: float, threshold_rank: int|None=None) -> list:
    """Search with hybrid method.

    Args:
        query (dict[str, object]): query to search with.
        collection (weaviate.Collection): collection of registries to search in.
        alpha (float): alpha parameter for hybrid search.
        threshold_rank (int|None, optional): number of registries to return. Defaults to None.

    Returns:
        list: search results.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return collection.query.hybrid(
                query["query_text"],
                limit=threshold_rank,
                alpha=alpha,
                return_metadata=weaviate.classes.query.MetadataQuery(
                    score=True,
                    distance=True,
                    explain_score=True,
                    is_consistent=True,
                ),
            ).objects
        except weaviate.exceptions.WeaviateQueryError as e:
            if "Rate limit exceeded" in str(e):
                time.sleep(2 ** attempt)
                continue
            raise
    # final attempt
    return collection.query.hybrid(
        query["query_text"],
        limit=threshold_rank,
        alpha=alpha,
        return_metadata=weaviate.classes.query.MetadataQuery(
            score=True,
            distance=True,
            explain_score=True,
            is_consistent=True,
        ),
    ).objects

def search_with_setting(query: dict[str, object], collection: weaviate.Collection, searching_setting: dict, thresholding_setting: dict) -> list:
    """Search with the specified setting.

    Args:
        query (dict[str, object]): query to search with.
        collection (weaviate.Collection): collection of registries to search in.
        setting (dict): setting for searching the registries.
        threshold_rank (int|None, optional): number of registries to return. Defaults to None.

    Returns:
        list: search results.
    """
    match searching_setting["method"]:
        case "vector":
            return search_with_vector(
                query=query, collection=collection, 
                threshold_rank=thresholding_setting.get('rank'), 
                threshold_distance=-thresholding_setting.get("distance") if thresholding_setting.get("distance") is not None else None, 
            )
        case "keyword":
            return search_with_keyword(query=query, collection=collection, threshold_rank=thresholding_setting.get('rank'))
        case "hybrid":
            return search_with_hybrid(query=query, collection=collection, alpha=searching_setting["alpha"], threshold_rank=thresholding_setting.get('rank'))
        case _:
            raise ValueError(f"Unknown searching method: {searching_setting['method']}")

def search_registries_for_query(query: dict[str, object], collection: weaviate.Collection, searching_setting: dict, thresholding_setting: dict) -> list[dict[str, object]]:
    """Search registries for a single query.

    Args:
        query (dict[str, object]): query to search with.
        collection (weaviate.Collection): collection of registries to search in.
        setting (dict): setting for searching the registries.
        threshold_rank (int|None, optional): number of registries to return. Defaults to None.
        threshold_relevance (float|None, optional): threshold_relevance parameter for vector search. Defaults to None.

    Returns:
        list[dict[str, object]]: search results.
    """
    
    def compute_relevance(metadata) -> float|None:
        if metadata.distance is not None:
            return -metadata.distance
        elif metadata.score is not None:
            return metadata.score
        else:
            return None

    return [
        {
            "query_id": query["query_id"], 
            "registry_id": search_result.properties["registry_id"], 
            "search_rank": search_rank, 
            "search_score": search_result.metadata.score, 
            "search_distance": search_result.metadata.distance, 
            # change distance to negative for consistency with scores (higher is better)
            "search_relevance": compute_relevance(search_result.metadata), 
        }
        for search_rank, search_result in enumerate(
            search_with_setting(query=query, collection=collection, searching_setting=searching_setting, thresholding_setting=thresholding_setting), 
            start=1, 
        )
        # weaviate cannot set threshold on score for keyword and hybrid searches, so we filter here
        if (
            ((search_rank <= thresholding_setting.get("rank", np.inf)) or (thresholding_setting.get("rank") is None))
            and
            ((compute_relevance(search_result.metadata) >= thresholding_setting.get("relevance", -np.inf)) or (thresholding_setting.get("relevance") is None))
        )
    ]

def search_registries_for_queries(queries: list[dict[str, object]], collection: weaviate.Collection, searching_setting: dict, thresholding_setting: dict) -> list[dict[str, object]]:
    """Search registries for multiple queries.

    Args:
        queries (list[dict[str, object]]): queries to search with.
        collection (weaviate.Collection): collection of registries to search in.
        setting (dict): setting for searching the registries.
        threshold_rank (int|None, optional): number of registries to return. Defaults to None.

    Returns:
        list[dict[str, object]]: search results.
    """
    return [
        search
        for query in queries
        for search in search_registries_for_query(
            query=query, collection=collection, 
            searching_setting=searching_setting, 
            thresholding_setting=thresholding_setting, 
        )
    ]
