import weaviate

from . import preparing, searching



# Collection

def generate_registries_collection_name(setting_id: str) -> str:
    """Generate the name of the collection of registries.

    Args:
        preparing_setting (dict): setting for preparing the registries.
        searching_setting (dict): setting for searching the registries.

    Returns:
        str: name of the collection of registries.
    """

    return f"Registry___{setting_id}"

def get_registries_collection(client: weaviate.Client, collection_name: str) -> weaviate.Collection:
    """Get the collection of registries.

    Args:
        client (weaviate.Client): Weaviate client.
        collection_name (str): name of the collection of registries to get.

    Returns:
        weaviate.Collection: collection of registries.
    """
    return client.collections.get(collection_name)

def create_registries_collection(client: weaviate.Client, collection_name: str, searching_setting: dict) -> weaviate.Collection:
    """Create the collection of registries.

    Args:
        client (weaviate.Client): Weaviate client.
        collection_name (str): name of the collection of registries.
        searching_setting (dict): setting for searching the registries.

    Returns:
        weaviate.Collection: collection of registries.
    """

    return client.collections.create(
        name=collection_name, 
        vector_config=weaviate.classes.config.Configure.Vectors.text2vec_mistral(
            vectorize_collection_name=False, 
            source_properties=["registry_name", "registry_acronym", "registry_locations", "registry_conditions"], 
            vector_index_config=weaviate.classes.config.Configure.VectorIndex.flat(
                distance_metric=searching_setting.get("distance"), 
            ), 
        ), 
        properties=[
            weaviate.classes.config.Property(
                name="registry_id", 
                description="Id of the registry", 
                data_type=weaviate.classes.config.DataType.INT, 
                skip_vectorization=True, 
            ), 
            weaviate.classes.config.Property(
                name="registry_name", 
                description="Name of the registry", 
                data_type=weaviate.classes.config.DataType.TEXT, 
                vectorize_property_name=False, 
                tokenization=searching_setting.get("tokenization"), 
            ), 
            weaviate.classes.config.Property(
                name="registry_acronym", 
                description="Acronym of the registry", 
                data_type=weaviate.classes.config.DataType.TEXT, 
                vectorize_property_name=False, 
                tokenization=searching_setting.get("tokenization"), 
            ), 
            weaviate.classes.config.Property(
                name="registry_locations", 
                description="Locations of the registry", 
                data_type=weaviate.classes.config.DataType.TEXT_ARRAY, 
                vectorize_property_name=False, 
                tokenization=searching_setting.get("tokenization"), 
            ), 
            weaviate.classes.config.Property(
                name="registry_conditions", 
                description="Conditions of the registry", 
                data_type=weaviate.classes.config.DataType.TEXT_ARRAY, 
                vectorize_property_name=False, 
                tokenization=searching_setting.get("tokenization"), 
            ), 
        ], 
    )

def get_or_create_registries_collection(client: weaviate.Client, collection_name: str, searching_setting: dict) -> weaviate.Collection:
    """Get or create the collection of registries.

    Args:
        client (weaviate.Client): Weaviate client.
        collection_name (str): name of the collection of registries to get or create.

    Returns:
        weaviate.Collection: collection of registries.
    """

    # get collection
    if client.collections.exists(collection_name):
        return get_registries_collection(
            client=client, 
            collection_name=collection_name, 
        )

    # create collection
    return create_registries_collection(
        client=client, 
        collection_name=collection_name, 
        searching_setting=searching_setting, 
    )

# # Objects

def upload_registries(collection: weaviate.Collection, registries: list[dict[str, object]], batch_size: int=100) -> None:
    """Upload registries to the collection.

    Args:
        collection (weaviate.Collection): collection of registries to upload to.
        registries (list[dict[str, object]]): registries to upload.
        batch_size (int, optional): size of the batch to upload. Defaults to 100.
    """

    # upload registries per batches
    with collection.batch.fixed_size(batch_size=batch_size) as batch:

        # iterate over registries
        for registry in registries:

            # add registry to batch
            batch.add_object(
                uuid=weaviate.util.generate_uuid5("Registry", registry["registry_id"]), 
                properties=registry, 
            )

            # check for errors
            if batch.number_errors > 0:
                raise RuntimeError
