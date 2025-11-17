import weaviate
from weaviate import WeaviateAsyncClient
from weaviate.connect import ConnectionParams


collection_name = "RegistryCollection"


async def get_async_client(http_host, http_port, grpc_host, grpc_port, mistral_api_key) -> WeaviateAsyncClient:
    """
    Minimal async example: connect to Weaviate with required arguments.
    """
    client = WeaviateAsyncClient(
        connection_params=ConnectionParams.from_params(
            http_host=http_host,
            http_port=int(http_port),
            http_secure=False,
            grpc_host=grpc_host,
            grpc_port=int(grpc_port),
            grpc_secure=False,
        ),
        additional_headers={"X-Mistral-Api-Key": mistral_api_key},
    )

    await client.connect()
    if not await client.is_ready():
        raise RuntimeError("Weaviate client is not ready")

    return client

def get_connection(http_host, http_port, grpc_host, grpc_port, mistral_api_key) -> weaviate.Client:
    """Connect to Weaviate using provided arguments (no fallbacks).

    Args:
        http_host (str): HTTP host for Weaviate.
        http_port (int): HTTP port for Weaviate.
        grpc_host (str): gRPC host for Weaviate.
        grpc_port (int): gRPC port for Weaviate.
        mistral_api_key (str, optional): Mistral API key for header. 
    """
    if not all([http_host, http_port, grpc_host, grpc_port]):
        raise RuntimeError(
            "http_host, http_port, grpc_host and grpc_port must be provided"
        )

    http_port = int(http_port)
    grpc_port = int(grpc_port)

    headers = {"X-Mistral-Api-Key": mistral_api_key}

    client = weaviate.connect_to_custom(
        http_host=http_host,
        http_port=http_port,
        http_secure=False,
        grpc_host=grpc_host,
        grpc_port=grpc_port,
        grpc_secure=False,
        headers=headers,
    )

    if not client.is_ready():
        raise RuntimeError("Weaviate client is not ready")

    return client