import os

import weaviate

from dotenv import load_dotenv
load_dotenv()


def connect_to_weaviate(*args, **kwargs):
    config = {
        "http_host": os.getenv("WEAVIATE__HTTP_HOST"),
        "http_port": os.getenv("WEAVIATE__HTTP_PORT"),
        "http_secure": True,
        "grpc_host": os.getenv("WEAVIATE__GRPC_HOST"),
        "grpc_port": os.getenv("WEAVIATE__GRPC_PORT"),
        "grpc_secure": False,
        "skip_init_checks": True,
    }
    config.update(kwargs)

    weaviate_client = weaviate.connect_to_custom(**config)

    return weaviate_client
