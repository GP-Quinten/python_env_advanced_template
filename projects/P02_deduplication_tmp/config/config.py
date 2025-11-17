COLLECTION_NAME_PUBLICATIONS = "Publication_v2"
BUCKET_NAME_DEV = "s3-common-dev20231214174437248800000002"
BUCKET_NAME_PROD = "s3-common-prod20231214174437251600000003"
PUBLICATION_DATA_PATH = f"{COLLECTION_NAME_PUBLICATIONS}/data"
PUBLICATION_EXPLO_PATH = f"{COLLECTION_NAME_PUBLICATIONS}/exploration"
WEAVIATE_DEV_CONF = {
    "http_host": "weaviate-mvp.eu-more-europa-gpu.quinten.io",
    "http_port": 443,
    "http_secure": True,
    "grpc_host": "10.23.3.69",
    "grpc_port": 3052,
    "grpc_secure": False,
    "skip_init_checks": False,
}

WEAVIATE_PROD_CONF = {
    "http_host": "weaviate-new.eu-more-europa-gpu.quinten.io",
    "http_port": 443,
    "http_secure": True,
    "grpc_host": "10.23.3.69",
    "grpc_port": 3051,
    "grpc_secure": False,
    "skip_init_checks": False,
}
