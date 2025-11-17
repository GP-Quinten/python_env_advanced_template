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

MAPPING = {
    "Registry related": "registry_related",
    "Population description": "population_description",
    "Intervention": "intervention",
    "Comparator": "comparator",
    "Registry name": "registry_name",  # data_source_name
    "Outcome measure": "outcome_measure",
    "Medical condition": "medical_condition",
    "Population sex": "population_sex",
    "Population age group": "population_age_group",
    "Design model": "design_model",
    "Population size": "population_size",
    "Geographical area": "geographical_area",
    "Population follow-up": "population_follow_up",
}
# nverse mapping
INVERSE_MAPPING = {v: k for k, v in MAPPING.items()}

# NAME the columns
COL_PAIR_ID = "pair_hash_id"
COL_LABEL = "label"
LEFT_PREFIX = "left"
RIGHT_PREFIX = "right"
COL_NAME = "name"
COL_ACRONYM = "acronym"
COL_OBJECT_ID = "source_document_id"
COL_EMBEDDING = "embedding"
COL_MEDICAL_CONDITION = "medical_conditions"
COL_GEOGRAPHICAL_AREA = "geographical_areas"

# Configuration
COLUMN_CONFIG = {
    "name": {
        "left": f"{LEFT_PREFIX}_{COL_NAME}",
        "right": f"{RIGHT_PREFIX}_{COL_NAME}",
    },
    "acronym": {
        "left": f"{LEFT_PREFIX}_{COL_ACRONYM}",
        "right": f"{RIGHT_PREFIX}_{COL_ACRONYM}",
    },
    "embedding": {
        "left": f"{LEFT_PREFIX}_{COL_EMBEDDING}",
        "right": f"{RIGHT_PREFIX}_{COL_EMBEDDING}",
    },
    "medical_condition": {
        "left": f"{LEFT_PREFIX}_{COL_MEDICAL_CONDITION}",
        "right": f"{RIGHT_PREFIX}_{COL_MEDICAL_CONDITION}",
    },
    "geographical_area": {
        "left": f"{LEFT_PREFIX}_{COL_GEOGRAPHICAL_AREA}",
        "right": f"{RIGHT_PREFIX}_{COL_GEOGRAPHICAL_AREA}",
    },
}
