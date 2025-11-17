import os
import weaviate

from dotenv import load_dotenv

load_dotenv()



##### PATHS #####
CURRENT_FILE = os.path.abspath(__file__)

# Go up 3 levels: p06_search_engine/src/p06_search_engine -> project root
PATH_PROJECT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(CURRENT_FILE)
    )
)

PATH_DATA = f"{PATH_PROJECT}/data"

PATH_DATA_1ST = f"{PATH_DATA}/1st_dataset"
PATH_DATA_1ST_QUERIES = f"{PATH_DATA_1ST}/queries.json"
PATH_DATA_1ST_ANNOTATIONS = f"{PATH_DATA_1ST}/annotations.json"

PATH_DATA_2ND = f"{PATH_DATA}/2nd_dataset"
PATH_DATA_2ND_QUERIES = f"{PATH_DATA_2ND}/queries.json"
PATH_DATA_2ND_ANNOTATIONS = f"{PATH_DATA_2ND}/annotations.json"

PATH_DATA_REGISTRIES = f"{PATH_DATA}/registries.json"

PATH_DATA_SEARCHES = f"{PATH_DATA}/searches"



### CREDENTIALS ###

# Mistral
MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]

print(f"MISTRAL_API_KEY: {MISTRAL_API_KEY[:8]}...{MISTRAL_API_KEY[-8:]} (len={len(MISTRAL_API_KEY)})")

### MLFlow ###

MLFLOW_URI = "file:mlruns/"
MLFLOW_EXPERIMENT_NAME = "search_engine"



### SETTINGS ###

PREPARING_SETTINGS_SPACE = {
    "name": [True], 
    "acronym": [False, True], 
    #"locations": [0, 1, 3, 5], 
    "conditions": [0, 1, 3, 5], 
}

SEARCHING_SETTINGS_SPACE = {
    "vector": {
        "distance": [weaviate.classes.config.VectorDistances.COSINE], 
    }, 
    "keyword": {
        "tokenization": [weaviate.classes.config.Tokenization.WORD, weaviate.classes.config.Tokenization.TRIGRAM], 
    }, 
    "hybrid": {
        "distance": [weaviate.classes.config.VectorDistances.COSINE], 
        "tokenization": [weaviate.classes.config.Tokenization.TRIGRAM], 
        "alpha": [0.25, 0.5, 0.75], 
        "fusion_type": [weaviate.classes.query.HybridFusion.RELATIVE_SCORE, weaviate.classes.query.HybridFusion.RANKED], 
    }, 
}


METRICS_FOR_BEST_PARAMS = "thresholding_f1"
