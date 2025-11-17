import sys
from pathlib import Path
import logging

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# print working directory
print(Path.cwd())

import weaviate
import pandas as pd
from config import config
from src.utils.io_functions import upload_file_to_s3
from sentence_transformers import SentenceTransformer

pubmedbertmodel_name = "neuml/pubmedbert-base-embeddings"
folder_path = (
    "registry_data_catalog_experiments/task_1_deduplication/publications_embeddings/"
)
file_name = "data_source_name_pubmedbert_embeddings.parquet"


def main():
    weaviate_client = weaviate.connect_to_custom(**config.WEAVIATE_PROD_CONF)
    collections = weaviate_client.collections  #
    # load publications
    collection_publications = collections.get("Publication_v2")
    # load data source names
    items = []
    for item in collection_publications.iterator(include_vector=False):
        # Extract subset of properties
        items.append(
            {
                k: v
                for k, v in item.properties.items()
                if k
                in [
                    "object_id",
                    # "title",
                    "data_source_name",
                ]
            }
        )
    # close weaviate connection
    weaviate_client.close()
    df_items = pd.DataFrame(items)

    # select only random (fixed) 30k items
    df_items = df_items.sample(n=30000, random_state=42)

    # logging numnber of items
    logging.warning(f"Number of items: {len(df_items)}")
    # drop missing data source names
    df_items.dropna(axis=0, inplace=True)
    df_items.drop(df_items[df_items["data_source_name"] == ""].index, inplace=True)
    logging.warning(f"Number of items after dropping missing values: {len(df_items)}")
    # drop not specified data source names
    df_items = df_items[
        ~df_items["data_source_name"].str.lower().str.contains("not specified")
    ]
    df_items.drop(
        df_items[df_items["data_source_name"] == "Not specified"].index, inplace=True
    )
    logging.warning(
        f"Number of items after dropping missing values and not specified: {len(df_items)}"
    )

    model_pubmed = SentenceTransformer(pubmedbertmodel_name)
    logging.warning("Embedding ...")
    df_items["embedding"] = df_items["data_source_name"].apply(
        lambda data_source_name: model_pubmed.encode(
            data_source_name, convert_to_numpy=True
        )
    )

    logging.warning("Uploading ...")
    upload_file_to_s3(
        df_items,
        config.BUCKET_NAME_DEV,
        folder_path,
        file_name,
    )

    logging.warning("Publications embedded & uploaded")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
