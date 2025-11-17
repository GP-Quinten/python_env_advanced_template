import sys
import os
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
from src.logger import init_logger
from dotenv import load_dotenv
from mistralai import Mistral
import time

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
api_key = os.environ["MISTRAL_API_KEY"]
model_name = "mistral-embed"
folder_path = (
    "registry_data_catalog_experiments/task_1_deduplication/publications_embeddings/"
)
file_name = "data_source_name_MistralEmbed_embeddings_all.parquet"


def main():
    # track time of execution
    start_time = time.time()
    # initialize logger
    init_logger(level="WARNING", file=True, file_path="logs/logs.txt")

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
    logging.warning(
        f"Number of items after dropping missing values and not specified: {len(df_items)}"
    )
    # add column "name_lower_case"
    df_items["name_lower_case"] = df_items["data_source_name"].str.lower()
    # show value counts top5
    logging.warning(df_items["name_lower_case"].value_counts().head(5))
    # deduplicates (keep first) on data_source_name.lower(), then remove the column name_lower_case
    df_items = df_items.drop_duplicates(subset=["name_lower_case"], keep="first").drop(
        columns=["name_lower_case"]
    )
    # count after remove duplicates
    logging.warning(
        f"Number of items after dropping missing values, not specified and duplicates: {len(df_items)}"
    )

    # remove the rows with data_source_name with more than 10 words
    df_items = df_items[
        df_items["data_source_name"].apply(lambda x: len(x.split()) <= 10)
    ]
    logging.warning(
        f"Number of items after dropping missing values, not specified, duplicates and data_source_name with more than 10 words: {len(df_items)}"
    )

    # # select only random (fixed) 30k items
    # df_items = df_items.sample(n=30000, random_state=42)

    # Initialize the Mistral client
    client = Mistral(api_key=api_key)

    logging.warning("Embedding ...")

    def get_embeddings_by_chunks(data, chunk_size):
        chunks = [data[x : x + chunk_size] for x in range(0, len(data), chunk_size)]
        embeddings_response = [
            client.embeddings.create(model=model_name, inputs=c) for c in chunks
        ]
        return [d.embedding for e in embeddings_response for d in e.data]

    df_items["embedding"] = get_embeddings_by_chunks(
        df_items["data_source_name"].tolist(), 1000
    )

    logging.warning("Uploading ...")
    upload_file_to_s3(
        df_items,
        config.BUCKET_NAME_DEV,
        folder_path,
        file_name,
    )

    logging.warning("Publications embedded & uploaded")
    # end time
    end_time = time.time()
    logging.warning(f"Execution time: {round(end_time - start_time, 0)} seconds")


if __name__ == "__main__":
    main()
