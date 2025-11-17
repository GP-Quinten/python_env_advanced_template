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
from p02_deduplication.config import config
from p02_deduplication.logger import init_logger
from dotenv import load_dotenv
import time
import click

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
api_key = os.environ["MISTRAL_API_KEY"]
model_name = "mistral-embed"


@click.command()
@click.option(
    "--output_parquet",
    type=str,
    help="Output path where the parquet file will be saved.",
)
def main(output_parquet):
    # track time of execution
    start_time = time.time()
    # initialize logger
    init_logger(level="WARNING")

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
    # remove the rows with data_source_name with more than 10 words
    df_items = df_items[
        df_items["data_source_name"].apply(lambda x: len(x.split()) <= 10)
    ]
    logging.warning(
        f"Number of items after dropping missing values, not specified, and data_source_name with more than 10 words: {len(df_items)}"
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
        f"Number of items after dropping missing values, not specified, data_source_name with more than 10 words, and naïve/perfect duplicates: {len(df_items)}"
    )

    # Save to output parquet
    # make sure the directory exists
    output_dir = os.path.dirname(output_parquet)
    os.makedirs(output_dir, exist_ok=True)
    # save to parquet
    df_items.to_parquet(output_parquet, index=False)

    logging.warning("Publications embedded & uploaded")
    # end time
    end_time = time.time()
    logging.warning(f"Execution time: {round(end_time - start_time, 0)} seconds")


if __name__ == "__main__":
    main()
