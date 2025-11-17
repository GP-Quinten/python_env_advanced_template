import sys
import os
from pathlib import Path
import logging

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# print working directory
print(Path.cwd())

import pandas as pd
from p02_deduplication.logger import init_logger
from dotenv import load_dotenv
from mistralai import Mistral
import time
import click

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
api_key = os.environ["MISTRAL_API_KEY"]
model_name = "mistral-embed"


@click.command()
@click.option(
    "--remaining_data_source_names",
    "-i",
    type=str,
    required=True,
    help="Input parquet file path: 90k remaining data source names.",
)
@click.option(
    "--embeddings_all",
    type=str,
    help="Output path where the parquet file will be saved.",
)
@click.option(
    "--embeddings_sample",
    type=str,
    help="Output path where only a sample (30k) parquet file will be saved.",
)
def main(remaining_data_source_names, embeddings_all, embeddings_sample):
    # track time of execution
    start_time = time.time()
    # initialize logger
    init_logger(level="WARNING")

    # load data
    df_items = pd.read_parquet(remaining_data_source_names)

    # Initialize the Mistral client
    client = Mistral(api_key=api_key)
    print(client)

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
    df_items.to_parquet(embeddings_all, index=False)

    # upload sample
    df_items.sample(30000).to_parquet(embeddings_sample, index=False)

    logging.warning("Publications embedded & uploaded")
    # end time
    end_time = time.time()
    logging.warning(f"Execution time: {round(end_time - start_time, 0)} seconds")


if __name__ == "__main__":
    main()
