import sys
import os
from pathlib import Path
import logging
import boto3
import tqdm
import json

import llm_backends
from llm_backends.cache import DiskCacheStorage

# print working directory
print(Path.cwd())

import pandas as pd
from dotenv import load_dotenv
from mistralai import Mistral
import time
import click

from src.p05_refine_dedup import config
from src.p05_refine_dedup.utils.s3_io_functions import (
    load_jsonl_from_s3,
    upload_parquet_to_s3,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
api_key = os.environ["MISTRAL_API_KEY"]
MISTRAL_EMBEDDING_MODEL = "mistral-embed"
MISTRAL_EMBEDDING_CONFIG = {
    "model": MISTRAL_EMBEDDING_MODEL,
}


@click.command()
@click.option(
    "--input_ema_reg_json",
    type=str,
    help="Path to the input EMA registry JSON file.",
)
@click.option(
    "--local_output_embeddings_parquet",
    type=str,
    help="Local output file template for the embeddings in JSONL format.",
)
@click.option(
    "--s3_output_embeddings_parquet",
    type=str,
    help="S3 directory where the output embeddings will be stored.",
)
def embed_registry_names(
    input_ema_reg_json,
    local_output_embeddings_parquet,
    s3_output_embeddings_parquet,
):
    # track time of execution
    start_time = time.time()

    # 1. Load registry data
    # Retrieve total_batches = how many files are in the folder
    # reload the file to check
    with open(input_ema_reg_json, "r", encoding="utf-8") as f:
        registry_dataset = json.load(f)

    # # test
    # registry_dataset = registry_dataset[:5]
    # logger.warning(f"Using a subset of {len(registry_dataset)} records for testing.")

    # 2. Initialize the backend for Mistral embeddings
    backend = llm_backends.MistralEmbeddingBackend(
        api_key=os.getenv("MISTRAL_API_KEY"), cache_storage=DiskCacheStorage()
    )

    # 3. Prepare prompts for embeddings
    # for all registries, create a new field "full_name"
    for registry in registry_dataset:
        registry_name = registry.get("registry_name", "")
        acronym = registry.get("acronym", "")
        if acronym:
            full_name = f"{registry_name} ({acronym})"
        else:
            full_name = registry_name
        registry["full_name"] = full_name

    prompt_items = [
        {"custom_id": registry["object_id"], "prompt": registry["full_name"]}
        for registry in registry_dataset
    ]
    batch_prompts_list = []
    batch_size = 100
    for i in range(0, len(prompt_items), batch_size):
        batch_prompts_list.append(prompt_items[i : i + batch_size])
    # print how many batches we have
    print(f"Total batches created: {len(batch_prompts_list)}")

    # 4. Run inference to get embeddings
    start_time = time.time()
    for batch_prompts in tqdm.tqdm(batch_prompts_list, desc="Processing batches"):

        batch_responses = backend.infer_many(
            prompt_items=batch_prompts, model_config=MISTRAL_EMBEDDING_CONFIG
        )
        for result_item in batch_responses:
            for registry in registry_dataset:
                if registry["object_id"] == result_item["custom_id"]:
                    registry["full_name_embedding"] = result_item["embedding"]
        intermediate_time = time.time()
        logger.warning(
            f"Processed batch in {intermediate_time - start_time:.0f} seconds"
        )
    end_time = time.time()
    # logg time in minutes
    logger.warning(
        f"Total time for inference: {(end_time - start_time) / 60:.1f} minutes"
    )

    # 5. Save locally then on s3 bucket in parquet format
    # Save to local file using to_parquet
    # convert to dataframe
    output_df = pd.DataFrame(registry_dataset)[
        ["object_id", "full_name", "full_name_embedding", "number_of_occurrences"]
    ]
    output_df.to_parquet(local_output_embeddings_parquet, index=False)
    logger.warning(f"Saved embeddings to local file: {local_output_embeddings_parquet}")
    # Upload to S3 with upload_parquet_to_s3
    # split s3_output_embeddings_parquet in folder_path and file name
    s3_folder_path = os.path.dirname(s3_output_embeddings_parquet)
    s3_file_name = os.path.basename(s3_output_embeddings_parquet)
    upload_parquet_to_s3(
        output_df, config.BUCKET_NAME_DEV, s3_folder_path, s3_file_name
    )


if __name__ == "__main__":
    embed_registry_names()
