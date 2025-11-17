#!/usr/bin/env python3
import os
import sys
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import logging
import time
import pandas as pd
from tqdm import tqdm
import click

# Set project root and update system path for local modules
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from p02_deduplication.logger import init_logger
from llm_inference.backends.mistral_async import MistralAsyncBackend
from llm_inference.backends.mistral_batch import MistralBatchBackend
from llm_inference.cache.tmp import TmpCacheStorage
from llm_inference.cache.disk import DiskCacheStorage

# Load environment variables from .env file and get API key
load_dotenv()
api_key = os.environ["MISTRAL_API_KEY"]

# Load large Mistral model configuration
config_path = Path("etc/configs/large_mistral_config.json")
with open(config_path, "r") as f:
    model_config = json.load(f)


# Define the prompt template using hybrid approach
def construct_prompt(base_prompt: str, ds1: str, ds2: str) -> str:
    """
    Constructs the final prompt by replacing placeholders with provided dataset strings.

    Args:
        base_prompt (str): The base prompt containing placeholders "{{content_a}}" and "{{content_b}}".
        ds1 (str): The string to replace "{{content_a}}".
        ds2 (str): The string to replace "{{content_b}}".

    Returns:
        str: The final prompt with the placeholders replaced.
    """
    # Replace the placeholders with the provided strings.
    prompt = base_prompt.replace("{{content_a}}", ds1).replace("{{content_b}}", ds2)
    return prompt


@click.command()
@click.option(
    "--pairs_parquet",
    type=str,
    help="Path to the parquet file containing pairs data",
)
@click.option(
    "--prompt",
    type=str,
    help="Path to the prompt file",
)
@click.option(
    "--output_eval_dataset",
    type=str,
    help="Path to the output evaluation dataset",
)
@click.option(
    "--track_category_switches",
    type=bool,
    default=False,
    help="Flag to track category switches",
)
def main(pairs_parquet, prompt, output_eval_dataset, track_category_switches):
    start_time = time.time()
    # Initialize logger (logging warnings for major steps, info for iteration progress)
    init_logger(level="WARNING")
    logging.warning("Starting LLM Inference for Pairing Assignment")

    # Load df_pairs
    df_pairs = pd.read_parquet(pairs_parquet)
    # For testing purposes, we sample 1000 pairs
    # df_pairs = df_pairs.sample(n=1000, random_state=42)
    logging.warning("Loaded df_pairs")

    # Log initial ratio of positive vs. negative pairs
    pos_count = (df_pairs["label"] == 1).sum()
    neg_count = (df_pairs["label"] == 0).sum()
    logging.warning(
        f"Initial pairs: {pos_count} positive, {neg_count} negative (ratio {pos_count/len(df_pairs):.2f})"
    )

    # ----- BUILD PROMPTS -----
    # build prompts
    prompts = []
    total = len(df_pairs)
    # logg Building prompts
    logging.info(f"Building prompts for {total} pairs")
    # Read the base prompt from file (we expect the file to contain the base prompt content)
    prompt_file_path = Path(prompt)
    with open(prompt_file_path, "r") as pf:
        base_prompt = pf.read().strip()
    for idx, row in df_pairs.iterrows():
        ds1 = row["data_source_name_1"]
        ds2 = row["data_source_name_2"]
        # prompts = [{"prompt": f"{i}. Tell me a story.", "custom_id": i} for i in range(num_requests)]
        prompt_dict = {
            "prompt": construct_prompt(base_prompt, ds1, ds2),
            "custom_id": idx,
        }
        prompts.append(prompt_dict)

    # ----- BATCH INFERENCE (FULL DATASET) -----
    logging.warning("Starting batch inference on full dataset...")
    backend_batch = MistralBatchBackend(
        api_key=api_key, cache_storage=DiskCacheStorage(subdir="mistral_batch_cache")
    )
    # Perform batch inference
    inferred_results = backend_batch.infer_many(prompts, model_config)
    results = []
    for result in inferred_results:
        parsed_result = backend_batch._parse_response(result)
        parsed_result["custom_id"] = result["custom_id"]
        results.append(parsed_result)
    logging.warning("Batch inference complete on full dataset")

    # ----- POST-PROCESSING -----
    # Assign 3 columns to test_sample: "explanation", "final_decision", "custom_id"
    for result in results:
        idx = int(result["custom_id"])
        df_pairs.at[idx, "final_decision"] = result["final_decision"]
        # create a new column "final_label" in the test sample (if final_decision == 'same' -> 1, elif 'different' -> 0)
        df_pairs.at[idx, "final_label"] = 1 if result["final_decision"] == "same" else 0
        df_pairs.at[idx, "explanation"] = result["explanation"]
    # Drop the final_decision column
    df_pairs.drop(columns=["final_decision"], inplace=True)

    # ----- ANALYSIS -----
    # Log ratio changes in the test sample
    orig_pos = (df_pairs["label"] == 1).sum()
    orig_neg = (df_pairs["label"] == 0).sum()
    final_pos = (df_pairs["final_label"] == 1).sum()
    final_neg = (df_pairs["final_label"] == 0).sum()
    logging.warning(
        f"Test sample before inference: {orig_pos} positive, {orig_neg} negative"
    )
    logging.warning(
        f"Test sample after inference: {final_pos} positive, {final_neg} negative"
    )

    # Log modifications: count where positive became negative and vice versa
    pos_to_neg = ((df_pairs["label"] == 1) & (df_pairs["final_label"] == 0)).sum()
    neg_to_pos = ((df_pairs["label"] == 0) & (df_pairs["final_label"] == 1)).sum()
    logging.warning(
        f"Test sample modifications: {pos_to_neg} changed from positive to negative, {neg_to_pos} changed from negative to positive"
    )

    # Log global ratio change (using final_label)
    global_final_pos = (df_pairs["final_label"] == 1).sum()
    global_final_neg = (df_pairs["final_label"] == 0).sum()
    logging.warning(
        f"Global final ratio (based on test sample updates): {global_final_pos} positive, {global_final_neg} negative"
    )

    # ----- CREATE PIVOT TABLE -----
    # create new column diff to show the difference between the original label and the final label
    df_pairs["diff"] = df_pairs["final_label"] - df_pairs["label"]
    # create df_pairs_pivot by pivoting on the column 'diff'
    df_pairs_pivot = df_pairs.pivot_table(
        index="category", columns="diff", values="label", aggfunc="count"
    )
    df_pairs_pivot = (
        df_pairs_pivot.fillna(0).astype(int).sort_values(by=-1, ascending=False)
    )
    # rename columns 1 -> N_to_P, -1 -> P_to_N, 0 -> same. if a column is not present, you must add it and set all rows to 0
    df_pairs_pivot.rename(columns={1: "N_to_P", -1: "P_to_N", 0: "same"}, inplace=True)

    # add a column to_include
    # create a column 'to_include' with value 'Yes' or 'No' if 'explanation' starts with 'Inclusion-uncertainty'
    df_pairs["to_include"] = df_pairs.apply(
        lambda row: (
            "No"
            if row["explanation"].startswith("Inclusion-uncertainty")
            and row["diff"] != 0
            else "Yes"
        ),
        axis=1,
    )

    # ----- SAVE DATA -----
    # Save final pairs dataset
    logging.warning("Uploading final evaluation dataset ...")
    # Save df_pairs to parquet file locally (output_eval_dataset)
    local_parquet_path = Path(output_eval_dataset)
    local_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df_pairs.to_parquet(local_parquet_path, index=False)
    logging.warning("Upload completed.")

    # save df_pairs_pivot as track_category_switches file path
    track_category_switches_path = Path(track_category_switches)
    track_category_switches_path.parent.mkdir(parents=True, exist_ok=True)
    df_pairs_pivot.to_excel(track_category_switches_path, index=True, header=True)
    elapsed = time.time() - start_time
    logging.warning(f"Script completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
