#!/usr/bin/env python3
import os
import time
import logging
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import click
import asyncio

from dotenv import load_dotenv

# Import your LLM backend and cache
import llm_backends
from llm_inference.cache.tmp import TmpCacheStorage

# Logging setup
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Load environment variables from .env file
load_dotenv()


# Async inference
async def run_async_inference(prompts, backend, model_cfg):
    start_time = time.time()
    raw_responses = []
    pbar = tqdm(total=len(prompts), desc="LLM Inference")
    for prompt in prompts:
        raw_response = await backend.infer_one(prompt, model_cfg)
        raw_response["custom_id"] = prompt["custom_id"]
        raw_responses.append(raw_response)
        pbar.update(1)
    pbar.close()
    elapsed = (time.time() - start_time) / 60
    logger.warning(f"Elapsed time for async inference: {elapsed:.1f} minutes")
    return raw_responses


@click.command()
@click.option(
    "--input_pairs_xlsx",
    type=str,
    required=True,
    help="Input Excel file with registry name pairs.",
)
@click.option(
    "--prompt_txt",
    type=str,
    required=True,
    help="Prompt template file.",
)
@click.option(
    "--model_config",
    type=str,
    required=True,
    help="Model config JSON file.",
)
@click.option(
    "--output_assessed_pairs_xlsx",
    type=str,
    required=True,
    help="Output Excel file for LLM-assessed pairs.",
)
def eval_pairs_similarity_assessment_with_llm(
    input_pairs_xlsx,
    prompt_txt,
    model_config,
    output_assessed_pairs_xlsx,
):
    start_time = time.time()
    logger.warning("Loading pairs for LLM assessment...")

    df_pairs = pd.read_excel(input_pairs_xlsx)
    # # limit to N pairs for testing
    # df_pairs = df_pairs.head(500)

    # Read prompt template
    with open(prompt_txt, "r") as pf:
        base_prompt = pf.read().strip()

    # Load model config
    with open(model_config, "r", encoding="utf-8") as f:
        model_cfg = json.load(f)

    # Build prompts
    def construct_prompt(base_prompt: str, ds1: str, ds2: str) -> str:
        return base_prompt.replace("{{content_a}}", ds1).replace("{{content_b}}", ds2)

    prompts = []
    for idx, row in df_pairs.iterrows():
        ds1 = row["full_name"]
        ds2 = row["alias"]
        prompt_dict = {
            "prompt": construct_prompt(base_prompt, ds1, ds2),
            "custom_id": row["object_id"],
        }
        prompts.append(prompt_dict)
    logger.warning(f"Built {len(prompts)} prompts.")

    # Setup LLM backend
    cache_storage = TmpCacheStorage()
    backend = llm_backends.OpenAIAsyncBackend(
        api_key=os.getenv("OPENAI_API_KEY"), cache_storage=cache_storage
    )

    logger.warning("Starting asynchronous LLM inference...")
    loop = asyncio.get_event_loop()
    # raw_responses = await run_async_inference(prompts, backend, model_cfg)
    raw_responses = asyncio.run(run_async_inference(prompts, backend, model_cfg))

    # Post-processing
    prompt_map = {p["custom_id"]: p for p in prompts}
    llm_responses = []
    prompt_response_records = []

    for raw_response in tqdm(raw_responses, desc="Processing responses", leave=False):
        custom_id = raw_response.get("custom_id", "")
        prompt_obj = prompt_map.get(custom_id)
        if prompt_obj:
            prompt_response_records.append(
                {
                    "custom_id": custom_id,
                    "prompt": prompt_obj["prompt"],
                    "llm_response": raw_response,
                }
            )
            parsed_response = backend._parse_response(raw_response)
            parsed_response["custom_id"] = custom_id
            llm_responses.append(parsed_response)
    logging.warning("Batch inference complete on full dataset")

    # Assign columns to df_pairs
    for result in tqdm(llm_responses, desc="Assigning LLM results", leave=False):
        custom_id = result.get("custom_id", "")
        explanation = result.get("explanation", "")
        final_decision = result.get("final_decision", "uncertain")
        # Find the corresponding prompt response record
        prompt_response = next(
            (pr for pr in prompt_response_records if pr["custom_id"] == custom_id), None
        )
        if prompt_response:
            # Extract the explanation from the LLM response
            explanation = result.get("explanation", "")

            # Determine the final decision based on the LLM response
            final_decision = result.get("final_decision", "uncertain")

            # Add the new columns to the DataFrame
            df_pairs.loc[df_pairs.object_id == custom_id, "explanation"] = explanation
            df_pairs.loc[df_pairs.object_id == custom_id, "final_label"] = (
                1 if final_decision == "same" else 0
            )
            df_pairs.loc[df_pairs.object_id == custom_id, "custom_id"] = custom_id
            df_pairs.loc[df_pairs.object_id == custom_id, "uncertain"] = result.get(
                "uncertain", False
            )

    # Reorder columns if needed
    cols = [
        "object_id",
        # "alias_object_id",
        "full_name",
        "alias",
        "number_of_occurrences",
        # "alias_number_of_occurrences",
        "similarity",
        "final_label",
        "uncertain",
        "explanation",
    ]
    df_pairs = df_pairs[[c for c in cols if c in df_pairs.columns]]

    # Save output
    output_path = Path(output_assessed_pairs_xlsx).parent
    output_path.mkdir(parents=True, exist_ok=True)
    df_pairs.to_excel(output_assessed_pairs_xlsx, index=False)
    logger.warning(f"Saved LLM-assessed pairs to {output_assessed_pairs_xlsx}")

    # Logg % of positive and negative pairs 'final_label'
    total_pairs = len(df_pairs)
    # convert final_label to int if it's not already
    df_pairs["final_label"] = df_pairs["final_label"].astype(int)
    # logg % of positive pairs
    logger.warning(
        f"Percentage of positive pairs (final_label=1): "
        f"{df_pairs['final_label'].sum() / total_pairs * 100:.2f}%"
    )
    # logg % of uncertain
    # uncertain has value "yes" or "no" -> 1 or 0
    df_pairs["uncertain"] = df_pairs["uncertain"].map({"yes": 1, "no": 0})
    uncertain_pairs = df_pairs["uncertain"].sum()
    logger.warning(
        f"Percentage of uncertain pairs: " f"{uncertain_pairs / total_pairs * 100:.2f}%"
    )

    elapsed = (time.time() - start_time) / 60
    logger.warning(f"Total time taken: {elapsed:.2f} minutes.")


if __name__ == "__main__":
    eval_pairs_similarity_assessment_with_llm()
