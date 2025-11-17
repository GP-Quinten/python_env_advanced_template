import os
import json
import click
import logging
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from llm_inference.backends.mistral_batch import MistralBatchBackend
from llm_inference.cache.tmp import TmpCacheStorage
from llm_inference.cache.disk import DiskCacheStorage

# Load environment variables from .env file and get API key
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise EnvironmentError("MISTRAL_API_KEY not found in environment variables")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def safe_parse_response(backend, result):
    """
    Safely parse model response with error handling.
    If parsing fails, return a default empty structure.
    """
    try:
        # First try the standard parsing
        return backend._parse_response(result)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        logger.error(
            f"Problematic response content: {result.get('content', 'No content')[:500]}..."
        )

        # Try to extract content and force parse it
        try:
            content = result.get("content", "{}")
            # Try to find valid JSON by looking for opening and closing braces
            start_idx = content.find("{")
            end_idx = content.rfind("}")
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx : end_idx + 1]
                return json.loads(json_str)
        except Exception:
            pass

        # Return empty response if all parse attempts fail
        return {
            "Registry name": "Failed to parse response",
            "Population description": "Failed to parse response",
            "Intervention": "Failed to parse response",
            "Comparator": "Failed to parse response",
            "Outcome measure": "Failed to parse response",
            "Medical condition": "Failed to parse response",
            "Population sex": "Failed to parse response",
            "Population age group": "Failed to parse response",
            "Design model": "Failed to parse response",
            "Population size": "Failed to parse response",
            "Geographical area": "Failed to parse response",
            "Population follow-up": "Failed to parse response",
        }


def run_model_inference(records, model_cfg, annotation_prompt):
    """
    Run inference using the specified model configuration on the records.
    """
    # Prepare prompts for LLM inference
    prompts = []
    for record in records:
        object_id = record.get("object_id", "<unknown>")
        title = record.get("title", "<no title>")
        abstract = record.get("abstract", "<no abstract>")
        full_prompt = f"{annotation_prompt}\n\nText_to_analyze:\nTitle: {title}\nAbstract: {abstract}"
        prompts.append({"prompt": full_prompt, "custom_id": object_id})

    # Batch inference
    logger.warning(
        f"Starting batch inference with {model_cfg.get('model', 'unknown')}..."
    )
    # Use a model-specific cache directory to prevent cache conflicts between models
    backend_batch = MistralBatchBackend(
        api_key=MISTRAL_API_KEY, cache_storage=DiskCacheStorage()
    )

    inferred_results = backend_batch.infer_many(prompts, model_cfg)
    llm_responses = []

    for result in inferred_results:
        parsed_result = safe_parse_response(backend_batch, result)
        parsed_result["custom_id"] = result["custom_id"]
        llm_responses.append(parsed_result)

    logger.warning(f"Batch inference complete for {model_cfg.get('model', 'unknown')}")

    # Process results
    results = []
    for record in records:
        object_id = record.get("object_id", "<unknown>")
        resp = next((x for x in llm_responses if x["custom_id"] == object_id), None)
        if resp:
            result = {"object_id": object_id, "llm_response": resp}
            results.append(result)

    return results


@click.command()
@click.option(
    "--base_pubmed_dataset_jsonl",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to input JSONL file with base PubMed dataset",
)
@click.option(
    "--registry_related_dataset_json",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to input JSON file with registry related dataset",
)
@click.option(
    "--prompt_txt",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the annotation prompt text file",
)
@click.option(
    "--model_a_config",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the small model configuration JSON file",
)
@click.option(
    "--model_b_config",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the large model configuration JSON file",
)
@click.option(
    "--output_json",
    type=str,
    required=True,
    help="Path to output JSON file with extractions from both models",
)
def main(
    base_pubmed_dataset_jsonl,
    registry_related_dataset_json,
    prompt_txt,
    model_a_config,
    model_b_config,
    output_json,
):
    """Extract metadata from PubMed dataset using both small and large Mistral models."""
    start_time = time.time()

    # Ensure output directory exists
    out_dir = Path(output_json).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model configurations
    with open(model_a_config, "r", encoding="utf-8") as f:
        model_a_cfg = json.load(f)
    with open(model_b_config, "r", encoding="utf-8") as f:
        model_b_cfg = json.load(f)

    logger.warning(f"Loaded small model config: {model_a_cfg.get('model', 'unknown')}")
    logger.warning(f"Loaded large model config: {model_b_cfg.get('model', 'unknown')}")

    # Load the annotation prompt
    with open(prompt_txt, "r", encoding="utf-8") as f:
        annotation_prompt = f.read().strip()

    # Load registry_names_dataset_json to filter publications
    with open(registry_related_dataset_json, "r") as file:
        registry_related_dataset = json.load(file)

    # Get object_ids of registry-related publications
    registry_related_objectIDs = [
        item["object_id"]
        for item in registry_related_dataset
        if item["correct_registry_related"] == "yes"
    ]
    logger.warning(
        f"Found {len(registry_related_objectIDs)} registry-related publications"
    )

    # Load and filter PubMed records
    records = []
    with open(base_pubmed_dataset_jsonl, "r") as file:
        for line in file:
            record = json.loads(line)
            object_id = record.get("object_id", "<unknown>")
            if object_id in registry_related_objectIDs:
                records.append(record)
    logger.warning(f"Loaded and filtered to {len(records)} registry-related records")
    # test on a small subset of 5 records
    # records = records[:5]  # Uncomment for testing with a small subset
    # logger.warning(
    #     f"Using {len(records)} records for inference (limited to 5 for testing)"
    # )

    # Run inference with model A (small)
    small_model_results = run_model_inference(records, model_a_cfg, annotation_prompt)
    logger.warning(
        f"Completed small model inference on {len(small_model_results)} records"
    )

    # Run inference with model B (large)
    large_model_results = run_model_inference(records, model_b_cfg, annotation_prompt)
    logger.warning(
        f"Completed large model inference on {len(large_model_results)} records"
    )

    # Merge results from both models
    merged_results = []
    for record in records:
        object_id = record.get("object_id", "<unknown>")
        small_resp = next(
            (r for r in small_model_results if r["object_id"] == object_id), None
        )
        large_resp = next(
            (r for r in large_model_results if r["object_id"] == object_id), None
        )

        if small_resp and large_resp:
            merged_result = {
                "object_id": object_id,
                "title": record.get("title", "<no title>"),
                "abstract": record.get("abstract", "<no abstract>"),
                "small_model_response": small_resp["llm_response"],
                "large_model_response": large_resp["llm_response"],
            }
            merged_results.append(merged_result)

    logger.warning(f"Merged results for {len(merged_results)} records")

    # Save to JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(merged_results, f, indent=4, ensure_ascii=False)

    logger.warning(f"Saved merged extraction results to {output_json}")

    elapsed_total = time.time() - start_time
    logger.warning(f"Total time taken: {elapsed_total:.2f} seconds")


if __name__ == "__main__":
    main()
