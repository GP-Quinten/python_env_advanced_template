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
from llm_inference.backends.openai_batch import OpenAIBatchBackend
from llm_inference.cache.tmp import TmpCacheStorage
from llm_inference.cache.disk import DiskCacheStorage

# Load environment variables from .env file and get API keys
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_mistral_response(response):
    """
    Parse Mistral response to extract the specific field.
    """
    try:
        parsed_response = response
        parsed_response["custom_id"] = response.get("custom_id", "")
        return parsed_response
    except Exception as e:
        logger.error(f"Error parsing Mistral response: {e}")
        logger.error(f"Original response: {response}")
        return {
            "error": "Failed to parse response",
            "custom_id": response.get("custom_id", ""),
        }


def parse_openai_response(response):
    """
    Parse OpenAI response to extract the specific field.
    """
    try:
        # Extract the content from the OpenAI response structure
        content = response["choices"][0]["message"]["content"]

        # Parse the content as JSON
        parsed_content = json.loads(content)

        # Add custom_id to the parsed content
        parsed_content["custom_id"] = response.get("custom_id", "")

        return parsed_content
    except Exception as e:
        logger.error(f"Error parsing OpenAI response: {e}")
        logger.error(f"Original response: {response}")
        return {
            "error": "Failed to parse response",
            "custom_id": response.get("custom_id", ""),
        }


@click.command()
@click.option(
    "--base_pubmed_dataset_jsonl",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to input JSONL file with base PubMed dataset",
)
@click.option(
    "--registry_related_eval_dataset_json",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to input JSON file with registry names dataset",
)
@click.option(
    "--model_config",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the model configuration JSON file",
)
@click.option(
    "--prompt_field",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the field-specific prompt text file",
)
@click.option(
    "--model",
    type=str,
    required=True,
    help="Model to use for inference (e.g., 'small_mistral', 'large_mistral')",
)
@click.option(
    "--field",
    type=str,
    required=True,
    help="Field to extract (e.g., 'medical_condition', 'outcome_measure')",
)
@click.option(
    "--output_field_json",
    type=str,
    required=True,
    help="Path to output JSON file with field-specific extractions",
)
@click.option(
    "--output_field_records_jsonl",
    type=str,
    required=True,
    help="Path to output JSONL file with one line per record (object_id, prompt, response) for this field",
)
def main(
    base_pubmed_dataset_jsonl,
    registry_related_eval_dataset_json,
    model_config,
    prompt_field,
    model,
    field,
    output_field_json,
    output_field_records_jsonl,
):
    """Extract a specific metadata field from PubMed dataset using the specified model."""
    start_time = time.time()

    # Ensure output directory exists
    out_dir = Path(output_field_json).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure output records directory exists
    records_dir = Path(output_field_records_jsonl).parent
    records_dir.mkdir(parents=True, exist_ok=True)

    # Load model configuration
    with open(model_config, "r", encoding="utf-8") as f:
        model_cfg = json.load(f)

    model_name = model_cfg.get("model", "unknown")
    logger.warning(f"Using model: {model_name} to extract field: {field}")

    # Determine if we're using OpenAI or Mistral model
    is_openai_model = "gpt" in model_name.lower() or "o3" in model_name.lower()

    # Check API keys
    if is_openai_model and not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY not found in environment variables")
    elif not is_openai_model and not MISTRAL_API_KEY:
        raise EnvironmentError("MISTRAL_API_KEY not found in environment variables")

    # Load the field-specific prompt
    with open(prompt_field, "r", encoding="utf-8") as f:
        field_prompt = f.read().strip()

    # Load registry_names_dataset_json to filter publications
    with open(registry_related_eval_dataset_json, "r") as file:
        registry_related_eval_dataset = json.load(file)

    # Get object_ids of registry-related publications
    registry_related_objectIDs = [
        item["object_id"]
        for item in registry_related_eval_dataset
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

    # Prepare prompts for LLMs
    prompts = []
    for rec in records:
        object_id = rec.get("object_id", "<unknown>")
        title = rec.get("title", "<no title>")
        abstract = rec.get("abstract", "<no abstract>")
        full_prompt = (
            f"{field_prompt}\nText_to_analyze:\nTitle: {title}\nAbstract: {abstract}"
        )
        prompts.append({"prompt": full_prompt, "custom_id": object_id})

    # Create a list to store the records with object_id, prompt, and raw response
    prompt_response_records = []

    # Run batch inference based on model type
    logger.warning(f"Starting batch inference with {model_name} for field {field}...")

    llm_responses = []

    if is_openai_model:
        backend = OpenAIBatchBackend(
            api_key=OPENAI_API_KEY, cache_storage=DiskCacheStorage()
        )
        inferred_results = backend.infer_many(prompts, model_cfg)

        for result in inferred_results:
            # Store the raw response with object_id and prompt for the records file
            prompt_obj = next(
                (p for p in prompts if p["custom_id"] == result["custom_id"]), None
            )
            if prompt_obj:
                prompt_response_records.append(
                    {
                        "object_id": result["custom_id"],
                        "prompt": prompt_obj["prompt"],
                        "llm_response": result,
                    }
                )

            parsed_result = parse_openai_response(result)
            llm_responses.append(parsed_result)
    else:
        backend = MistralBatchBackend(
            api_key=MISTRAL_API_KEY, cache_storage=DiskCacheStorage()
        )
        inferred_results = backend.infer_many(prompts, model_cfg)

        for result in inferred_results:
            # Store the raw response with object_id and prompt for the records file
            prompt_obj = next(
                (p for p in prompts if p["custom_id"] == result["custom_id"]), None
            )
            if prompt_obj:
                prompt_response_records.append(
                    {
                        "object_id": result["custom_id"],
                        "prompt": prompt_obj["prompt"],
                        "llm_response": result,
                    }
                )

            parsed_result = parse_mistral_response(backend._parse_response(result))
            parsed_result["custom_id"] = result["custom_id"]
            llm_responses.append(parsed_result)

    logger.warning("Batch inference complete")

    # Build results DataFrame - only include the specific field in the results
    results = []
    for rec in records:
        object_id = rec.get("object_id", "<unknown>")
        title = rec.get("title", "<no title>")
        abstract = rec.get("abstract", "<no abstract>")

        resp = next((x for x in llm_responses if x["custom_id"] == object_id), None)

        if resp:
            results.append(
                {
                    "object_id": object_id,
                    "title": title,
                    "abstract": abstract,
                    "llm_response": resp,
                }
            )

    # Save to JSON
    with open(output_field_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    logger.warning(f"Saved field extraction for {field} to {output_field_json}")

    # Save records to JSONL file
    with open(output_field_records_jsonl, "w", encoding="utf-8") as fp:
        for record in prompt_response_records:
            fp.write(json.dumps(record) + "\n")
    logger.warning(
        f"Saved {len(prompt_response_records)} records to {output_field_records_jsonl}"
    )

    elapsed_total = time.time() - start_time
    logger.warning(f"Total time taken: {elapsed_total:.2f} seconds")


if __name__ == "__main__":
    main()
