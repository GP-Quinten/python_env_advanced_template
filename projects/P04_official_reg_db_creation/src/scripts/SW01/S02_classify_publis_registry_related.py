import logging
import time
import json
import os
from pathlib import Path
import glob
import click

import llm_backends
from llm_backends.cache import DiskCacheStorage

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


@click.command()
@click.option(
    "--input_jsonl_template",
    type=str,
    help="Input file template pattern where the batch JSONL files are stored.",
)
@click.option(
    "--output_raw_inferences_template",
    type=str,
    help="Output file template where the raw LLM inference results will be saved.",
)
@click.option(
    "--output_jsonl_template",
    type=str,
    help="Output file template where the registry-related publications will be saved.",
)
@click.option(
    "--prompt_txt",
    type=str,
    help="Path to the prompt template file.",
)
@click.option(
    "--model_config",
    type=str,
    help="Path to the model configuration file.",
)
@click.option(
    "--model",
    type=str,
    help="Model name to use for inference.",
)
def classify_publis_registry_related(
    input_jsonl_template: str,
    output_raw_inferences_template: str,
    output_jsonl_template: str,
    prompt_txt: str,
    model_config: str,
    model: str,
):
    """
    Classify publications as registry-related or not using an LLM.

    This script:
    1. Loads publications from batch files
    2. Uses an LLM to classify if they are registry-related
    3. Saves the registry-related publications and raw inference results
    """
    # Track execution time
    start_time = time.time()

    # Load the model configuration
    with open(model_config, "r", encoding="utf-8") as f:
        model_cfg = json.load(f)

    model_name = model_cfg.get("model", "unknown")
    logger.warning(f"Using model: {model_name}")

    # Load the annotation prompt
    with open(prompt_txt, "r", encoding="utf-8") as f:
        annotation_prompt = f.read().strip()

    # Get list of all batch files
    input_dir = Path(input_jsonl_template).parent
    batch_files = sorted(input_dir.glob("*.jsonl"), key=lambda p: int(p.stem))
    logger.warning(f"Found {len(batch_files)} batch files to process")

    # Create the output directories
    output_raw_inferences_dir = Path(output_raw_inferences_template).parent
    output_registry_related_dir = Path(output_jsonl_template).parent

    output_raw_inferences_dir.mkdir(parents=True, exist_ok=True)
    output_registry_related_dir.mkdir(parents=True, exist_ok=True)

    # Process each batch file separately
    total_registry_related = 0
    total_not_registry_related = 0
    total_records = 0

    # batch_files = batch_files[:2]  # Uncomment this line to use all batch files
    for batch_file in batch_files:
        batch_num = batch_file.stem  # Get batch number from filename
        logger.warning(f"Processing batch {batch_num}...")

        # Format output paths for this batch
        output_raw_inferences = output_raw_inferences_dir / f"{batch_num}.jsonl"
        output_registry_related = output_registry_related_dir / f"{batch_num}.jsonl"

        # Load records from this batch
        records = []
        with open(batch_file, "r") as file:
            for line in file:
                record = json.loads(line)
                records.append(record)
        # records = records[:3]  # Uncomment this line to use all records
        logger.warning(f"Loaded {len(records)} records from batch {batch_num}")

        # Prepare prompts for LLMs
        prompts_items = []
        for rec in records:
            object_id = rec.get("object_id", "<unknown>")
            title = rec.get("title", "<no title>")
            abstract = rec.get("abstract", "<no abstract>")
            full_prompt = f"{annotation_prompt}\nTitle: {title}\nAbstract: {abstract}"
            prompts_items.append({"prompt": full_prompt, "custom_id": object_id})

        # Create a list to store the records with object_id, prompt, and raw response
        prompt_response_records = []

        # Run batch inference
        start_inference_time = time.time()
        logger.warning(
            f"Starting batch inference for batch {batch_num} with {model_name}..."
        )
        llm_responses = []

        # Initialize the appropriate backend based on model type
        is_openai_model = "openai" in model_config.lower()
        is_mistral_model = "istral" in model_config.lower()

        if is_mistral_model:
            backend = llm_backends.MistralBatchBackend(
                api_key=os.getenv("MISTRAL_API_KEY"), cache_storage=DiskCacheStorage()
            )
        elif is_openai_model:
            backend = llm_backends.OpenAIAsyncBackend(
                api_key=os.getenv("OPENAI_API_KEY"), cache_storage=DiskCacheStorage()
            )
        else:
            raise ValueError(f"Unsupported model type: {model_config}")

        # Run the inference
        raw_responses = backend.infer_many(
            prompt_items=prompts_items,
            model_config=model_cfg,
        )

        # Process the responses
        for raw_response in raw_responses:
            # Store the raw response with object_id and prompt for the records file
            prompt_obj = next(
                (
                    p
                    for p in prompts_items
                    if p["custom_id"] == raw_response["custom_id"]
                ),
                None,
            )
            if prompt_obj:
                prompt_response_records.append(
                    {
                        "object_id": raw_response["custom_id"],
                        "prompt": prompt_obj["prompt"],
                        "llm_response": raw_response,
                    }
                )
                # Parse raw response
                parsed_response = backend._parse_response(raw_response)
                parsed_response["custom_id"] = raw_response.get("custom_id", "")
                llm_responses.append(parsed_response)

        elapsed_total = time.time() - start_inference_time
        logger.warning(
            f"Batch {batch_num} inference completed with {len(llm_responses)} responses in {elapsed_total:.2f} seconds"
        )

        # Build results for this batch
        registry_related_records = []
        for rec in records:
            object_id = rec.get("object_id", "<unknown>")
            resp = next((x for x in llm_responses if x["custom_id"] == object_id), None)

            if resp and resp.get("Registry related") == "yes":
                registry_related_records.append(rec)

        # Count statistics for this batch
        batch_registry_related = len(registry_related_records)
        batch_not_registry_related = len(records) - batch_registry_related
        batch_total = len(records)

        logger.warning(
            f"Batch {batch_num}: Found {batch_registry_related} records related to registry"
        )
        logger.warning(
            f"Batch {batch_num}: Found {batch_not_registry_related} records not related to registry"
        )

        rel_percentage = (
            batch_registry_related / batch_total * 100 if batch_total > 0 else 0
        )
        logger.warning(
            f"Batch {batch_num}: {rel_percentage:.1f}% of records are registry-related"
        )

        # Update totals
        total_registry_related += batch_registry_related
        total_not_registry_related += batch_not_registry_related
        total_records += batch_total

        # Save the raw inference results
        with open(output_raw_inferences, "w", encoding="utf-8") as fp:
            for record in prompt_response_records:
                fp.write(json.dumps(record) + "\n")

        # Save the registry-related records
        with open(output_registry_related, "w", encoding="utf-8") as fp:
            for record in registry_related_records:
                fp.write(json.dumps(record) + "\n")

        logger.warning(
            f"Saved batch {batch_num} raw inferences to {output_raw_inferences}"
        )
        logger.warning(
            f"Saved batch {batch_num} registry-related records to {output_registry_related}"
        )

    # Print overall statistics
    logger.warning("\n--- Overall Statistics ---")
    logger.warning(f"Total records processed: {total_records}")
    logger.warning(f"Total registry-related records: {total_registry_related}")
    logger.warning(f"Total not registry-related records: {total_not_registry_related}")

    if total_records > 0:
        overall_percentage = total_registry_related / total_records * 100
        logger.warning(
            f"Overall percentage of registry-related records: {overall_percentage:.1f}%"
        )

    # Log total execution time in hours + minutes
    total_execution_time = time.time() - start_time
    hours = int(total_execution_time // 3600)
    minutes = int((total_execution_time % 3600) // 60)
    logger.warning(f"Total execution time: {hours}h {minutes}m")


if __name__ == "__main__":
    classify_publis_registry_related()
