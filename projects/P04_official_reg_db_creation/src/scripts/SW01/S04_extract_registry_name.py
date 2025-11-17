import logging
import time
import json
import os
from pathlib import Path
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
    "--output_raw_inferences_jsonl_template",
    type=str,
    help="Output file template where the raw LLM inference results will be saved.",
)
@click.option(
    "--output_publis_json_template",
    type=str,
    help="Output file template where the publications with extracted registry names will be saved.",
)
@click.option(
    "--output_registries_jsonl_template",
    type=str,
    help="Output file template where the extracted registry names will be saved.",
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
@click.option(
    "--batch_limit",
    type=int,
    default=-1,
    help="Maximum number of input batches to process. Use -1 to process all available batches.",
)
def extract_registry_name(
    input_jsonl_template: str,
    output_raw_inferences_jsonl_template: str,
    output_publis_json_template: str,
    output_registries_jsonl_template: str,
    prompt_txt: str,
    model_config: str,
    model: str,
    batch_limit: int,
):
    """
    Extract registry names from registry-related publications using an LLM.

    This script:
    1. Loads registry-related publications from batch files
    2. Uses an LLM to extract registry names mentioned in each publication
    3. Saves the raw inference results, publications with registry info, and registry names
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

    # Limit the number of batches if specified
    if batch_limit > 0:
        batch_files = batch_files[:batch_limit]
        logger.warning(f"Processing first {batch_limit} input batches")
    else:
        logger.warning(f"Processing all {len(batch_files)} input batches")

    # Create the output directories
    output_raw_inferences_dir = Path(output_raw_inferences_jsonl_template).parent
    output_publis_json_dir = Path(output_publis_json_template).parent
    output_registries_jsonl_dir = Path(output_registries_jsonl_template).parent

    output_raw_inferences_dir.mkdir(parents=True, exist_ok=True)
    output_publis_json_dir.mkdir(parents=True, exist_ok=True)
    output_registries_jsonl_dir.mkdir(parents=True, exist_ok=True)

    # Process each batch file separately
    total_publications = 0
    total_registries_extracted = 0
    index = 1
    for batch_file in batch_files:
        batch_num = batch_file.stem  # Get batch number from filename
        logger.warning(f"Processing batch {batch_num}...")

        # Format output paths for this batch
        output_raw_inferences = output_raw_inferences_dir / f"{batch_num}.jsonl"
        output_publis_json = output_publis_json_dir / f"{batch_num}.jsonl"
        output_registries_jsonl = output_registries_jsonl_dir / f"{batch_num}.jsonl"

        # Load records from this batch
        records = []
        with open(batch_file, "r") as file:
            for line in file:
                record = json.loads(line)
                records.append(record)

        logger.warning(f"Loaded {len(records)} records from batch {batch_num}")
        total_publications += len(records)

        # Prepare prompts for LLMs
        prompts_items = []
        for rec in records:
            object_id = rec.get("object_id", "<unknown>")
            title = rec.get("title", "<no title>")
            abstract = rec.get("abstract", "<no abstract>")
            full_prompt = f"{annotation_prompt}\nText_to_analyze:\nTitle: {title}\nAbstract: {abstract}"
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

        # Build results for this batch - publications with registry names
        publications_with_registries = []
        for rec in records:
            object_id = rec.get("object_id", "<unknown>")
            resp = next((x for x in llm_responses if x["custom_id"] == object_id), None)

            if resp:
                publications_with_registries.append(
                    {
                        "object_id": object_id,
                        "title": rec.get("title", "<no title>"),
                        "abstract": rec.get("abstract", "<no abstract>"),
                        "data_source_name": rec.get("data_source_name", ""),
                        "llm_response": resp,
                    }
                )

        # Extract registry names from publications
        registry_names_list = []
        batch_registries_count = 0

        for publication in publications_with_registries:
            object_id = publication["object_id"]
            list_of_registries = publication["llm_response"].get(
                "List of Registry names", []
            )

            for i, registry in enumerate(list_of_registries):
                registry_names_list.append(
                    {
                        "index": index,
                        "registry_name": registry.get("registry_name", ""),
                        "acronym": registry.get("acronym", ""),
                        "is_official": registry.get("is_official", ""),
                        "object_id": object_id,
                    }
                )
                index += 1
                batch_registries_count += 1

        total_registries_extracted += batch_registries_count

        logger.warning(
            f"Batch {batch_num}: Extracted {batch_registries_count} registry names from {len(publications_with_registries)} publications"
        )

        # Save the raw inference results
        with open(output_raw_inferences, "w", encoding="utf-8") as fp:
            for record in prompt_response_records:
                fp.write(json.dumps(record) + "\n")

        # Save the publications with registry info
        with open(output_publis_json, "w", encoding="utf-8") as fp:
            for record in publications_with_registries:
                fp.write(json.dumps(record) + "\n")

        # Save the registry names
        with open(output_registries_jsonl, "w", encoding="utf-8") as fp:
            for registry in registry_names_list:
                fp.write(json.dumps(registry) + "\n")

        logger.warning(
            f"Saved batch {batch_num} raw inferences to {output_raw_inferences}"
        )
        logger.warning(
            f"Saved batch {batch_num} publications with registry info to {output_publis_json}"
        )
        logger.warning(
            f"Saved batch {batch_num} registry names to {output_registries_jsonl}"
        )

    # Print overall statistics
    logger.warning("\n--- Overall Statistics ---")
    logger.warning(f"Total publications processed: {total_publications}")
    logger.warning(f"Total registry names extracted: {total_registries_extracted}")

    if total_publications > 0:
        avg_registries_per_pub = total_registries_extracted / total_publications
        logger.warning(
            f"Average registry names per publication: {avg_registries_per_pub:.2f}"
        )

    # Log total execution time in hours and minutes
    hours = int((time.time() - start_time) / 3600)
    minutes = int((time.time() - start_time) % 3600 / 60)
    logger.warning(f"Total execution time: {hours} hours and {minutes} minutes")


if __name__ == "__main__":
    extract_registry_name()
