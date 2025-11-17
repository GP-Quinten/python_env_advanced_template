import logging
import time
import json
import os
from pathlib import Path
import click
import boto3
from tqdm import tqdm
import weaviate

import llm_backends
from llm_backends.cache import DiskCacheStorage
from src.p04_official_reg_db_creation import config
from src.p04_official_reg_db_creation.utils import (
    load_jsonl_from_s3,
    upload_jsonl_to_s3,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


def format_string(string):
    """Format string to remove unwanted characters."""
    # remove punctuation and special characters, lower case
    return "".join(e for e in string if e.isalnum() or e.isspace()).lower().strip()


@click.command()
@click.option(
    "--input_registry_data_jsonl_template",
    type=str,
    help="Input file template pattern where the registry data files are stored.",
)
@click.option(
    "--local_output_raw_inferences_jsonl_template",
    type=str,
    help="Output file template where the raw LLM inference results will be saved.",
)
@click.option(
    "--local_output_publis_jsonl_template",
    type=str,
    help="Output file template where the publications with extracted medical conditions will be saved.",
)
@click.option(
    "--local_output_registries_jsonl_template",
    type=str,
    help="Output file template where the registries with medical conditions will be saved.",
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
    "--collection",
    type=str,
    help="Weaviate collection name to query.",
)
@click.option(
    "--s3_output_publis_dir",
    type=str,
    help="S3 directory path where the publications data will be saved.",
)
@click.option(
    "--s3_output_registries_dir",
    type=str,
    help="S3 directory path where the registry data will be saved.",
)
@click.option(
    "--batch_limit",
    type=int,
    default=-1,
    help="Maximum number of input batches to process. Use -1 to process all available batches.",
)
def update_medical_condition(
    input_registry_data_jsonl_template: str,
    local_output_raw_inferences_jsonl_template: str,
    local_output_publis_jsonl_template: str,
    local_output_registries_jsonl_template: str,
    prompt_txt: str,
    model_config: str,
    collection: str,
    s3_output_publis_dir: str,
    s3_output_registries_dir: str,
    batch_limit: int,
):
    """
    Update medical condition fields for registry-related publications using an LLM.

    This script:
    1. Loads registry data and related publications
    2. Uses an LLM to extract medical conditions mentioned in each publication
    3. Updates the medical condition field in publications and registries
    4. Saves the results locally and to S3
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
        prompt_template = f.read().strip()

    # Create the output directories
    local_output_raw_inferences_dir = Path(
        local_output_raw_inferences_jsonl_template
    ).parent
    local_output_publis_json_dir = Path(local_output_publis_jsonl_template).parent
    local_output_registries_jsonl_dir = Path(
        local_output_registries_jsonl_template
    ).parent

    local_output_raw_inferences_dir.mkdir(parents=True, exist_ok=True)
    local_output_publis_json_dir.mkdir(parents=True, exist_ok=True)
    local_output_registries_jsonl_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load registry data from S3
    input_dir = Path(input_registry_data_jsonl_template).parent
    input_dir_str = str(input_dir)

    # Retrieve total_batches = how many files are in the folder
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=config.BUCKET_NAME_DEV, Prefix=input_dir_str)
    total_batches = len(response.get("Contents", []))

    registry_dataset = []
    for batch_number in range(1, total_batches + 1):
        file_name = f"{batch_number}.jsonl"
        batch = load_jsonl_from_s3(config.BUCKET_NAME_DEV, input_dir, file_name)
        registry_dataset.extend(batch)

    logger.warning(
        f"Loaded {len(registry_dataset)} records from the registry data, from {total_batches} batches."
    )

    # 2. Load publication data from Weaviate
    weaviate_client = weaviate.connect_to_custom(**config.WEAVIATE_PROD_CONF)
    collections = weaviate_client.collections
    collection_publications = collections.get(collection)

    publis_dataset_all = []
    for item in collection_publications.iterator(include_vector=False):
        # Extract subset of properties
        publis_dataset_all.append(
            {
                k: v
                for k, v in item.properties.items()
                if k
                in [
                    "object_id",
                    "title",
                    "abstract",
                    "geographical_area",
                    "medical_condition",
                ]
            }
        )
    weaviate_client.close()

    logger.warning(f"Loaded {len(publis_dataset_all)} publications from Weaviate.")

    # 3. Filter publications to only include those referenced in registries
    registry_publi_ids = set()
    for record in registry_dataset:
        if isinstance(record.get("list_publi_ids"), list):
            registry_publi_ids.update(record["list_publi_ids"])

    logger.warning(
        f"Found {len(registry_publi_ids)} unique publication IDs in the registry dataset."
    )

    # Filter publications to only those referenced in registries
    publis_dataset = [
        publi
        for publi in publis_dataset_all
        if publi["object_id"] in registry_publi_ids
    ]

    logger.warning(
        f"Filtered to {len(publis_dataset)} publications referenced in registries."
    )

    # Limit the number of publications if specified
    if batch_limit > 0:
        publis_dataset = publis_dataset[:batch_limit]
        logger.warning(f"Limited to first {batch_limit} publications for processing")

    # 4. Prepare prompts for LLM
    batch_size = 1000
    prompts_items = []
    for publi in publis_dataset:
        object_id = publi.get("object_id", "<unknown>")
        title = publi.get("title", "<no title>")
        abstract = publi.get("abstract", "<no abstract>")
        full_prompt = (
            f"{prompt_template}\nText_to_analyze:\nTitle: {title}\nAbstract: {abstract}"
        )
        prompts_items.append({"prompt": full_prompt, "custom_id": object_id})

    logger.warning(f"Prepared {len(prompts_items)} prompts for LLM processing.")

    # Split prompts into batches
    batch_prompts_list = []
    for i in range(0, len(prompts_items), batch_size):
        batch_prompts_list.append(prompts_items[i : i + batch_size])

    logger.warning(
        f"Split prompts into {len(batch_prompts_list)} batches of size {batch_size}."
    )

    # 5. Make LLM inferences
    batch_llm_responses_list = []
    prompt_response_records = []

    # Precompute a mapping from custom_id to prompt object
    prompt_map = {p["custom_id"]: p for p in prompts_items}

    initial_time = time.time()
    batch_number = 1

    for batch_prompts in tqdm(batch_prompts_list, desc="Processing batches"):
        logger.warning(f"--- Processing Batch N°{batch_number} ---")
        start_batch_time = time.time()

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
        batch_raw_responses = backend.infer_many(
            prompt_items=batch_prompts,
            model_config=model_cfg,
        )

        llm_responses = []
        for raw_response in tqdm(
            batch_raw_responses, desc=f"Batch {batch_number} processing", leave=False
        ):
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
                # Parse raw response and add additional info
                parsed_response = backend._parse_response(raw_response)
                parsed_response["custom_id"] = custom_id
                llm_responses.append(parsed_response)

        batch_llm_responses_list.append(llm_responses)

        elapsed_total = (time.time() - start_batch_time) / 60  # Convert to minutes
        logger.warning(
            f"Batch {batch_number} inference completed with {len(llm_responses)} responses"
        )
        logger.warning(
            f"Total time for batch {batch_number} inference: {elapsed_total:.1f} minutes\n"
        )
        batch_number += 1

    total_computation_time = (time.time() - initial_time) / 60  # Convert to minutes
    logger.warning(
        f"--> Total computation time for all batches: {total_computation_time:.1f} minutes <--"
    )

    # 6. Update the medical condition field in the publications dataset
    # Prebuild a mapping from publication object_id to the publication record
    pub_map = {pub.get("object_id", ""): pub for pub in publis_dataset}

    # Update the publis_dataset with the extracted field
    for llm_responses in tqdm(
        batch_llm_responses_list, desc="Updating publications batches"
    ):
        if not llm_responses:
            continue  # Skip empty batches
        for response in tqdm(llm_responses, desc="Processing responses", leave=False):
            publi_id = response.get("custom_id", "")
            updated_field = response.get("medical_condition", None)
            if updated_field is not None:
                publi = pub_map.get(publi_id)
                if publi is None:
                    continue
                details = None
                formatted_details = []
                if "[" in updated_field and "]" in updated_field:
                    start_idx = updated_field.index("[") + 1
                    end_idx = updated_field.index("]")
                    details = updated_field[start_idx:end_idx]
                    formatted_details = [
                        detail.strip()
                        for detail in details.split(";")
                        if detail.strip()
                    ]
                    updated_field = (
                        updated_field.replace(details, "")
                        .replace("[", "")
                        .replace("]", "")
                        .strip()
                    )
                formatted_updated_field = [
                    condition.strip()
                    for condition in updated_field.split(";")
                    if condition.strip()
                ]
                publi["medical_condition"] = formatted_updated_field
                publi["medical_condition_details"] = (
                    formatted_details if details else []
                )

    logger.warning(
        f"Updated medical_condition field for {len(publis_dataset)} publications."
    )

    # 7. Update the medical_condition field in the registry_dataset
    # Prebuild a dictionary mapping publication object_ids to their medical conditions
    pub_dict = {
        publi.get("object_id"): publi.get("medical_condition", [])
        for publi in publis_dataset
    }

    # Update medical_condition in registry_dataset
    for registry in tqdm(registry_dataset, desc="Processing registries"):
        terms_counts = {}
        for pub_id in registry.get("list_publi_ids", []):
            updated_field = pub_dict.get(pub_id, [])
            for term in updated_field:
                formatted_term = format_string(term)
                if formatted_term:
                    key = formatted_term.title()
                    terms_counts[key] = terms_counts.get(key, 0) + 1
        # Rank the medical conditions by count (highest first)
        ranked_terms = dict(
            sorted(terms_counts.items(), key=lambda item: item[1], reverse=True)
        )
        registry["medical_condition"] = ranked_terms
    # Count only non-empty medical conditions
    nb_new_medical_conditions = sum(
        1 for registry in registry_dataset if registry.get("medical_condition")
    )
    logger.warning(
        f"Updated medical_condition field for {nb_new_medical_conditions}/{len(registry_dataset)} registries."
    )

    # 8. Save the results
    # Save raw responses
    for i in range(0, len(prompt_response_records), batch_size):
        batch = prompt_response_records[i : i + batch_size]
        batch_number = (i // batch_size) + 1
        local_file_path = local_output_raw_inferences_dir / f"{batch_number}.jsonl"
        with open(local_file_path, "w", encoding="utf-8") as fp:
            for record in batch:
                fp.write(json.dumps(record) + "\n")
        logger.warning(
            f"Saved batch {batch_number} of raw responses to {local_file_path}"
        )

    # Save publications with medical conditions
    for i in range(0, len(publis_dataset), batch_size):
        batch = publis_dataset[i : i + batch_size]
        batch_number = i // batch_size + 1
        local_file_path = local_output_publis_json_dir / f"{batch_number}.jsonl"
        # Save locally
        with open(local_file_path, "w", encoding="utf-8") as fp:
            for record in batch:
                fp.write(json.dumps(record) + "\n")
        # Save to S3
        s3_file_name = f"{batch_number}.jsonl"
        upload_jsonl_to_s3(
            batch, config.BUCKET_NAME_DEV, s3_output_publis_dir, s3_file_name
        )
        logger.warning(
            f"Saved batch {batch_number} of publications to {local_file_path} and S3"
        )

    # Save registries with medical conditions
    for i in range(0, len(registry_dataset), batch_size):
        batch = registry_dataset[i : i + batch_size]
        batch_number = i // batch_size + 1
        local_file_path = local_output_registries_jsonl_dir / f"{batch_number}.jsonl"
        # Save locally
        with open(local_file_path, "w", encoding="utf-8") as fp:
            for record in batch:
                fp.write(json.dumps(record) + "\n")
        # Save to S3
        s3_file_name = f"{batch_number}.jsonl"
        upload_jsonl_to_s3(
            batch,
            config.BUCKET_NAME_DEV,
            s3_output_registries_dir,
            s3_file_name,
        )
        logger.warning(
            f"Saved batch {batch_number} of registries to {local_file_path} and S3"
        )

    # Log total execution time in hours and minutes
    total_hours = int((time.time() - start_time) / 3600)
    total_minutes = int((time.time() - start_time) % 3600 / 60)
    logger.warning(
        f"Total execution time: {total_hours} hours and {total_minutes} minutes"
    )


if __name__ == "__main__":
    update_medical_condition()
