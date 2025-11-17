import os
import json
import click
import logging
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

from llm_inference.backends.openai_batch import OpenAIBatchBackend
from llm_inference.backends.mistral_batch import MistralBatchBackend
from llm_inference.cache.tmp import TmpCacheStorage
from llm_inference.cache.disk import DiskCacheStorage
from p03_evaluate_extractions.config import MAPPING, INVERSE_MAPPING

# Load environment variables from .env file and get API key
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not found in environment variables")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise EnvironmentError("MISTRAL_API_KEY not found in environment variables")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_openai_response(response):
    """
    Parse OpenAI response to extract the LLM judge decision.
    """
    try:
        # Extract the content from the OpenAI response structure
        content = response["choices"][0]["message"]["content"]

        # Parse the content as JSON
        parsed_content = json.loads(content)

        # Return the parsed JSON
        return {
            "explanation": parsed_content.get("explanation", ""),
            "final_decision": parsed_content.get("final_decision", ""),
            "custom_id": response.get("custom_id", ""),
        }
    except Exception as e:
        logger.error(f"Error parsing OpenAI response: {e}")
        logger.error(f"Original response: {response}")
        # Return a default structure if parsing fails
        return {
            "explanation": "Failed to parse response",
            "final_decision": "unknown",
            "custom_id": response.get("custom_id", ""),
        }


def parse_mistral_response(response):
    """
    Parse Mistral response to extract the specific field.
    """
    try:
        parsed_response = response
        # Add custom_id to the parsed response
        parsed_response["custom_id"] = response.get("custom_id", "")
        return parsed_response
    except Exception as e:
        logger.error(f"Error parsing Mistral response: {e}")
        logger.error(f"Original response: {response}")
        return {
            "error": "Failed to parse response",
            "custom_id": response.get("custom_id", ""),
        }


@click.command()
@click.option(
    "--metadata_extractions",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to field-specific metadata extractions JSON file from a specific model",
)
@click.option(
    "--eval_dataset_json",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to evaluation dataset JSON file for the specific field",
)
@click.option(
    "--llm_judge_model_config",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to model configuration file for the LLM judge",
)
@click.option(
    "--prompt_llm_judge",
    type=str,
    required=True,
    help="Path to the prompt for the LLM judge",
)
@click.option(
    "--model",
    type=str,
    required=True,
    help="Model name (e.g., small_mistral, large_mistral)",
)
@click.option(
    "--field",
    type=str,
    required=True,
    help="Field to compare (e.g., medical_condition, outcome_measure, etc.)",
)
@click.option(
    "--output_json",
    type=str,
    required=True,
    help="Path to output JSON file with comparison results",
)
def main(
    metadata_extractions,
    eval_dataset_json,
    llm_judge_model_config,
    prompt_llm_judge,
    model,
    field,
    output_json,
):
    """
    Compare field-specific metadata extractions from a model with the evaluation dataset.

    The process:
    1. Load field-specific metadata extractions from the model
    2. Load evaluation dataset with ground truth annotations for this field
    3. For each record:
       a. If model extraction matches ground truth exactly -> final_label=1, labeling_reason="model_agreement"
       b. If one of them is "Not specified" -> final_label=0, labeling_reason="one_model_unspecified"
       c. Otherwise, use LLM judge to determine if they're the same or different
    4. Save the results to a JSON and Excel file
    """
    start_time = time.time()

    # Ensure output directory exists
    out_dir = Path(output_json).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get the mapped field name if it exists
    field_name = INVERSE_MAPPING.get(field, field)
    logger.warning(f"Processing field: {field} (mapped to {field_name})")
    logger.warning(f"Extraction inferences made from model: {model}")

    # Format for OpenAI LLM judge prompt
    # prompt_path = f"etc/prompts/llm_as_a_judge_annotation/compare_{field}_for_eval_dataset_creation.txt"
    if not os.path.exists(prompt_llm_judge):
        raise FileNotFoundError(f"LLM judge prompt file not found: {prompt_llm_judge}")

    with open(prompt_llm_judge, "r", encoding="utf-8") as f:
        prompt_template = f.read().strip()

    # Load the model configuration
    # if "openai" in the name of llm_judge_model_config, then we are using OpenAI model
    is_openai_model = "openai" in llm_judge_model_config.lower()
    # if "istral" in the name of llm_judge_model_config, then we are using Mistral model
    is_mistral_model = "istral" in llm_judge_model_config.lower()
    with open(llm_judge_model_config, "r", encoding="utf-8") as f:
        judge_model_cfg = json.load(f)
    logger.warning(
        f"Using judge model config: {judge_model_cfg.get('model', 'unknown')}"
    )

    # Load metadata extractions
    with open(metadata_extractions, "r", encoding="utf-8") as f:
        extraction_records = json.load(f)
    logger.warning(
        f"Loaded {len(extraction_records)} records from field-specific metadata extractions"
    )

    # Load evaluation dataset
    with open(eval_dataset_json, "r", encoding="utf-8") as f:
        eval_records = json.load(f)
    logger.warning(f"Loaded {len(eval_records)} records from evaluation dataset")

    # Filter out records that need manual annotation
    filtered_eval_records = [
        r for r in eval_records if not r.get("needs_manual_annotation", False)
    ]
    logger.warning(
        f"Filtered to {len(filtered_eval_records)} records with completed annotations"
    )

    # Create a lookup dictionary for evaluation records by object_id
    eval_lookup = {r["object_id"]: r for r in filtered_eval_records}

    # Prepare records for processing
    processed_records = []
    llm_judge_prompts = []

    # Statistics counters
    stats = {
        "total_extracted": len(extraction_records),
        "in_eval_dataset": 0,
        "model_agreement": 0,
        "one_model_unspecified": 0,
        "need_llm_judge": 0,
        "llm_same": 0,
        "llm_different": 0,
    }

    correct_field_col = f"correct_{field}"

    # Process each extraction record
    for rec in extraction_records:
        object_id = rec.get("object_id")

        # Check if this record exists in the evaluation dataset
        if object_id not in eval_lookup:
            continue

        stats["in_eval_dataset"] += 1
        eval_rec = eval_lookup[object_id]

        # Get the model's extracted value for this field
        model_response = rec.get("llm_response", {})

        # For field-by-field extraction, the field value might be directly in the response
        # or nested in a field-specific location
        if field_name in model_response:
            inferred_value = model_response.get(field_name, "Not found")
        else:
            # Try to extract from the response in different formats
            inferred_value = "Not found"
            for key in model_response:
                if key.lower() == field_name.lower() or field.lower() in key.lower():
                    inferred_value = model_response[key]
                    break

        correct_value = eval_rec.get(correct_field_col, "Not found")

        # Create the basic record structure
        output_record = {
            "object_id": object_id,
            "title": rec.get("title", ""),
            "abstract": rec.get("abstract", ""),
            f"inferred_{field}": inferred_value,
            correct_field_col: correct_value,
        }

        # Check for exact match (case insensitive)
        if inferred_value.lower() == correct_value.lower():
            output_record["final_label"] = 1
            output_record["labeling_reason"] = "model_agreement"
            stats["model_agreement"] += 1
            processed_records.append(output_record)
            continue

        # Check if one is "Not specified"
        inferred_unspecified = (
            inferred_value.lower() == "not specified" or inferred_value == "not found"
        )
        correct_unspecified = (
            correct_value.lower() == "not specified" or correct_value == "not found"
        )

        if (inferred_unspecified and not correct_unspecified) or (
            not inferred_unspecified and correct_unspecified
        ):
            output_record["final_label"] = 0
            output_record["labeling_reason"] = "one_model_unspecified"
            stats["one_model_unspecified"] += 1
            processed_records.append(output_record)
            continue

        # Need LLM judge
        stats["need_llm_judge"] += 1

        # Prepare prompt for LLM judge
        full_prompt = prompt_template.replace("{{content_a}}", inferred_value)
        full_prompt = full_prompt.replace("{{content_b}}", correct_value)

        llm_judge_prompts.append({"prompt": full_prompt, "custom_id": object_id})

        # Save the record for later updating with LLM judgment
        processed_records.append(output_record)

    # Run LLM judgments if needed
    if llm_judge_prompts:
        logger.warning(f"Running LLM judgment for {len(llm_judge_prompts)} records")
        # if judge_model_cfg contains
        if is_openai_model:
            # Initialize the OpenAI backend
            backend = OpenAIBatchBackend(
                api_key=OPENAI_API_KEY, cache_storage=DiskCacheStorage()
            )
        elif is_mistral_model:
            # Initialize the Mistral backend
            backend = MistralBatchBackend(
                api_key=MISTRAL_API_KEY, cache_storage=DiskCacheStorage()
            )
        else:
            raise ValueError(
                "Unsupported model type. Please use OpenAI or Mistral models."
            )

        # Perform batch inference
        judge_results = backend.infer_many(llm_judge_prompts, judge_model_cfg)

        # Process LLM judge results
        for result in judge_results:
            if is_openai_model:
                parsed_result = parse_openai_response(result)
            elif is_mistral_model:
                parsed_result = parse_mistral_response(result)
            else:
                raise ValueError(
                    "Unsupported model type. Please use OpenAI or Mistral models."
                )
            object_id = parsed_result["custom_id"]

            # Find the corresponding record
            for record in processed_records:
                if record["object_id"] == object_id and "final_label" not in record:
                    if parsed_result["final_decision"].lower() == "same":
                        record["final_label"] = 1
                        record["labeling_reason"] = "llm_same"
                        stats["llm_same"] += 1
                    else:
                        record["final_label"] = 0
                        record["labeling_reason"] = "llm_different"
                        stats["llm_different"] += 1

                    # Store LLM explanation
                    record["llm_explanation"] = parsed_result["explanation"]
                    break

    # Ensure all records have the necessary fields
    for record in processed_records:
        if "final_label" not in record:
            record["final_label"] = (
                0  # Default to not matching if we couldn't determine
            )
            record["labeling_reason"] = "undetermined"

    # Log statistics
    logger.warning("Comparison statistics:")
    for key, value in stats.items():
        logger.warning(f"  {key}: {value}")

    # Count the final labels
    positive_labels = sum(1 for r in processed_records if r.get("final_label") == 1)
    negative_labels = sum(1 for r in processed_records if r.get("final_label") == 0)
    logger.warning(
        f"Final labels: {positive_labels} positive, {negative_labels} negative"
    )

    # Detailed breakdown of LLM judge results
    if stats["need_llm_judge"] > 0:
        llm_same_percent = (stats["llm_same"] / stats["need_llm_judge"]) * 100
        llm_different_percent = (stats["llm_different"] / stats["need_llm_judge"]) * 100
        logger.warning(
            f"LLM judge breakdown: {stats['llm_same']} same ({llm_same_percent:.1f}%), {stats['llm_different']} different ({llm_different_percent:.1f}%)"
        )

    # Save the results to JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(processed_records, f, indent=4, ensure_ascii=False)
    logger.warning(f"Saved comparison results to {output_json}")

    # Save to Excel for easier viewing
    output_excel = output_json.replace(".json", ".xlsx")
    pd.DataFrame(processed_records).to_excel(output_excel, index=False)
    logger.warning(f"Saved comparison results to {output_excel}")

    elapsed_time = time.time() - start_time
    logger.warning(f"Total processing time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
