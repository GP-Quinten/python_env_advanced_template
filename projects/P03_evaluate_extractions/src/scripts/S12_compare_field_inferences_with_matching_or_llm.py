import os
import json
import click
import logging
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# from llm_inference.backends.openai_async import OpenAIAsyncBackend
from llm_inference.backends.openai_batch import OpenAIBatchBackend
from llm_inference.backends.mistral_batch import MistralBatchBackend
from llm_inference.cache.tmp import TmpCacheStorage
from llm_inference.cache.disk import DiskCacheStorage
from p03_evaluate_extractions.config import INVERSE_MAPPING, MAPPING

# Load environment variables from .env file and get API key
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not found in environment variables")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise EnvironmentError("MISTRAL_API_KEY not found in environment variables")

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


def parse_openai_response(response):
    """
    Parse OpenAI response to extract the content.
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


@click.command()
@click.option(
    "--metadata_extractions",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to input JSON file with metadata extractions from both models",
)
# @click.option(
#     "--previous_annotations",
#     type=click.Path(exists=True, dir_okay=False),
#     required=False,
#     help="Path to previous manual annotations Excel file",
# )
@click.option(
    "--model_config",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the model configuration JSON file for LLM as a judge",
)
@click.option(
    "--prompt_txt",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the prompt text file for LLM as a judge",
)
@click.option(
    "--field",
    type=str,
    required=True,
    help="Field to compare between the two models",
)
@click.option(
    "--output_json",
    type=str,
    required=True,
    help="Path to output JSON file",
)
def main(
    metadata_extractions,
    # previous_annotations,
    model_config,
    prompt_txt,
    field,
    output_json,
):
    """
    Compare field inferences from two models using matching or LLM as a judge.

    The process:
    1. Load previous annotations to avoid redundant LLM calls
    2. For publications without annotations:
        - If both models match exactly (case insensitive), use the large model's extraction
        - If only one model returns "Not specified", flag for manual annotation
        - Otherwise, use LLM as a judge to decide
    """
    start_time = time.time()

    # Ensure output directory exists
    out_dir = Path(output_json).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get the mapped field name if it exists
    field_name = INVERSE_MAPPING.get(field, field)
    logger.warning(f"Processing field: {field} (mapped to {field_name})")

    # Load the model configuration
    with open(model_config, "r", encoding="utf-8") as f:
        model_cfg = json.load(f)
    logger.warning(f"Using model config: {model_cfg.get('model', 'unknown')}")
    is_openai_model = "openai" in model_config.lower()
    is_mistral_model = "istral" in model_config.lower()

    # Load metadata extractions from both models
    with open(metadata_extractions, "r", encoding="utf-8") as f:
        records = json.load(f)
    # # test on a small subset of 5 records
    # records = records[:5]  # Uncomment for testing with a small subset
    # print("Loaded records:", len(records))
    logger.warning(f"Loaded {len(records)} records from {metadata_extractions}")

    # # Load previous annotations if available
    # previous_annotations_df = None
    # if previous_annotations:
    #     previous_annotations_df = pd.read_excel(previous_annotations)
    #     logger.warning(
    #         f"Loaded {len(previous_annotations_df)} previous annotations from {previous_annotations}"
    #     )
    # else:
    #     logger.warning("No previous annotations provided")

    # Load the prompt template
    with open(prompt_txt, "r", encoding="utf-8") as f:
        prompt_template = f.read().strip()

    # Prepare the records for processing
    all_records = []
    llm_prompts = []

    # String variations that count as "Not specified"
    unspecified_strings = ["Not specified", "Not found"]

    for record in records:
        object_id = record.get("object_id")
        title = record.get("title", "")
        abstract = record.get("abstract", "")

        # Extract content from both models' nested structure in the metadata_extractions.json file
        small_model_response = record.get("small_model_response", {})
        large_model_response = record.get("large_model_response", {})

        # Extract the actual field value from the model responses
        small_model_field_value = (
            small_model_response.get(field_name, "") if small_model_response else ""
        )
        large_model_field_value = (
            large_model_response.get(field_name, "") if large_model_response else ""
        )

        # Log extracted values for debugging
        logger.debug(f"Object ID: {object_id}")
        logger.debug(f"Small model field '{field}' value: {small_model_field_value}")
        logger.debug(f"Large model field '{field}' value: {large_model_field_value}")

        # Check if we already have an annotation for this record
        correct_field_value = None
        annotated_by = None

        # if previous_annotations_df is not None:
        #     matching_row = previous_annotations_df[
        #         previous_annotations_df["object_id"] == object_id
        #     ]
        #     if not matching_row.empty:
        #         correct_field_col = f"correct_{field}"
        #         if correct_field_col in matching_row.columns:
        #             correct_field_value = matching_row[correct_field_col].iloc[0]
        #             annotated_by = (
        #                 matching_row["annotated_by"].iloc[0]
        #                 if "annotated_by" in matching_row.columns
        #                 else "Unknown"
        #             )
        #             logger.info(
        #                 f"Found previous annotation for {object_id}: {correct_field_value}"
        #             )

        # Create basic record structure
        output_record = {
            "object_id": object_id,
            "title": title,
            "abstract": abstract,
            f"small_model_{field}": small_model_field_value,
            f"large_model_{field}": large_model_field_value,
            f"correct_{field}": correct_field_value,
            "annotated_by": annotated_by,
            "needs_manual_annotation": False,
            "annotation_method": (
                "previous" if correct_field_value is not None else None
            ),
        }

        # # If we already have an annotation, we're done with this record
        # if correct_field_value is not None:
        #     all_records.append(output_record)
        #     continue

        # Check if both models return the same value (case insensitive)
        if small_model_field_value.lower() == large_model_field_value.lower():
            output_record[f"correct_{field}"] = large_model_response.get(field_name, "")
            output_record["annotation_method"] = "model_agreement"
            all_records.append(output_record)
            continue

        # Check if one model says "Not specified" and the other doesn't
        small_is_unspecified = any(
            unspec in small_model_field_value for unspec in unspecified_strings
        )
        large_is_unspecified = any(
            unspec in large_model_field_value for unspec in unspecified_strings
        )

        if small_is_unspecified and large_is_unspecified:
            # Both models say "Not specified", no manual annotation needed
            output_record["annotation_method"] = "model_agreement"
            all_records.append(output_record)
            continue

        if small_is_unspecified and not large_is_unspecified:
            output_record["needs_manual_annotation"] = True
            output_record["annotation_method"] = "one_model_unspecified"
            all_records.append(output_record)
            continue

        if large_is_unspecified and not small_is_unspecified:
            output_record["needs_manual_annotation"] = True
            output_record["annotation_method"] = "one_model_unspecified"
            all_records.append(output_record)
            continue

        # Neither model says "Not specified" but they disagree, prepare for LLM judgment
        # Make sure we use the actual values for comparison, not model response objects
        full_prompt = prompt_template
        full_prompt = full_prompt.replace("{{content_a}}", small_model_field_value)
        full_prompt = full_prompt.replace("{{content_b}}", large_model_field_value)

        llm_prompts.append({"prompt": full_prompt, "custom_id": object_id})

        # Add to all records (LLM judgment will be added later)
        output_record["annotation_method"] = "pending_llm_judgment"
        all_records.append(output_record)

    # Run LLM judgment if needed
    if llm_prompts:
        logger.warning(f"Running LLM judgment for {len(llm_prompts)} records")
        if is_openai_model:
            # Initialize the OpenAI backend
            backend = OpenAIBatchBackend(
                api_key=OPENAI_API_KEY, cache_storage=DiskCacheStorage()
            )
        elif is_mistral_model:
            # Initialize the MistralBatch backend
            backend = MistralBatchBackend(
                api_key=MISTRAL_API_KEY, cache_storage=DiskCacheStorage()
            )
        else:
            raise ValueError("Unsupported model type in configuration")

        # Perform batch inference
        logger.warning(
            f"Starting batch inference with {model_cfg.get('model', 'unknown')}..."
        )
        results = backend.infer_many(llm_prompts, model_cfg)

        # Process LLM results
        for result in results:
            if is_openai_model:
                parsed_result = parse_openai_response(result)
            elif is_mistral_model:
                parsed_result = parse_mistral_response(result)
            else:
                raise ValueError("Unsupported model type in configuration")
            object_id = parsed_result["custom_id"]

            # Find the corresponding record
            for record in all_records:
                if (
                    record["object_id"] == object_id
                    and record["annotation_method"] == "pending_llm_judgment"
                ):
                    if parsed_result["final_decision"].lower() == "same":
                        # If LLM says they're the same, use large model's extraction
                        record_idx = next(
                            (
                                i
                                for i, r in enumerate(records)
                                if r["object_id"] == object_id
                            ),
                            None,
                        )
                        if record_idx is not None:
                            large_model_val = records[record_idx][
                                "large_model_response"
                            ].get(field_name, "")
                            record[f"correct_{field}"] = large_model_val
                        record["annotation_method"] = "llm_same"
                    # CHECK if there is a previous annotation
                    # elif final_decision is different, but there is an existing previous annotation, use the previous annotation
                    elif parsed_result["final_decision"].lower() == "different":
                        # Check if there's a previous annotation for this record
                        if record[f"correct_{field}"] is not None:
                            # Use the existing previous annotation
                            record["annotation_method"] = "previous"
                        else:
                            # If there's no previous annotation, flag for manual annotation
                            record["needs_manual_annotation"] = True
                            record["annotation_method"] = "llm_different"

                    # else:
                    #     # If LLM says they're different, flag for manual annotation
                    #     record["needs_manual_annotation"] = True
                    #     record["annotation_method"] = "llm_different"

                    # Store LLM explanation
                    record["llm_explanation"] = parsed_result["explanation"]
                    break

    # Log statistics
    # Make sure all annotation methods are accounted for
    possible_annotation_methods = [
        "previous",
        "model_agreement",
        "one_model_unspecified",
        "llm_same",
        "llm_different",
        # "both_empty",
    ]
    annotation_stats = {method: 0 for method in possible_annotation_methods}
    need_manual_count = 0

    for record in all_records:
        method = record["annotation_method"]
        annotation_stats[method] += 1

        if record["needs_manual_annotation"]:
            need_manual_count += 1

    logger.warning("Annotation statistics:")
    for method, count in annotation_stats.items():
        logger.warning(f"  {method}: {count}")

    logger.warning(f"Records requiring manual annotation: {need_manual_count}")

    # Save the results
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=4, ensure_ascii=False)

    logger.warning(f"Results saved to {output_json}")

    # also save to Excel, so we can share it to the team
    output_excel = output_json.replace(".json", ".xlsx")
    all_records_df = pd.DataFrame(all_records)
    # add an empty column 'comment' to the output excel
    all_records_df["comment"] = ""
    all_records_df.to_excel(output_excel, index=False)
    logger.warning(f"Results saved to {output_excel}")

    elapsed_time = time.time() - start_time
    logger.warning(f"Total processing time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
