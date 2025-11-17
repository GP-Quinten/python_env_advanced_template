import os
import json
import click
import logging
import time
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

from llm_inference.backends.mistral_batch import MistralBatchBackend
from llm_inference.backends.openai_batch import OpenAIBatchBackend
from llm_inference.cache.tmp import TmpCacheStorage
from llm_inference.cache.disk import DiskCacheStorage

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
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
    Parse Mistral response to extract the registry related field.
    """
    try:
        content = response
        if isinstance(content, str):
            parsed_content = json.loads(content)
        else:
            parsed_content = content

        registry_related = parsed_content.get("Registry related", "").lower()
        return registry_related
    except Exception as e:
        logger.error(f"Error parsing Mistral response: {e}")
        logger.error(f"Original response: {response}")
        return "error"


def parse_openai_response(response):
    """
    Parse OpenAI response to extract the registry related field.
    """
    try:
        content = response["choices"][0]["message"]["content"]
        parsed_content = json.loads(content)
        registry_related = parsed_content.get("Registry related", "").lower()
        return registry_related
    except Exception as e:
        logger.error(f"Error parsing OpenAI response: {e}")
        logger.error(f"Original response: {response}")
        return "error"


@click.command()
@click.option(
    "--base_pubmed_dataset_jsonl",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to input JSONL file with PubMed records.",
)
@click.option(
    "--current_eval_dataset_xlsx",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the current evaluation dataset excel file.",
)
@click.option(
    "--prompt_txt",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the annotation prompt text file.",
)
@click.option(
    "--model_a_config",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the GPT-4.1 model configuration JSON file.",
)
@click.option(
    "--model_b_config",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the Mistral large model configuration JSON file.",
)
@click.option(
    "--output_json",
    type=str,
    required=True,
    help="Path to output JSON file with the regenerated evaluation dataset.",
)
def main(
    base_pubmed_dataset_jsonl: str,
    current_eval_dataset_xlsx: str,
    prompt_txt: str,
    model_a_config: str,
    model_b_config: str,
    output_json: str,
) -> None:
    """
    Regenerate the registry-related evaluation dataset with new extractions from
    GPT-4.1 and Mistral Large models using an updated prompt.
    """
    start_time = time.time()

    # Ensure output directory exists
    out_dir = Path(output_json).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model configurations
    with open(model_a_config, "r", encoding="utf-8") as f:
        model_a_cfg = json.load(f)
    with open(model_b_config, "r", encoding="utf-8") as f:
        model_b_cfg = json.load(f)

    model_a_name = model_a_cfg.get("model", "unknown")
    model_b_name = model_b_cfg.get("model", "unknown")
    logger.warning(f"Using models: {model_a_name} and {model_b_name}")

    # Check API keys
    if not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY not found in environment variables")
    if not MISTRAL_API_KEY:
        raise EnvironmentError("MISTRAL_API_KEY not found in environment variables")

    # Load current evaluation dataset
    logger.warning(
        f"Loading current evaluation dataset from {current_eval_dataset_xlsx}..."
    )
    current_eval_df = pd.read_excel(current_eval_dataset_xlsx, engine="openpyxl")
    logger.warning(
        f"Current evaluation dataset contains {len(current_eval_df)} records"
    )

    # Load annotation prompt
    annotation_prompt = Path(prompt_txt).read_text(encoding="utf-8").strip()

    # Load PubMed records (JSONL)
    records = []
    with open(base_pubmed_dataset_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    logger.warning(f"Loaded {len(records)} PubMed records")

    # Extract object_ids from current evaluation dataset
    eval_object_ids = set(current_eval_df["object_id"].tolist())

    # Filter records to only those in the evaluation dataset
    records = [rec for rec in records if rec.get("object_id") in eval_object_ids]
    logger.warning(f"Filtered to {len(records)} records in evaluation dataset")

    # # select the first 5 records for testing
    # # records = records[:5]  # Uncomment for testing with a small subset
    # # check on th distinct values of annotated_by in records
    # distinct_annotators = set(rec.get("annotated_by", "<unknown>") for rec in records)
    # logging.warning(f"Distinct annotators in records: {distinct_annotators}")

    # Prepare prompts for LLMs
    prompts = []
    for rec in records:
        object_id = rec.get("object_id", "<unknown>")
        title = rec.get("title", "<no title>")
        abstract = rec.get("abstract", "<no abstract>")
        full_prompt = f"{annotation_prompt}\nText_to_analyze:\nTitle: {title}\nAbstract: {abstract}"
        prompts.append({"prompt": full_prompt, "custom_id": object_id})

    # ----- BATCH INFERENCE WITH MODEL A (GPT-4.1) -----
    logger.warning(f"Starting batch inference with {model_a_name}...")

    model_a_responses = {}
    backend_a = OpenAIBatchBackend(
        api_key=OPENAI_API_KEY, cache_storage=DiskCacheStorage()
    )
    inferred_results_a = backend_a.infer_many(prompts, model_a_cfg)

    for result in inferred_results_a:
        object_id = result["custom_id"]
        model_a_responses[object_id] = parse_openai_response(result)

    logger.warning(f"Completed batch inference with {model_a_name}")

    # ----- BATCH INFERENCE WITH MODEL B (Mistral Large) -----
    logger.warning(f"Starting batch inference with {model_b_name}...")

    model_b_responses = {}
    backend_b = MistralBatchBackend(
        api_key=MISTRAL_API_KEY, cache_storage=DiskCacheStorage()
    )
    inferred_results_b = backend_b.infer_many(prompts, model_b_cfg)

    for result in inferred_results_b:
        object_id = result["custom_id"]
        model_b_responses[object_id] = parse_mistral_response(
            backend_b._parse_response(result)
        )

    logger.warning(f"Completed batch inference with {model_b_name}")

    # Build regenerated evaluation dataset
    regenerated_dataset = []
    manual_annotators = ["sonia", "ghinwa", "gaetan"]  # List of human annotators

    for rec in records:
        object_id = rec.get("object_id", "<unknown>")
        title = rec.get("title", "<no title>")
        abstract = rec.get("abstract", "<no abstract>")

        # Get model inferences
        gpt_4_1_inference = model_a_responses.get(object_id, "error")
        large_mistral_inference = model_b_responses.get(object_id, "error")

        # Find current evaluation data for this record if it exists
        current_eval_record = current_eval_df[
            current_eval_df["object_id"] == object_id
        ].to_dict("records")
        current_eval_record = current_eval_record[0] if current_eval_record else None

        # Default values
        needs_manual_annotation = True
        annotated_by = None
        annotation_method = None
        correct_registry_related = None
        answer_switch = None
        comment = ""

        # If record exists in current evaluation dataset
        if current_eval_record:
            # Get existing annotation if available
            existing_annotator = current_eval_record.get("annotated_by", "")
            if isinstance(existing_annotator, float) and pd.isna(existing_annotator):
                existing_annotator = ""
            # print(f"Existing annotator: {existing_annotator}")
            existing_correct_value = current_eval_record.get(
                "correct_registry_related", ""
            )
            if isinstance(existing_correct_value, float) and pd.isna(
                existing_correct_value
            ):
                existing_correct_value = ""
            comment = current_eval_record.get("comment", "")
            if isinstance(comment, float) and pd.isna(comment):
                comment = ""

            # If models agree, use model agreement
            if gpt_4_1_inference == large_mistral_inference:
                needs_manual_annotation = False
                annotated_by = "mistral_large"
                annotation_method = "model_agreement"
                correct_registry_related = large_mistral_inference
                answer_switch = (
                    existing_correct_value != large_mistral_inference
                    if existing_correct_value
                    else None
                )
            # Check if it was manually annotated
            elif existing_annotator.lower() in manual_annotators:
                needs_manual_annotation = False
                annotated_by = existing_annotator
                annotation_method = "manual_annotation"
                correct_registry_related = existing_correct_value
                answer_switch = (
                    False  # No switch because we're keeping the manual annotation
                )
            # Models disagree and no manual annotation
            else:
                needs_manual_annotation = True
                annotated_by = None
                annotation_method = None
                correct_registry_related = None
                answer_switch = None
        # New record (shouldn't happen, but just in case)
        else:
            # If models agree, use model agreement
            if gpt_4_1_inference == large_mistral_inference:
                needs_manual_annotation = False
                annotated_by = "mistral_large"
                annotation_method = "model_agreement"
                correct_registry_related = large_mistral_inference
                answer_switch = None  # No previous value to compare
            # Models disagree
            else:
                needs_manual_annotation = True
                annotated_by = None
                annotation_method = None
                correct_registry_related = None
                answer_switch = None

        # Add record to regenerated dataset
        regenerated_dataset.append(
            {
                "object_id": object_id,
                "title": title,
                "abstract": abstract,
                "gpt_4_1_inference": gpt_4_1_inference,
                "large_mistral_inference": large_mistral_inference,
                "needs_manual_annotation": needs_manual_annotation,
                "annotated_by": annotated_by,
                "annotation_method": annotation_method,
                "correct_registry_related": correct_registry_related,
                "answer_switch": answer_switch,
                "comment": comment,
            }
        )

    # Save regenerated dataset to JSON
    with open(output_json, "w", encoding="utf-8") as fp:
        json.dump(regenerated_dataset, fp, indent=4)
    logger.warning(f"Saved regenerated evaluation dataset to {output_json}")

    # Also save as Excel for easier review
    excel_path = output_json.replace(".json", ".xlsx")
    pd.DataFrame(regenerated_dataset).to_excel(excel_path, index=False)
    logger.warning(f"Saved Excel version to {excel_path}")

    # Calculate and print statistics
    df = pd.DataFrame(regenerated_dataset)
    total_records = len(df)

    # Annotation needs stats
    need_annotation_count = df["needs_manual_annotation"].sum()
    already_annotated_count = total_records - need_annotation_count

    # Annotator stats
    annotator_counts = df["annotated_by"].value_counts().to_dict()

    # Answer switch stats
    answer_switch_count = df["answer_switch"].fillna(False).sum()

    # Print statistics
    print("\n========== DATASET STATISTICS ==========")
    print(f"Total records: {total_records}")
    print(
        f"Records needing manual annotation: {need_annotation_count} ({need_annotation_count/total_records:.1%})"
    )
    print(
        f"Records already annotated: {already_annotated_count} ({already_annotated_count/total_records:.1%})"
    )
    print("\nAnnotator distribution:")
    for annotator, count in annotator_counts.items():
        if annotator is not None:
            print(f"  - {annotator}: {count} ({count/total_records:.1%})")
    print(
        f"\nRecords with answer switch: {answer_switch_count} ({answer_switch_count/total_records:.1%})"
    )
    print("=======================================\n")

    elapsed_total = time.time() - start_time
    logger.warning(f"Total time taken: {elapsed_total:.2f} seconds")


if __name__ == "__main__":
    main()
