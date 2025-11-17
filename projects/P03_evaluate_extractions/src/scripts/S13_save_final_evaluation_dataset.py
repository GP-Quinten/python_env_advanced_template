import os
import json
import click
import logging
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from p03_evaluate_extractions.config import MAPPING, INVERSE_MAPPING

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--publications_already_annotated",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to JSON file with already annotated publications",
)
@click.option(
    "--publications_newly_annotated",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to Excel file with newly annotated publications",
)
@click.option(
    "--field",
    type=str,
    required=True,
    help="Field to process (e.g., 'medical_condition', 'outcome_measure', etc.)",
)
@click.option(
    "--output_json",
    type=str,
    required=True,
    help="Path to output JSON file for the final evaluation dataset",
)
def main(
    publications_already_annotated,
    publications_newly_annotated,
    field,
    output_json,
):
    """
    Create the final evaluation dataset by merging publications already annotated with newly annotated ones.

    The process:
    1. Load publications that were already partially annotated
    2. Load newly annotated publications
    3. Update records that needed manual annotation with new annotations
    4. Update annotator information based on annotation method
    5. Save the final dataset in both JSON and Excel format
    """
    start_time = time.time()

    # Get the mapped field name if it exists
    field_name = INVERSE_MAPPING.get(field, field)
    logger.warning(f"Processing field: {field} (mapped to {field_name})")

    # Ensure output directories exist
    output_dir = Path(output_json).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output Excel file has the same path as JSON but with .xlsx extension
    output_excel = str(output_json).replace(".json", ".xlsx")
    if output_excel == output_json:
        output_excel = output_json + ".xlsx"

    # 1. Load publications that were already partially annotated
    with open(publications_already_annotated, "r", encoding="utf-8") as f:
        all_records = json.load(f)
    logger.warning(f"Loaded {len(all_records)} records from already annotated dataset")

    # 2. Load newly annotated publications
    newly_annotated_df = pd.read_excel(publications_newly_annotated).fillna("")
    logger.warning(
        f"Loaded {len(newly_annotated_df)} records from newly annotated dataset"
    )

    # Determine field column name in newly annotated data
    correct_field_col = f"correct_{field}"

    # 3. Create lookup dict from newly annotated records
    new_annotations = {}
    for _, row in newly_annotated_df.iterrows():
        # if the object_id is not NaN and the correct field is not NaN or None, or empty string, add to new_annotations
        if (
            pd.notna(row.get("object_id"))
            and pd.notna(row.get(correct_field_col))
            and row.get(correct_field_col, "").strip()
        ):
            new_annotations[row["object_id"]] = {
                correct_field_col: row[correct_field_col],
                "comment": row.get("comment", ""),
                "annotated_by": row.get("annotated_by", "probably sonia"),
            }

    logger.warning(f"Found {len(new_annotations)} usable new annotations")

    # 4. Process each record in the already annotated dataset
    processed_records = []

    # Statistics counters
    stats = {
        "total": len(all_records),
        "updated_with_new_annotations": 0,
        "already_had_annotations": 0,
        "still_needs_annotation": 0,
    }
    annotators_stats = {
        "mistral-large": 0,
        "sonia": 0,
        "ghinwa": 0,
        "gaetan": 0,
    }

    for record in all_records:
        object_id = record["object_id"]

        # Check if this record needs manual annotation and has a new annotation
        if (
            record.get("needs_manual_annotation", False)
            and object_id in new_annotations
        ):
            record["needs_manual_annotation"] = False
            record[correct_field_col] = new_annotations[object_id][correct_field_col]
            record["annotated_by"] = new_annotations[object_id].get(
                "annotated_by", "Unknown"
            )
            record["comment"] = new_annotations[object_id].get("comment", "")
            stats["updated_with_new_annotations"] += 1
            annotators_stats[new_annotations[object_id]["annotated_by"]] += 1

        # Update annotated_by based on annotation_method
        annotation_method = record.get("annotation_method", "")
        current_annotator = record.get("annotated_by", "")

        if annotation_method in ["llm_same", "model_agreement"]:
            if not current_annotator or current_annotator == "Unknown":
                record["annotated_by"] = "mistral-large"
                annotators_stats["mistral-large"] += 1

        # elif annotation_method in ["previous"]:
        #     if not current_annotator or current_annotator in ["Unknown", "sonia"]:
        #         record["annotated_by"] = "sonia"
        #         annotators_stats["sonia"] += 1
        #     elif current_annotator == "mistral-large":
        #         record["annotated_by"] = "mistral-large"
        #         annotators_stats["mistral-large"] += 1
        #     elif current_annotator == "ghinwa":
        #         record["annotated_by"] = "ghinwa"
        #         annotators_stats["ghinwa"] += 1

        # # new annotations
        # elif annotation_method in ["one_model_unspecified", "llm_different"]:
        #     if (not current_annotator or current_annotator == "Unknown") and (
        #         record["needs_manual_annotation"] == False
        #     ):
        #         record["annotated_by"] = "ghinwa"
        #         annotators_stats["ghinwa"] += 1

        # Count records that still need annotation
        if record.get("needs_manual_annotation", False):
            stats["still_needs_annotation"] += 1

        # Count records that already had annotations
        if record.get(correct_field_col) and not record.get(
            "needs_manual_annotation", False
        ):
            stats["already_had_annotations"] += 1

        processed_records.append(record)

    # 5. Log statistics
    logger.warning("Annotation statistics:")
    for key, value in stats.items():
        logger.warning(f"  {key}: {value}")

    logger.warning("Annotators statistics:")
    for key, value in annotators_stats.items():
        logger.warning(f"  {key}: {value}")

    # 6. Save the final evaluation dataset
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(processed_records, f, indent=4)
    logger.warning(f"Saved final evaluation dataset to {output_json}")

    # 7. Save as Excel for easier viewing
    df = pd.DataFrame(processed_records)
    df.to_excel(output_excel, index=False)
    logger.warning(f"Saved final evaluation dataset to {output_excel}")

    elapsed_time = time.time() - start_time
    logger.warning(f"Total processing time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
