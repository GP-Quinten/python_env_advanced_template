import os
import json
import click
import logging
import pandas as pd
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def format_percentage(value):
    """Format a float as percentage with 1 decimal place."""
    return f"{value * 100:.1f}%"


@click.command()
@click.option(
    "--extraction_results",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to extraction results JSON file from R32.",
)
@click.option(
    "--model",
    type=str,
    required=True,
    help="Model name (e.g., small_mistral, large_mistral).",
)
@click.option(
    "--field",
    type=str,
    required=True,
    help="Field name (e.g., medical_condition, outcome_measure).",
)
@click.option(
    "--output_json",
    type=str,
    required=True,
    help="Path to output JSON file for performance metrics.",
)
def main(extraction_results, model, field, output_json):
    """
    Compute and save performance metrics from field-specific extraction results.

    This script:
    1. Loads the extraction comparison results for a specific field
    2. Calculates overall accuracy and counts of correct/incorrect extractions
    3. Breaks down metrics by labeling reason (model_agreement, llm_same, etc.)
    4. Saves all metrics to a JSON file
    """
    start_time = pd.Timestamp.now()
    logger.warning(f"Starting performance calculation at {start_time}")

    # Ensure output directory exists
    out_dir = Path(output_json).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load extraction results
    with open(extraction_results, "r", encoding="utf-8") as f:
        results = json.load(f)

    logger.warning(f"Loaded {len(results)} records from {extraction_results}")
    logger.warning(f"Computing metrics for field '{field}' and model '{model}'")

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)

    # Calculate overall metrics
    total_samples = len(df)
    correct_extractions = int(df["final_label"].sum())
    incorrect_extractions = int(total_samples - correct_extractions)
    accuracy = correct_extractions / total_samples if total_samples > 0 else 0

    # Group by labeling_reason and final_label
    reason_groups = (
        df.groupby(["labeling_reason", "final_label"]).size().reset_index(name="count")
    )

    # Structure the breakdown data
    correct_reasons = {}
    incorrect_reasons = {}

    for _, row in reason_groups.iterrows():
        reason = row["labeling_reason"]
        is_correct = row["final_label"] == 1
        count = int(row["count"])

        if is_correct:
            correct_reasons[reason] = count
        else:
            incorrect_reasons[reason] = count

    # Create a structured breakdown
    reason_breakdown = {
        "correct_extractions": {
            "total": correct_extractions,
            "percentage": accuracy,
            "reasons": correct_reasons,
        },
        "incorrect_extractions": {
            "total": incorrect_extractions,
            "percentage": 1 - accuracy,
            "reasons": incorrect_reasons,
        },
    }

    # Create the summary structure
    summary = {
        "total_samples": total_samples,
        "accuracy": accuracy,
        "correct_extractions": correct_extractions,
        "incorrect_extractions": incorrect_extractions,
    }

    # Collect all metrics into a single dictionary
    all_metrics = {
        "field": field,
        "model": model,
        "summary": summary,
        "reason_breakdown": reason_breakdown,
    }

    # Display summary metrics in logs with the requested format
    logger.warning(
        f"Overall performance metrics for field '{field}' with model '{model}':"
    )
    logger.warning(f"  Total samples: {summary['total_samples']}")
    logger.warning(
        f"  Accuracy: {format_percentage(accuracy)} ({correct_extractions} / {total_samples})"
    )

    # Display correct extractions breakdown
    logger.warning(
        f"  Correct extractions: {correct_extractions} ({format_percentage(accuracy)})"
    )
    for reason, count in correct_reasons.items():
        percentage = count / total_samples if total_samples > 0 else 0
        logger.warning(f"        {reason}: {count} ({format_percentage(percentage)})")

    # Display incorrect extractions breakdown
    logger.warning(
        f"  Incorrect extractions: {incorrect_extractions} ({format_percentage(1 - accuracy)})"
    )
    for reason, count in incorrect_reasons.items():
        percentage = count / total_samples if total_samples > 0 else 0
        logger.warning(f"        {reason}: {count} ({format_percentage(percentage)})")

    # Save metrics to JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4, ensure_ascii=False)

    logger.warning(f"Saved performance metrics to {output_json}")

    end_time = pd.Timestamp.now()
    elapsed = (end_time - start_time).total_seconds()
    logger.warning(f"Completed performance calculation in {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
