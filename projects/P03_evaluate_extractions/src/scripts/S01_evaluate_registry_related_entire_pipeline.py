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
from sklearn.metrics import precision_score, recall_score, confusion_matrix

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
        parsed_content = response.get("Registry related", "").lower()
        return {"Registry related": parsed_content}
    except Exception as e:
        logger.error(f"Error parsing Mistral response: {e}")
        logger.error(f"Original response: {response}")
        return {"Registry related": "error"}


def parse_openai_response(response):
    """
    Parse OpenAI response to extract the registry related field.
    """
    try:
        # Extract the content from the OpenAI response structure
        content = response["choices"][0]["message"]["content"]

        # Parse the content as JSON
        parsed_content = json.loads(content)

        # Extract the registry related field and convert to lowercase
        registry_related = parsed_content.get("Registry related", "").lower()

        return {"Registry related": registry_related}
    except Exception as e:
        logger.error(f"Error parsing OpenAI response: {e}")
        logger.error(f"Original response: {response}")
        return {"Registry related": "error"}


def normalize_binary_values(value):
    """
    Normalize binary values to a consistent format (yes/no).
    Handles true/false, yes/no conversions.
    """
    value = str(value).lower()
    if value in ["true", "yes"]:
        return "yes"
    elif value in ["false", "no"]:
        return "no"
    else:
        return value


def calculate_binary_metrics(
    eval_df: pd.DataFrame, correct_col: str, model_col: str
) -> dict:
    """
    Calculate binary classification metrics for a pair of binary columns.

    Args:
        eval_df: DataFrame containing the ground truth and predictions.
        correct_col: Name of the column with ground-truth values ("yes"/"no" or "true"/"false").
        model_col: Name of the column with model-predicted values ("yes"/"no").

    Returns:
        A dict with TP, FP, TN, FN, precision, recall, f1_score, and accuracy.
    """
    # Convert textual labels to binary (1 = yes/true, 0 = no/false)
    # First normalize both columns to the same format (yes/no)
    eval_df["correct_normalized"] = eval_df[correct_col].apply(normalize_binary_values)
    eval_df["model_normalized"] = eval_df[model_col].apply(normalize_binary_values)

    # Then convert to binary
    eval_df["true_binary"] = (eval_df["correct_normalized"] == "yes").astype(int)
    eval_df["pred_binary"] = (eval_df["model_normalized"] == "yes").astype(int)

    # Compute confusion matrix (handles edge cases internally)
    cm = confusion_matrix(eval_df["true_binary"], eval_df["pred_binary"]).tolist()
    if len(cm) == 2 and len(cm[0]) == 2:
        tn, fp = cm[0]
        fn, tp = cm[1]
    else:
        # Single-class edge cases
        total = len(eval_df)
        positives = eval_df["true_binary"].sum()
        preds = eval_df["pred_binary"].sum()
        if positives == 0 and preds == 0:
            tn, fp, fn, tp = total, 0, 0, 0
        elif positives == total and preds == total:
            tn, fp, fn, tp = 0, 0, 0, total
        else:
            tn = fp = fn = tp = 0
            logger.warning(
                "Could not derive full confusion matrix; defaulting to zeros"
            )

    precision = precision_score(
        eval_df["true_binary"], eval_df["pred_binary"], zero_division=0
    )
    recall = recall_score(
        eval_df["true_binary"], eval_df["pred_binary"], zero_division=0
    )
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )
    accuracy = (tp + tn) / len(eval_df) if len(eval_df) > 0 else 0

    return {
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "accuracy": float(accuracy),
    }


@click.command()
@click.option(
    "--base_pubmed_dataset_jsonl",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to input JSONL file with PubMed records.",
)
@click.option(
    "--prompt_txt",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the annotation prompt text file.",
)
@click.option(
    "--model_config",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the model configuration JSON file.",
)
@click.option(
    "--eval_dataset_json",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the evaluation dataset JSON file.",
)
@click.option(
    "--output_excel",
    type=str,
    required=True,
    help="Path to output Excel file with extraction results.",
)
@click.option(
    "--output_perf_json",
    type=str,
    required=True,
    help="Path to output JSON file with performance metrics.",
)
@click.option(
    "--output_records_jsonl",
    type=str,
    required=True,
    help="Path to output JSONL file with one line per record (object_id, prompt, response).",
)
def main(
    base_pubmed_dataset_jsonl: str,
    prompt_txt: str,
    model_config: str,
    eval_dataset_json: str,
    output_excel: str,
    output_perf_json: str,
    output_records_jsonl: str,
) -> None:
    """
    Evaluate performance of registry_related extraction with a specified model.
    """
    start_time = time.time()

    # Ensure output directory exists
    out_dir = Path(output_excel).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure output records directory exists
    records_dir = Path(output_records_jsonl).parent
    records_dir.mkdir(parents=True, exist_ok=True)

    # Load model configuration
    with open(model_config, "r", encoding="utf-8") as f:
        model_cfg = json.load(f)

    model_name = model_cfg.get("model", "unknown")
    logger.warning(f"Using model: {model_name}")

    # Determine if we're using OpenAI or Mistral model
    is_openai_model = "gpt" in model_name.lower() or "o3" in model_name.lower()

    # Check API keys
    if is_openai_model and not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY not found in environment variables")
    elif not is_openai_model and not MISTRAL_API_KEY:
        raise EnvironmentError("MISTRAL_API_KEY not found in environment variables")

    # Load evaluation dataset
    with open(eval_dataset_json, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    eval_df = pd.DataFrame(eval_data).rename(
        columns={"registry_related": "correct_registry_related"}
    )
    logger.warning(f"Evaluation dataset contains {len(eval_df)} records")

    # Load annotation prompt
    annotation_prompt = Path(prompt_txt).read_text(encoding="utf-8").strip()

    # Load PubMed records (JSONL)
    records = []
    with open(base_pubmed_dataset_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    records = records[:5]
    logger.warning(f"Loaded {len(records)} PubMed records")

    # Extract object_ids from evaluation dataset
    eval_object_ids = set(eval_df["object_id"].tolist())

    # Filter records to only those in the evaluation dataset
    records = [rec for rec in records if rec.get("object_id") in eval_object_ids]
    logger.warning(f"Filtered to {len(records)} records in evaluation dataset")

    # Prepare prompts for LLMs
    prompts = []
    for rec in records:
        object_id = rec.get("object_id", "<unknown>")
        title = rec.get("title", "<no title>")
        abstract = rec.get("abstract", "<no abstract>")
        full_prompt = f"{annotation_prompt}\nText_to_analyze:\nTitle: {title}\nAbstract: {abstract}"
        prompts.append({"prompt": full_prompt, "custom_id": object_id})

    # Create a list to store the records with object_id, prompt, and raw response
    prompt_response_records = []

    # ----- BATCH INFERENCE -----
    print(f"Starting batch inference with {model_name}...")

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
            parsed_result["custom_id"] = result["custom_id"]
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

    print("Batch inference complete")

    # Build results DataFrame
    results = []
    for rec in records:
        object_id = rec.get("object_id", "<unknown>")
        title = rec.get("title", "<no title>")
        abstract = rec.get("abstract", "<no abstract>")

        resp = next((x for x in llm_responses if x["custom_id"] == object_id), None)

        if resp:
            registry_related = resp.get("Registry related", "").lower()
            results.append(
                {
                    "object_id": object_id,
                    "title": title,
                    "abstract": abstract,
                    "inferred_registry_related": registry_related,
                }
            )

    results_df = pd.DataFrame(results)

    # Merge with evaluation data
    merged_df = eval_df.merge(
        results_df[["object_id", "inferred_registry_related"]],
        on="object_id",
        how="left",
    )

    # # Normalize values for comparison
    # merged_df["normalized_registry_related"] = merged_df["registry_related"].apply(
    #     normalize_binary_values
    # )
    # merged_df["normalized_correct_registry_related"] = merged_df[
    #     "correct_registry_related"
    # ].apply(normalize_binary_values)

    # Add a column to check if the model's prediction matched the ground truth
    merged_df["is_correct"] = (
        merged_df["inferred_registry_related"] == merged_df["correct_registry_related"]
    )

    # Save to Excel
    merged_df.to_excel(output_excel, index=False)
    print(f"Saved extraction results to {output_excel}")

    # Compute metrics
    metrics = calculate_binary_metrics(
        merged_df, "correct_registry_related", "inferred_registry_related"
    )

    # Display metrics
    metrics_to_show = {
        "precision": f"{metrics['precision']:.1%}",
        "recall": f"{metrics['recall']:.1%}",
        "f1_score": f"{metrics['f1_score']:.1%}",
        "accuracy": f"{metrics['accuracy']:.1%}",
    }
    logger.warning(f"Performance metrics for {model_name}: {metrics_to_show}")

    # Save metrics to JSON
    performance_data = {
        "field": "inferred_registry_related",
        "model": model_name,
        "total_samples": len(merged_df),
        "binary_metrics": metrics,
    }

    with open(output_perf_json, "w", encoding="utf-8") as fp:
        json.dump(performance_data, fp, indent=4)
    print(f"Saved performance metrics to {output_perf_json}")

    # Save records to JSONL file
    with open(output_records_jsonl, "w", encoding="utf-8") as fp:
        for record in prompt_response_records:
            fp.write(json.dumps(record) + "\n")
    print(f"Saved {len(prompt_response_records)} records to {output_records_jsonl}")

    elapsed_total = time.time() - start_time
    logger.warning(f"Total time taken: {elapsed_total:.2f} seconds")


if __name__ == "__main__":
    main()
