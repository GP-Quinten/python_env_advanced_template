#!/usr/bin/env python3
# File: src/scripts/S10b_preload_medical_conditions.py

import click
from pathlib import Path
import boto3
import pandas as pd
from P07_fuzzy_dedup import config
import json
from P07_fuzzy_dedup.utils.s3_io_functions import load_jsonl_from_s3


def _process_medical_conditions(medical_condition_dict):
    """
    Process a medical_condition dictionary to extract the top 3 medical conditions
    with the highest scores and return them as a space-separated string.
    
    Returns an empty string if the input is not a valid dictionary.
    """
    if not isinstance(medical_condition_dict, dict) or not medical_condition_dict:
        return ""
    sorted_conditions = sorted(medical_condition_dict.items(), key=lambda x: x[1], reverse=True)
    names = [name for name, _ in sorted_conditions]
    return " ".join(names[:3])


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--s3_input_registry_medc_jsonl_template", type=str, required=True,
              help="S3 template path pointing to medical condition JSONL files.")
@click.option("--output_path", type=click.Path(dir_okay=False, path_type=Path),
              required=True, help="Where to save processed medc parquet.")
def main(s3_input_registry_medc_jsonl_template: str, output_path: Path):
    medc_input_dir = Path(s3_input_registry_medc_jsonl_template).parent
    medc_input_dir_str = str(medc_input_dir)

    s3 = boto3.client("s3")
    response = s3.list_objects_v2(
        Bucket=config.BUCKET_NAME_DEV, Prefix=medc_input_dir_str
    )
    total_batches = len(response.get("Contents", []))
    click.echo(f"[INFO] Found {total_batches} medc batch files")

    registry_dataset = []
    for batch_number in range(1, total_batches + 1):
        file_name = f"{batch_number}.jsonl"
        batch = load_jsonl_from_s3(config.BUCKET_NAME_DEV, medc_input_dir, file_name)
        registry_dataset.extend(batch)

    medc_df = pd.DataFrame(registry_dataset).loc[:, ["object_id", "medical_condition"]]
    medc_df["object_id"] = medc_df["object_id"].astype(str)

    # Replace medical_condition with top 3 medical conditions
    medc_df["medical_condition"] = medc_df["medical_condition"].apply(_process_medical_conditions)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    medc_df.to_parquet(output_path, index=False)
    click.echo(f"[SUCCESS] Saved processed medc_df to {output_path}")

    # save 1 json with object_id to medical_condition mapping for local tracking
    medc_mapping = dict(zip(medc_df["object_id"], medc_df["medical_condition"]))
    # select only top 10 
    check_medc_mapping = {k: v for k, v in list(medc_mapping.items())[:10]}
    mapping_output_path = output_path.with_suffix(".json")
    with open(mapping_output_path, "w", encoding="utf-8") as f:
        json.dump(check_medc_mapping, f, ensure_ascii=False, indent=2)
    click.echo(f"[SUCCESS] Saved medical_condition mapping to {mapping_output_path}")


if __name__ == "__main__":
    main()
