import os
import json
import boto3
import logging
from botocore.exceptions import ClientError

import click


# Config
BUCKET_NAME_DEV = "s3-common-dev20231214174437248800000002"
s3_input_registries_dir = "registry_data_catalog_experiments/P04_official_reg_db_creation/datasets_versions/update_medical_condition/registry_data"

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def load_jsonl_from_s3(bucket_name: str, folder_path: str, file_name: str) -> list:
    """
    Load a .jsonl file from S3 into a list of Python dicts.
    """
    file_path = f"s3://{bucket_name}/{folder_path}/{file_name}"
    logger.warning(f"Data will be loaded from {file_path}")
    s3 = boto3.client("s3")
    try:
        response = s3.get_object(
            Bucket=bucket_name,
            Key=f"{folder_path.rstrip('/')}/{file_name}"
        )
        data = response["Body"].read().decode("utf-8")
        return [json.loads(line) for line in data.splitlines() if line.strip()]
    except ClientError as e:
        logger.error(f"Error loading JSONL from S3: {e}")
        return []


@click.command()
@click.option(
    "--output-dir",
    default=".",
    show_default=True,
    help="Directory to save the output registry_dataset.json file."
)
def generate_dataset(output_dir):
    """
    Generate a dataset by loading all .jsonl batch files from S3, concatenating them,
    and saving the result locally as registry_dataset.json.
    """
    # Count total batches
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(
        Bucket=BUCKET_NAME_DEV,
        Prefix=s3_input_registries_dir
    )
    total_batches = len(response.get("Contents", []))

    # Load all batches
    registry_dataset = []
    for batch_number in range(1, total_batches + 1):
        file_name = f"{batch_number}.jsonl"
        batch = load_jsonl_from_s3(
            BUCKET_NAME_DEV,
            s3_input_registries_dir,
            file_name
        )
        registry_dataset.extend(batch)

    logger.warning(
        f"Loaded {len(registry_dataset)} records from the registry data, from {total_batches} batches."
    )

    # Save to local file (optional, for downstream use)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "registry_dataset.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(registry_dataset, f, ensure_ascii=False, indent=2)

    click.echo(f"Dataset generated and saved as {output_path}")


if __name__ == '__main__':
    generate_dataset()
