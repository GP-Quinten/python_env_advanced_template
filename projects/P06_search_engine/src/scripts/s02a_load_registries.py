import json
import click
import boto3
import os


@click.command()
@click.option(
    "--ema_registries",
    type=click.Path(exists=True),
    required=True,
    help="Path to EMA registries JSON file."
)
@click.option(
    "--output_registries",
    type=click.Path(),
    required=True,
    help="Path to output raw registries JSON file."
)
@click.option(
    "--output_dir",
    type=click.Path(),
    required=True,
    help="Directory to store output files and metadata."
)
def main(ema_registries, output_registries, output_dir):
    """
    Load registries from:
    - S3 bucket with official registry publications
    - EMA registries JSON file

    Merge them into a single raw dataset (no cleaning, no filtering).
    """

    # S3 setup (hardcoded for now)
    S3_BUCKET_NAME = "s3-common-dev20231214174437248800000002"
    PATH_DATA_REGISTRIES_PUBLICATIONS = (
        "registry_data_catalog_experiments/P04_official_reg_db_creation/"
        "datasets_versions/update_medical_condition/registry_data/"
    )

    click.echo("🔹 Fetching registries from S3...")

    s3 = boto3.client("s3")
    registries = []

    for file in s3.list_objects(Bucket=S3_BUCKET_NAME, Prefix=PATH_DATA_REGISTRIES_PUBLICATIONS).get("Contents", []):
        for line in s3.get_object(Bucket=S3_BUCKET_NAME, Key=file["Key"])["Body"].iter_lines():
            d_registry = json.loads(line)
            registry = {
                "registry_id": d_registry["object_id"],
                "registry_name": d_registry["registry_name"],
                "registry_acronym": d_registry["acronym"],
                "registry_locations": d_registry["geographical_area"] or [],
                "registry_conditions": d_registry["medical_condition"] or {},
                "registry_occurrences": d_registry["number_of_occurrences"],
                "source": "S3"
            }
            registries.append(registry)

    click.echo(f"✅ Loaded {len(registries)} registries from S3")

    click.echo(f"🔹 Loading EMA registries from {ema_registries}...")
    with open(ema_registries, "r") as file:
        l_registries = json.load(file)

    l_registries = sorted(l_registries, key=lambda x: x["title"])
    last_id = max(registries, key=lambda registry: registry["registry_id"])["registry_id"]

    for idx, d_registry in enumerate(l_registries, start=1):
        registry = {
            "registry_id": last_id + idx,
            "registry_name": d_registry["title"],
            "registry_acronym": d_registry.get("acronym"),
            "registry_locations": d_registry.get("geographical_area") or [],
            "registry_conditions": {condition: 1 for condition in d_registry.get("medical_condition") or []},
            "registry_occurrences": None,  # always keep EMA registries
            "source": "EMA"
        }
        registries.append(registry)

    click.echo(f"✅ Added {len(l_registries)} EMA registries")

    # Save registries to output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_registries_path = os.path.join(output_dir, os.path.basename(output_registries))
    with open(output_registries_path, "w") as file:
        json.dump(registries, file, indent=4)
    click.echo(f"🎉 Raw registries dataset saved to {output_registries_path}")

    # --- Add metadata.json creation ---

    metadata = {
        "output_file": output_registries_path,
        "num_total_registries": len(registries),
        "num_s3_registries": len(registries) - len(l_registries),
        "num_ema_registries": len(l_registries),
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as meta_file:
        json.dump(metadata, meta_file, indent=4)
    click.echo(f"📄 Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
