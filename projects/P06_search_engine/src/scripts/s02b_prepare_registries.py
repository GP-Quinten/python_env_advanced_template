import json
import click
import pandas as pd
import os


@click.command()
@click.option(
    "--input_registries",
    type=click.Path(exists=True),
    required=True,
    help="Path to raw registries JSON file (output of s02a_load_registries)."
)
@click.option(
    "--output_registries",
    type=click.Path(),
    required=True,
    help="Path to output cleaned registries JSON file."
)
@click.option(
    "--min_occurrences",
    type=int,
    default=3,
    show_default=True,
    help="Minimum number of occurrences required to keep a registry (except EMA registries)."
)
@click.option(
    "--output_dir",
    type=click.Path(),
    required=True,
    help="Directory to store output files and metadata."
)
def main(input_registries, output_registries, min_occurrences, output_dir):
    """
    Clean and prepare registries:
    - Keep registries with >= min_occurrences OR EMA registries
    - Drop None occurrences from conditions
    - Normalize locations into {location: count}
    """

    click.echo(f"🔹 Loading raw registries from {input_registries}...")
    with open(input_registries, "r") as file:
        registries = json.load(file)

    # Drop None occurrences in conditions
    registries = [
        {
            **registry,
            "registry_conditions": {
                condition: occurrence
                for condition, occurrence in registry["registry_conditions"].items()
                if occurrence is not None
            },
        }
        for registry in registries
    ]

    # Apply filtering by min_occurrences
    registries = [
        registry
        for registry in registries
        if registry.get("registry_occurrences") is None or registry.get("registry_occurrences", 0) >= min_occurrences
    ]

    click.echo(f"✅ {len(registries)} registries kept after filtering (min_occurrences={min_occurrences})")

    # Normalize location occurrences
    locations_occurrences = (
        pd.DataFrame(registries)["registry_locations"]
        .explode()
        .str.capitalize()
        .str.strip()
        .value_counts()
        .to_dict()
    )

    registries = [
        {
            **registry,
            "registry_locations": {
                location.strip().capitalize(): locations_occurrences.get(location.strip().capitalize(), 0)
                for location in registry["registry_locations"]
            },
        }
        for registry in registries
    ]

    os.makedirs(output_dir, exist_ok=True)
    output_registries_path = os.path.join(output_dir, os.path.basename(output_registries))
    # Save final registries
    with open(output_registries_path, "w") as file:
        json.dump(registries, file, indent=4)

    click.echo(f"🎉 Cleaned registries dataset saved to {output_registries_path}")

    # --- Add metadata.json creation ---
    num_records = len(registries)
    unique_locations = set()
    unique_conditions = set()
    for registry in registries:
        if isinstance(registry["registry_locations"], dict):
            unique_locations.update(registry["registry_locations"].keys())
        elif isinstance(registry["registry_locations"], list):
            unique_locations.update(registry["registry_locations"])
        unique_conditions.update(registry["registry_conditions"].keys())

    metadata = {
        "output_file": output_registries_path,
        "num_records": num_records,
        "num_unique_locations": len(unique_locations),
        "num_unique_conditions": len(unique_conditions),
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as meta_file:
        json.dump(metadata, meta_file, indent=4)
    click.echo(f"📄 Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
