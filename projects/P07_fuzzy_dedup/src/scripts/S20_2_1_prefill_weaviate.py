import json
import os
import pandas as pd
import weaviate
import click
from pathlib import Path
import hashlib

from dotenv import load_dotenv
load_dotenv()

from weaviate.classes.config import Configure, Property, DataType
from weaviate.util import generate_uuid5

from tqdm import tqdm


def _make_pair_hash(left_id: str, right_id: str) -> str:
    ids = sorted([str(left_id), str(right_id)])
    return hashlib.md5(f"{ids[0]}::{ids[1]}".encode("utf-8")).hexdigest()


########
# --- Registry DataFrame Builder ---
########
def build_registry_from_labeled_pairs(labeled_df: pd.DataFrame) -> pd.DataFrame:
    left = labeled_df[["left_object_id", "left_name", "left_acronym"]].rename(columns={
        "left_object_id": "object_id", "left_name": "registry_name", "left_acronym": "acronym"
    })
    right = labeled_df[["right_object_id", "right_name", "right_acronym"]].rename(columns={
        "right_object_id": "object_id", "right_name": "registry_name", "right_acronym": "acronym"
    })
    reg = pd.concat([left, right], ignore_index=True)
    reg["object_id"] = reg["object_id"].astype(str)
    reg = reg.groupby("object_id", as_index=False).agg({"registry_name": "first", "acronym": "first"})
    reg["registry_name"] = reg["registry_name"].fillna("").astype(str)
    reg["acronym"] = reg["acronym"].fillna("").astype(str)
    reg["combined"] = (reg["registry_name"] + " " + reg["acronym"]).str.strip()
    return reg

########
# --- Weaviate Connection Helper ---
########
def _connect_from_env() -> weaviate.Client:
    """Connect to Weaviate using environment variables (no fallbacks).

    Expects WEAVIATE__HTTP_HOST, WEAVIATE__HTTP_PORT, WEAVIATE__GRPC_HOST, WEAVIATE__GRPC_PORT to be set.
    """
    http_host = os.getenv("WEAVIATE__HTTP_HOST")
    http_port = os.getenv("WEAVIATE__HTTP_PORT")
    grpc_host = os.getenv("WEAVIATE__GRPC_HOST")
    grpc_port = os.getenv("WEAVIATE__GRPC_PORT")

    if not all([http_host, http_port, grpc_host, grpc_port]):
        raise RuntimeError(
            "WEAVIATE__HTTP_HOST, WEAVIATE__HTTP_PORT, WEAVIATE__GRPC_HOST and WEAVIATE__GRPC_PORT must be set in the environment"
        )

    http_port = int(http_port)
    grpc_port = int(grpc_port)

    # include Mistral API key header if provided
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    headers = {"X-Mistral-Api-Key": mistral_api_key} 

    client = weaviate.connect_to_custom(
        http_host=http_host,
        http_port=http_port,
        http_secure=False,
        grpc_host=grpc_host,
        grpc_port=grpc_port,
        grpc_secure=False,
        headers=headers,
    )

    if not client.is_ready():
        raise RuntimeError("Weaviate client is not ready")

    return client

########
# --- Main CLI Entrypoint ---
########
@click.command()
@click.option("--annotated_json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--output_dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--collection_name", type=str, default="RegistryCollection")
def main(annotated_json: Path, output_dir: Path, collection_name: str):
    """Flatten annotated dataset into unique registries and prefill Weaviate collection.

    The collection will have a single text property named 'registry_name_with_acronnym'.
    """
    ########
    # --- Load and Flatten Annotated Data ---
    ########
    annotated_json = Path(annotated_json)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotated data
    labeled_df = pd.read_json(annotated_json)
    click.echo(f"Loaded annotated JSON '{annotated_json}' with {len(labeled_df)} records")

    # add pair_hash_id column to annotated dataframe and save a copy
    labeled_df["left_object_id"] = labeled_df["left_object_id"].astype(str)
    labeled_df["right_object_id"] = labeled_df["right_object_id"].astype(str)
    labeled_df["pair_hash_id"] = labeled_df.apply(
        lambda r: _make_pair_hash(r["left_object_id"], r["right_object_id"]), axis=1
    )
    annotated_with_hash_path = output_dir / "annotated_with_pair_hash.json"
    labeled_df.to_json(annotated_with_hash_path, orient="records", indent=4)
    click.echo(f"Saved annotated JSON with pair_hash_id to '{annotated_with_hash_path}'")

    ########
    # --- Flatten to unique registries by object id ---
    ########
    registry_df = build_registry_from_labeled_pairs(labeled_df)
    click.echo(f"Flattened to {len(registry_df)} unique registries")

    ########
    # --- Connect to Weaviate ---
    ########
    client = _connect_from_env()
    click.echo("Connected to Weaviate")
    click.echo(f"Existing collections: {client.collections.list_all()}")

    ########
    # --- Create collection if not exists ---
    ########
    # delete all collections if they exist
    if collection_name not in client.collections.list_all():
        click.echo(f"Creating collection '{collection_name}'")
        client.collections.create(
            name=collection_name,
            properties=[
                Property(name="object_id", data_type=DataType.TEXT),
                Property(name="registry_name", data_type=DataType.TEXT),
                Property(name="acronym", data_type=DataType.TEXT),
            ],
            vector_config=Configure.Vectors.text2vec_mistral(
                source_properties=["registry_name", "acronym"]
            ),
        )
        click.echo(f"Created collection '{collection_name}'")
    else:
        click.echo(f"Collection '{collection_name}' already exists")

    ########
    # --- Insert data into Weaviate ---
    ########
    collection = client.collections.get(collection_name)
    with collection.batch.fixed_size(batch_size=100) as batch:
        for _, row in tqdm(registry_df.iterrows(), total=len(registry_df)):
            obj_properties = {
                "object_id": row["object_id"],
                "registry_name": row["registry_name"],
                "acronym": row["acronym"],
            }
            obj_uuid = generate_uuid5(obj_properties)

            if not collection.data.exists(obj_uuid):
                batch.add_object(
                    properties=obj_properties,
                    uuid=obj_uuid
                )
            else:
                click.echo(f"Object with UUID {obj_uuid} already exists, skipping.")


    click.echo(f"Inserted {len(registry_df)} objects into collection '{collection_name}'")


    ########
    # --- Close existing connections (if any) ---
    ########
    client.close()

    ########
    # --- Save registry dataframe to json ---
    ########
    registry_json_path = output_dir / "registry_unique.json"
    registry_df.to_json(registry_json_path, orient="records", indent=4)
    click.echo(f"Saved unique registries to '{registry_json_path}'")

    ########
    # --- Save metadata json ---
    ########
    metadata = {
        "source_annotated_json": str(annotated_json.resolve()),
        "num_annotated_pairs": len(labeled_df),
        "num_unique_registries": len(registry_df),
        "weaviate_collection": collection_name,
    }
    metadata_json_path = output_dir / "metadata.json"
    with open(metadata_json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    click.echo(f"Saved metadata to '{metadata_json_path}'")

########
if __name__ == "__main__":
    main()
########