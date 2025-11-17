#!/usr/bin/env python3
# File: src/scripts/S10c_clean_dataset.py

import json
import hashlib
from pathlib import Path
import click
import pandas as pd


def _extract_acronym(full_name: str) -> str:
    if not isinstance(full_name, str):
        return ""
    full_name = full_name.strip()
    if "(" in full_name and full_name.endswith(")"):
        inside = full_name.rsplit("(", 1)[-1].rstrip(")").strip()
        if 1 <= len(inside) <= 12 and inside.replace("-", "").replace("_", "").isalnum():
            return inside
    return ""


def _make_pair_hash(left_id: str, right_id: str) -> str:
    return hashlib.md5(f"{left_id}::{right_id}".encode("utf-8")).hexdigest()

@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--raw_json", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              required=True, help="Raw dataset JSON with registry_name/full_name/etc.")
@click.option("--output_dir", type=click.Path(file_okay=False, path_type=Path),
              required=True, help="Directory to save cleaned dataset and metadata.")
@click.option("--preloaded_embeddings", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Optional: local embeddings parquet from S10a_preload_embeddings.py")
@click.option("--preloaded_medc", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Optional: local medc parquet from S10_preload_geographical_areas.py")
@click.option("--preloaded_geoarea", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Optional: local geoarea parquet from S10_preload_medical_conditions.py")
def main(raw_json: Path, output_dir: Path,
         preloaded_embeddings: Path, preloaded_medc: Path, preloaded_geoarea: Path
         ):

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_json(raw_json)

    cleaned_df = pd.DataFrame({
        "pair_hash_id": [_make_pair_hash(str(lid), str(rid))
                         for lid, rid in zip(df["object_id"], df["alias_object_id"])],
        "label": df["final_label"].astype(int),
        "left_name": df["registry_name"].astype(str),
        "left_acronym": df["full_name"].apply(_extract_acronym),
        "right_name": df["alias"].astype(str),
        "right_acronym": df["alias_full_name"].apply(_extract_acronym),
        "left_object_id": df["object_id"].astype(str),
        "right_object_id": df["alias_object_id"].astype(str),
    })

    if preloaded_embeddings:
        emb = pd.read_parquet(preloaded_embeddings)
        cleaned_df = cleaned_df.merge(
            emb.rename(columns={"registry_name_embedding": "left_embedding",
                                "object_id": "left_object_id"}),
            on="left_object_id", how="left"
        ).merge(
            emb.rename(columns={"registry_name_embedding": "right_embedding",
                                "object_id": "right_object_id"}),
            on="right_object_id", how="left"
        )

    if preloaded_medc:
        medc = pd.read_parquet(preloaded_medc)

        def rename(df, prefix):
            out = df.copy()
            out.columns = [f"{prefix}_{c}" for c in out.columns]
            return out

        cleaned_df = cleaned_df.merge(rename(medc, "left"), on="left_object_id", how="left")
        cleaned_df = cleaned_df.merge(rename(medc, "right"), on="right_object_id", how="left")
    else:
        cleaned_df["left_medical_condition"] = ""
        cleaned_df["right_medical_condition"] = ""

    if preloaded_geoarea:
        geoarea = pd.read_parquet(preloaded_geoarea)

        def rename(df, prefix):
            out = df.copy()
            out.columns = [f"{prefix}_{c}" for c in out.columns]
            return out

        cleaned_df = cleaned_df.merge(rename(geoarea, "left"), on="left_object_id", how="left")
        cleaned_df = cleaned_df.merge(rename(geoarea, "right"), on="right_object_id", how="left")
    else:
        cleaned_df["left_geographical_area"] = ""
        cleaned_df["right_geographical_area"] = ""

    # # FInally rename left_object_id and right_object_id to left_raw_registry_id and right_raw_registry_id
    # cleaned_df = cleaned_df.rename(columns={
    #     "left_object_id": "left_raw_registry_id",
    #     "right_object_id": "right_raw_registry_id"
    # })

    # save outputs
    cleaned_path = output_dir / "clean_dataset.json"
    cleaned_df.to_json(cleaned_path, orient="records", indent=2)

    metadata = {
        "original_file": str(raw_json),
        "rows": len(cleaned_df),
        "columns": list(cleaned_df.columns),
        "label_distribution": cleaned_df["label"].value_counts().to_dict(),
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    click.echo(f"[SUCCESS] Cleaned dataset → {cleaned_path}")


if __name__ == "__main__":
    main()
