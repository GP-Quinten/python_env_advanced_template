#!/usr/bin/env python3
"""
Split a cleaned labeled pair dataset into train/test sets with stratified sampling.
Input must have at least:
    pair_hash_id, label, left_name, right_name
"""

import json
from pathlib import Path
import click
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import hashlib


# NAME the columns
COL_PAIR_ID = "pair_hash_id"
COL_LABEL = "label"
LEFT_PREFIX = "left"
RIGHT_PREFIX = "right"
COL_NAME = "name"
COL_ACRONYM = "acronym"
COL_OBJECT_ID = "object_id"
COL_EMBEDDING = "embedding"
COL_MEDICAL_CONDITION = "medical_condition"
COL_GEOGRAPHICAL_AREA = "geographical_area"


def _make_pair_hash(left_id: str, right_id: str) -> str:
    return hashlib.md5(f"{left_id}::{right_id}".encode("utf-8")).hexdigest()

def _add_exact_match_samples(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    # on df_train, add some exact matches to have this type of samples in train set
    # select random 5% of the df rows, and only the 'left' columns
    left_cols = [
        f"{LEFT_PREFIX}_{COL_NAME}", 
        f"{LEFT_PREFIX}_{COL_ACRONYM}", 
        f"{LEFT_PREFIX}_{COL_OBJECT_ID}", 
        f"{LEFT_PREFIX}_{COL_EMBEDDING}",
        f"{LEFT_PREFIX}_{COL_MEDICAL_CONDITION}", 
        f"{LEFT_PREFIX}_{COL_GEOGRAPHICAL_AREA}"
    ]
    df_sample_exact_match = df.sample(frac=0.05, random_state=random_state).reset_index(drop=True).loc[:, left_cols]
    # copy left columns to right columns
    for col in df_sample_exact_match.columns:
        new_col = col.replace(f"{LEFT_PREFIX}_", f"{RIGHT_PREFIX}_")
        df_sample_exact_match[new_col] = df_sample_exact_match[col]
    # generate a hash key for each row
    df_sample_exact_match[COL_PAIR_ID] = [_make_pair_hash(str(lid), str(rid))
                                        for lid, rid in zip(df_sample_exact_match[f"{LEFT_PREFIX}_{COL_OBJECT_ID}"],
                                                            df_sample_exact_match[f"{RIGHT_PREFIX}_{COL_OBJECT_ID}"])]
    # set label to 1
    df_sample_exact_match[COL_LABEL] = 1
    # reorder columns to match df_train
    df_sample_exact_match = df_sample_exact_match[df.columns]
    return pd.concat([df, df_sample_exact_match], ignore_index=True), len(df_sample_exact_match)

@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--dataset_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to cleaned labeled pairs JSON file.",
)
@click.option("--test_size", default=0.2, show_default=True, type=float)
@click.option("--random_state", default=42, show_default=True, type=int)
@click.option(
    "--output_dir", type=click.Path(file_okay=False, path_type=Path), required=True
)
def main(dataset_json: Path, test_size: float, random_state: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"[INFO] Reading cleaned dataset: {dataset_json}")
    df = pd.read_json(dataset_json)
    click.echo(f"[INFO] Loaded {len(df)} rows.")

    required_cols = [
        COL_PAIR_ID,
        COL_LABEL,
        f"{LEFT_PREFIX}_{COL_NAME}",
        f"{RIGHT_PREFIX}_{COL_NAME}",
        f"{LEFT_PREFIX}_{COL_EMBEDDING}",
        f"{RIGHT_PREFIX}_{COL_EMBEDDING}",
        f"{LEFT_PREFIX}_{COL_MEDICAL_CONDITION}",
        f"{RIGHT_PREFIX}_{COL_MEDICAL_CONDITION}",
        f"{LEFT_PREFIX}_{COL_GEOGRAPHICAL_AREA}",
        f"{RIGHT_PREFIX}_{COL_GEOGRAPHICAL_AREA}",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise click.ClickException(f"Missing required columns: {missing}")

    y = df[COL_LABEL].astype(int).values
    indices = np.arange(len(df))

    click.echo(f"[INFO] Splitting with stratified sampling (test_size={test_size})")
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=y
    )

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_train, n_added_exact = _add_exact_match_samples(df_train, random_state)
    click.echo(f"[INFO] Added {n_added_exact} exact match samples to training set.")
    click.echo(f"[INFO] Training set size after augmentation: {len(df_train)}")

    df_test = df.iloc[test_idx].reset_index(drop=True)

    # Save data
    df_train.to_json(output_dir / "train_data.json", orient="records", indent=2)
    df_test.to_json(output_dir / "test_data.json", orient="records", indent=2)
    np.save(output_dir / "train_indices.npy", train_idx)
    np.save(output_dir / "test_indices.npy", test_idx)

    # Metadata
    metadata = {
        "source_file": str(dataset_json),
        "total_rows": len(df),
        "test_size": test_size,
        "random_state": random_state,
        "train_rows": len(df_train),
        "test_rows": len(df_test),
        "class_distribution_train": {
            "positive": int(df_train[COL_LABEL].sum()),
            "negative": int(len(df_train) - df_train[COL_LABEL].sum()),
        },
        "class_distribution_test": {
            "positive": int(df_test[COL_LABEL].sum()),
            "negative": int(len(df_test) - df_test[COL_LABEL].sum()),
        },
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    click.echo(f"[SUCCESS] Train/test splits saved to {output_dir}")


if __name__ == "__main__":
    main()
