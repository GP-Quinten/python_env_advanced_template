#!/usr/bin/env python3
"""
Build candidate pairs from an unlabeled registry dataset using MinHash+LSH.

Input JSON must include at least:
  - object_id
  - registry_name
  - acronym

Output JSON contains:
  - pair_hash_id
  - left_name
  - left_acronym
  - right_name
  - right_acronym
  - left_object_id
  - right_object_id
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import click
import pandas as pd
import mmh3
from tqdm import tqdm
from P07_fuzzy_dedup.bucketer import LSHBucketer
from deduplication_model.normalization.utils import normalize_registry_names


def _pair_hash_id(left_id, right_id, *, seed: int = 13) -> str:
    """Deterministic 128-bit hash for an unordered pair."""
    a, b = (str(left_id), str(right_id))
    if a > b:
        a, b = b, a
    h = mmh3.hash128(f"{a}||{b}", seed=seed, signed=False)
    return f"{h:032x}"


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--dataset_json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Path to R01 registry_dataset.json (list of registry objects).")
@click.option("--output_dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Directory to write candidate_pairs.json")
@click.option("--best_params_json", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              required=True,
              help="Path to best_params.json to load LSH parameters")
def main(
    dataset_json: Path,
    output_dir: Path,
    best_params_json: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"[INFO] Reading dataset: {dataset_json}")
    df = pd.read_json(dataset_json)
    click.echo(f"[INFO] Loaded {len(df)} registry rows")

    required = ["object_id", "registry_name", "acronym"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise click.ClickException(f"Missing required columns: {missing}")

    # Map object_id → info (no normalization)
    info: Dict[Any, Dict[str, Any]] = {
        row["object_id"]: {
            "name": row["registry_name"],
            "acronym": row["acronym"],
        }
        for _, row in df.iterrows()
    }

    # Load LSH parameters (must be provided)
    params_raw = json.loads(Path(best_params_json).read_text(encoding="utf-8"))
    # Only keep valid LSHBucketer args
    lsh_keys = {"ngram", "num_perm", "bands", "seed", "min_block_size", "max_block_size"}
    params = {k: params_raw[k] for k in lsh_keys if k in params_raw}

    # Multi-pass LSH: raw, normalized, acronym, raw+acro, norm+acro
    records: Dict[str, Dict[str, Any]] = {}
    def run_pass(texts: List[str], desc: str):
        click.echo(f"[INFO] Pass: {desc}, indexing {len(texts)} docs")
        bucketer = LSHBucketer(**params)
        # Correctly pair each row.object_id with its text
        docs = [(row.object_id, txt) for row, txt in zip(df.itertuples(index=False), texts)]
        bucketer.build(tqdm(docs, desc=f"Index {desc}", unit="doc"))
        for l, r in bucketer.pairs():
            ph = _pair_hash_id(l, r, seed=params["seed"])
            if ph not in records:
                left, right = info[l], info[r]
                records[ph] = {
                    "pair_hash_id":    ph,
                    "left_name":       left["name"],
                    "left_acronym":    left["acronym"],
                    "right_name":      right["name"],
                    "right_acronym":   right["acronym"],
                    "left_object_id":  l,
                    "right_object_id": r,
                }

    raw = df["registry_name"].fillna("").astype(str).tolist()
    run_pass(raw, "raw")
    run_pass(normalize_registry_names(raw), "normalized")
    acr = df["acronym"].fillna("").astype(str).tolist()
    run_pass(acr, "acronym")
    raw_full = [f"{n} {a}" for n, a in zip(raw, acr)]
    run_pass(raw_full, "raw+acro")
    run_pass(normalize_registry_names(raw_full), "norm+acro")

    pairs_out = list(records.values())

    # Save a small sample (first 100 pairs) as readable JSON
    sample_path = output_dir / "candidate_pairs_sample.json"
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(pairs_out[:100], f, indent=2, ensure_ascii=False)

    # Save full output as optimized Parquet
    parquet_path = output_dir / "candidate_pairs.parquet"
    pd.DataFrame(pairs_out).to_parquet(parquet_path, index=False)

    click.echo(f"[SUCCESS] Wrote sample (100) pairs → {sample_path}")
    click.echo(f"[SUCCESS] Wrote full pairs as Parquet → {parquet_path}")

    # write metadata
    metadata = {
        "n_docs": len(df),
        "n_pairs": len(pairs_out),
        #        "stats": bucketer.stats(),
        "params": params
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    click.echo(f"[SUCCESS] Wrote metadata → {output_dir/'metadata.json'}")


if __name__ == "__main__":
    main()