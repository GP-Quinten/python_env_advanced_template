#!/usr/bin/env python3
"""
Build candidate pairs between EMA registries and existing registry dataset using MinHash+LSH.

Input requires:
  - EMA registry dataset (from R25) with object_id, registry_name, acronym
  - Original registry dataset with the same fields
  - LSH parameters from best_params.json

Output:
  - ema_candidate_pairs.parquet with columns:
    - pair_hash_id
    - left_name (EMA registry)
    - left_acronym (EMA registry)
    - right_name (existing registry)
    - right_acronym (existing registry)
    - left_object_id (EMA registry)
    - right_object_id (existing registry)
  - List of EMA registries that don't land in any bucket (ema_no_bucket.json)
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Set
from collections import defaultdict

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
@click.option("--ema_json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Path to cleaned EMA registry dataset from R25.")
@click.option("--registry_json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Path to original registry dataset from R01.")
@click.option("--best_params_json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Path to best_params.json to load LSH parameters.")
@click.option("--output_dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Directory to write outputs.")
def main(
    ema_json: Path,
    registry_json: Path,
    best_params_json: Path,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load both datasets
    click.echo(f"[INFO] Reading EMA registry dataset: {ema_json}")
    ema_df = pd.read_json(ema_json)
    click.echo(f"[INFO] Loaded {len(ema_df)} EMA registry rows")

    click.echo(f"[INFO] Reading original registry dataset: {registry_json}")
    registry_df = pd.read_json(registry_json)
    click.echo(f"[INFO] Loaded {len(registry_df)} original registry rows")

    # Check required columns
    required = ["object_id", "registry_name", "acronym"]
    for df_name, df in [("EMA", ema_df), ("Registry", registry_df)]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise click.ClickException(f"Missing required columns in {df_name} dataset: {missing}")

    # Map object_id → info
    registry_info: Dict[Any, Dict[str, Any]] = {
        row["object_id"]: {
            "name": row["registry_name"],
            "acronym": row["acronym"],
        }
        for _, row in registry_df.iterrows()
    }

    ema_info: Dict[Any, Dict[str, Any]] = {
        row["object_id"]: {
            "name": row["registry_name"],
            "acronym": row["acronym"],
        }
        for _, row in ema_df.iterrows()
    }

    # Load LSH parameters
    params_raw = json.loads(Path(best_params_json).read_text(encoding="utf-8"))
    # Only keep valid LSHBucketer args
    lsh_keys = {"ngram", "num_perm", "bands", "seed", "min_block_size", "max_block_size"}
    params = {k: params_raw[k] for k in lsh_keys if k in params_raw}

    # Track EMA registries that don't land in any bucket
    ema_no_bucket = set(ema_df["object_id"])
    
    # Store all generated pairs
    pairs_out = []

    # Function to run LSH and create pairs for one approach
    def run_pass(registry_texts: List[str], ema_texts: List[str], desc: str):
        click.echo(f"[INFO] Pass: {desc}, indexing {len(registry_texts) + len(ema_texts)} docs")
        
        # Create bucketer
        bucketer = LSHBucketer(**params)
        
        # Combine registry and EMA datasets with source flags
        combined_docs = []
        # Add registry docs with source flag
        for row, txt in zip(registry_df.itertuples(index=False), registry_texts):
            combined_docs.append((('registry', row.object_id), txt))
        
        # Add EMA docs with source flag
        for row, txt in zip(ema_df.itertuples(index=False), ema_texts):
            combined_docs.append((('ema', row.object_id), txt))
        
        # Build bucketer with combined docs
        bucketer.build(tqdm(combined_docs, desc=f"Index {desc}", unit="doc"))
        
        # Track EMA registries that appear in at least one pair
        found_ema_ids = set()
        
        # Get only EMA-registry pairs (not EMA-EMA or registry-registry)
        for doc1_info, doc2_info in bucketer.pairs():
            source1, id1 = doc1_info
            source2, id2 = doc2_info
            
            # Only keep pairs where one is EMA and one is registry
            if source1 != source2:
                if source1 == 'ema':
                    ema_id, registry_id = id1, id2
                else:
                    ema_id, registry_id = id2, id1
                    
                # Add to found set
                found_ema_ids.add(ema_id)
                    
                # Create pair record
                pair_id = _pair_hash_id(ema_id, registry_id, seed=params["seed"])
                pair = {
                    "pair_hash_id": pair_id,
                    "left_name": ema_info[ema_id]["name"],
                    "left_acronym": ema_info[ema_id]["acronym"],
                    "right_name": registry_info[registry_id]["name"],
                    "right_acronym": registry_info[registry_id]["acronym"],
                    "left_object_id": ema_id,
                    "right_object_id": registry_id,
                }
                pairs_out.append(pair)
        
        # Remove found EMA IDs from the no_bucket set
        for ema_id in found_ema_ids:
            if ema_id in ema_no_bucket:
                ema_no_bucket.remove(ema_id)

    # Multi-pass LSH: raw, normalized, acronym, raw+acro, norm+acro
    # Extract registry texts
    raw_registry = registry_df["registry_name"].fillna("").astype(str).tolist()
    norm_registry = normalize_registry_names(raw_registry)
    acr_registry = registry_df["acronym"].fillna("").astype(str).tolist()
    raw_acro_registry = [f"{n} {a}" for n, a in zip(raw_registry, acr_registry)]
    norm_acro_registry = normalize_registry_names(raw_acro_registry)
    
    # Extract EMA texts
    raw_ema = ema_df["registry_name"].fillna("").astype(str).tolist()
    norm_ema = normalize_registry_names(raw_ema)
    acr_ema = ema_df["acronym"].fillna("").astype(str).tolist()
    raw_acro_ema = [f"{n} {a}" for n, a in zip(raw_ema, acr_ema)]
    norm_acro_ema = normalize_registry_names(raw_acro_ema)
    
    # Run all passes
    run_pass(raw_registry, raw_ema, "raw")
    run_pass(norm_registry, norm_ema, "normalized")
    run_pass(acr_registry, acr_ema, "acronym")
    run_pass(raw_acro_registry, raw_acro_ema, "raw+acro")
    run_pass(norm_acro_registry, norm_acro_ema, "norm+acro")

    # Deduplicate pairs (same pair may be found in multiple passes)
    unique_pairs = {}
    for pair in pairs_out:
        unique_pairs[pair["pair_hash_id"]] = pair
    
    pairs_out = list(unique_pairs.values())

    # Save a small sample (first 100 pairs) as readable JSON
    sample_path = output_dir / "ema_candidate_pairs_sample.json"
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(pairs_out[:100], f, indent=2, ensure_ascii=False)

    # Save full output as optimized Parquet
    parquet_path = output_dir / "ema_candidate_pairs.parquet"
    pd.DataFrame(pairs_out).to_parquet(parquet_path, index=False)

    # Save list of EMA registries with no bucket
    no_bucket_path = output_dir / "ema_no_bucket.json"
    with open(no_bucket_path, "w", encoding="utf-8") as f:
        no_bucket_data = [
            {
                "object_id": ema_id,
                "registry_name": ema_info[ema_id]["name"],
                "acronym": ema_info[ema_id]["acronym"]
            }
            for ema_id in ema_no_bucket
        ]
        json.dump(no_bucket_data, f, indent=2, ensure_ascii=False)

    # Output warnings for EMA registries that didn't land in any bucket
    if ema_no_bucket:
        click.echo(f"[WARNING] {len(ema_no_bucket)} EMA registries didn't land in any bucket:")
        for i, ema_id in enumerate(list(ema_no_bucket)[:10]):  # Show first 10
            click.echo(f"  - {ema_info[ema_id]['name']} (ID: {ema_id})")
        if len(ema_no_bucket) > 10:
            click.echo(f"  - ... and {len(ema_no_bucket) - 10} more (see {no_bucket_path})")

    # Write metadata
    metadata = {
        "n_ema_registries": len(ema_df),
        "n_original_registries": len(registry_df),
        "n_pairs": len(pairs_out),
        "n_ema_no_bucket": len(ema_no_bucket),
        "params": params
    }
    
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    click.echo(f"[SUCCESS] Generated {len(pairs_out):,} candidate pairs to be featurized")
    click.echo(f"[SUCCESS] Wrote {len(pairs_out):,} pairs to {parquet_path}")
    click.echo(f"[SUCCESS] Wrote sample (100) pairs → {sample_path}")
    click.echo(f"[SUCCESS] Wrote {len(ema_no_bucket)} EMA registries with no bucket → {no_bucket_path}")
    click.echo(f"[SUCCESS] Wrote metadata → {output_dir/'metadata.json'}")


if __name__ == "__main__":
    main()