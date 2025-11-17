#!/usr/bin/env python3
"""
Assign EMA registries to clusters based on inference results.

For each EMA registry:
1. Find the most similar registry based on model prediction
2. If similarity > threshold, assign EMA registry to that registry's cluster
3. If no match is found, assign 'None' as cluster
4. Compute statistics on matched EMA registries

Input requires:
  - Pipeline with model for inference
  - Feature batches from R27_featurize_ema_pairs
  - Candidate pairs between EMA and registry dataset
  - Existing cluster assignments for registry dataset

Output:
  - Cluster assignments for EMA registries
  - Statistics on matching results
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
from collections import defaultdict, Counter

import click
import joblib
import numpy as np
import pandas as pd
from scipy import sparse as sp
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--pipeline", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Path to fitted sklearn Pipeline (joblib). Must contain step named 'model'.")
@click.option("--features_dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Directory containing R27 outputs (feature batches).")
@click.option("--summary_json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Path to R27 summary.json (contains 'features_batches').")
@click.option("--row_index_csv", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Path to R27 row_index.csv.gz (aligned with batches).")
@click.option("--ema_pairs_parquet", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Path to EMA candidate pairs parquet file.")
@click.option("--clusters_json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Path to existing cluster assignments for registry dataset.")
@click.option("--output_dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Directory to write outputs.")
@click.option("--threshold", type=float, default=0.87, show_default=True,
              help="Threshold for cluster assignment. EMA registries with max similarity below this are unassigned.")
@click.option("--seed", type=int, default=42, show_default=True,
              help="Random seed for reproducibility.")
def main(
    pipeline: Path,
    features_dir: Path,
    summary_json: Path,
    row_index_csv: Path,
    ema_pairs_parquet: Path,
    clusters_json: Path,
    output_dir: Path,
    threshold: float,
    seed: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline and extract model
    click.echo(f"[INFO] Loading pipeline: {pipeline}")
    pipe = joblib.load(pipeline)

    if "model" not in pipe.named_steps:
        raise click.ClickException(f"Pipeline has no step named 'model'. Steps: {list(pipe.named_steps.keys())}")

    est = pipe.named_steps["model"]
    click.echo(f"[INFO] Using model step: {type(est).__name__}")

    # Load batch information
    click.echo(f"[INFO] Reading summary: {summary_json}")
    with open(summary_json, "r") as f:
        summary = json.load(f)
    
    # Get feature batch paths
    outputs = summary.get("outputs", {})
    features_batches: List[str] = outputs.get("features_batches", []) or []
    if not features_batches:
        features_batches = sorted(str(p) for p in features_dir.glob("features_batch_*.npz"))
    if not features_batches:
        raise click.ClickException("No feature batches found.")

    # Load row index (to map feature rows to object IDs)
    click.echo(f"[INFO] Reading row index CSV: {row_index_csv}")
    idx_df_full = pd.read_csv(row_index_csv)
    
    # Load EMA candidate pairs
    click.echo(f"[INFO] Reading EMA candidate pairs: {ema_pairs_parquet}")
    ema_pairs_df = pd.read_parquet(ema_pairs_parquet)
    
    # Load existing cluster assignments
    click.echo(f"[INFO] Reading existing cluster assignments: {clusters_json}")
    with open(clusters_json, "r") as f:
        clusters_data = json.load(f)
    
    # Create mapping from object_id to cluster_id
    object_to_cluster = {}
    cluster_to_members = defaultdict(list)

    # clusters_data is a list of dicts with object_id and cluster_id
    for obj in clusters_data:
        object_id = obj["object_id"]
        cluster_id = obj["cluster_id"]
        object_to_cluster[object_id] = cluster_id
        if cluster_id is not None:
            cluster_to_members[cluster_id].append(obj)
    
    click.echo(f"[INFO] Loaded {len(object_to_cluster)} objects with cluster assignments")
    
    # Run inference on feature batches
    click.echo(f"[INFO] Running inference on {len(features_batches)} feature batches")
    
    # Dictionary to store predictions: pair_hash_id -> probability
    predictions = {}
    
    # Process feature batches
    start_idx = 0
    for b_idx, batch_path in enumerate(tqdm(sorted(features_batches), desc="Processing batches", unit="batch")):
        batch_path = Path(batch_path)
        Xb = sp.load_npz(batch_path)
        n_rows_b = Xb.shape[0]
        
        # Get corresponding rows from index
        end_idx = start_idx + n_rows_b
        if end_idx > len(idx_df_full):
            raise click.ClickException("Row index exhausted before feature batches; alignment mismatch.")
        
        idx_df = idx_df_full.iloc[start_idx:end_idx].copy()
        start_idx = end_idx
        
        # Predict probabilities
        proba = est.predict_proba(Xb)[:, 1]
        
        # Store predictions by pair_hash_id
        for i, (_, row) in enumerate(idx_df.iterrows()):
            predictions[row["pair_hash_id"]] = proba[i]
    
    # Create mapping from EMA object_id to candidate matches & metadata
    ema_to_candidates = defaultdict(list)
    ema_metadata = {}  # ema_id -> {name, acronym}
    
    # Process EMA pairs with predictions
    click.echo("[INFO] Organizing EMA registry matches")
    for _, row in tqdm(ema_pairs_df.iterrows(), desc="Processing EMA pairs", total=len(ema_pairs_df)):
        pair_hash_id = row["pair_hash_id"]
        ema_id = row["left_object_id"]
        # Capture left side metadata once
        if ema_id not in ema_metadata:
            ema_metadata[ema_id] = {
                "registry_name": row.get("left_name"),
                "acronym": row.get("left_acronym")
            }
        if pair_hash_id in predictions:
            registry_id = row["right_object_id"]
            proba = predictions[pair_hash_id]
            ema_to_candidates[ema_id].append({
                "registry_id": registry_id,
                "registry_name": row.get("right_name"),
                "acronym": row.get("right_acronym"),
                "cluster_id": object_to_cluster.get(registry_id),
                "similarity": proba,
            })
    
    # Find best (top-3) matches for each EMA registry and assign cluster if top1 ≥ threshold
    click.echo("[INFO] Assigning clusters to EMA registries with top-3 match extraction")
    ema_clusters_records: List[Dict[str, Any]] = []
    closest_matches_records: List[Dict[str, Any]] = []
    
    all_ema_ids = set(ema_metadata.keys())  # ensure we include EMA with no candidate pairs
    
    matched_count = 0
    clusters_with_ema = set()
    ema_per_cluster = Counter()
    aliases_per_ema = {}
    
    for ema_id in tqdm(sorted(all_ema_ids), desc="Assigning clusters"):
        candidates = ema_to_candidates.get(ema_id, [])
        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        # Take top 3 irrespective of threshold for closest matches output
        top3 = candidates[:3]
        top1 = candidates[0] if candidates else None
        # Cluster assignment only if top1 exists AND meets threshold
        if top1 and top1["similarity"] >= threshold and top1["cluster_id"] is not None:
            cluster_id = top1["cluster_id"]
            matched_count += 1
            clusters_with_ema.add(cluster_id)
            ema_per_cluster[cluster_id] += 1
            aliases_per_ema[ema_id] = len(cluster_to_members.get(cluster_id, []))
        else:
            cluster_id = None
        meta = ema_metadata.get(ema_id, {"registry_name": None, "acronym": None})
        cluster_record = {
            "object_id": ema_id,
            "registry_name": meta["registry_name"],
            "acronym": meta["acronym"],
            "cluster_id": cluster_id,
        }
        if cluster_id is not None:
            cluster_record["cluster_size"] = len(cluster_to_members.get(cluster_id, []))
        ema_clusters_records.append(cluster_record)
        # Build ema_closest_matches.json record (top-3 irrespective of threshold; fill None if missing)
        match_record = {
            "object_id": ema_id,
            "registry_name": meta["registry_name"],
            "acronym": meta["acronym"]
        }
        for rank in range(1, 4):
            if rank <= len(top3):
                cand = top3[rank - 1]
                match_record[f"top{rank}_object_id"] = cand["registry_id"]
                match_record[f"top{rank}_registry_name"] = cand["registry_name"]
                match_record[f"top{rank}_similarity"] = cand["similarity"]
                match_record[f"top{rank}_cluster_id"] = cand["cluster_id"]
            else:
                match_record[f"top{rank}_object_id"] = None
                match_record[f"top{rank}_registry_name"] = None
                match_record[f"top{rank}_similarity"] = None
                match_record[f"top{rank}_cluster_id"] = None
        closest_matches_records.append(match_record)
    
    # Compute statistics
    total_ema = len(all_ema_ids)
    match_rate = matched_count / total_ema if total_ema > 0 else 0
    clusters_with_multiple_ema = sum(1 for count in ema_per_cluster.values() if count >= 2)
    avg_aliases = (sum(aliases_per_ema.values()) / len(aliases_per_ema) if aliases_per_ema else 0)
    statistics = {
        "total_ema_registries": total_ema,
        "matched_ema_registries": matched_count,
        "match_rate": match_rate,
        "clusters_with_ema": len(clusters_with_ema),
        "clusters_with_multiple_ema": clusters_with_multiple_ema,
        "avg_aliases_per_matched_ema": avg_aliases,
        "threshold": threshold
    }
    
    # Save outputs (new formats)
    click.echo("[INFO] Saving outputs (ema_clusters.json & ema_closest_matches.json)")
    with open(output_dir / "ema_clusters.json", "w") as f:
        json.dump(ema_clusters_records, f, indent=2)
    with open(output_dir / "ema_closest_matches.json", "w") as f:
        json.dump(closest_matches_records, f, indent=2)
    with open(output_dir / "statistics.json", "w") as f:
        json.dump(statistics, f, indent=2)
    
    # Report (unchanged except references implied)
    matched_count_local = statistics["matched_ema_registries"]
    report_text = f"""# EMA Registry Matching Report

## Summary Statistics

- **Total EMA Registries**: {total_ema}
- **Matched EMA Registries**: {matched_count_local} ({match_rate:.2%})
- **Clusters with EMA Registries**: {len(clusters_with_ema)}
- **Clusters with Multiple EMA Registries**: {clusters_with_multiple_ema}
- **Average Aliases per Matched EMA**: {avg_aliases:.2f}
- **Similarity Threshold**: {threshold}

## Distribution of EMA Registries per Cluster

| EMA Registries | Number of Clusters |
|----------------|-------------------|
"""
    cluster_counts = Counter(ema_per_cluster.values())
    for count in sorted(cluster_counts.keys()):
        report_text += f"| {count} | {cluster_counts[count]} |\n"
    unmatched_count = total_ema - matched_count_local
    report_text += f"\n## Unmatched EMA Registries\n\n{unmatched_count} ({(unmatched_count/total_ema):.2%}) EMA registries did not match any existing registry above the threshold.\n"
    with open(output_dir / "report.md", "w") as f:
        f.write(report_text)
    
    # Histogram of top1 similarities (0 if none)
    plt.figure(figsize=(10, 6))
    all_similarities = []
    for rec in closest_matches_records:
        sim = rec.get("top1_similarity")
        if sim is None:
            sim = 0
        all_similarities.append(sim)
    plt.hist(all_similarities, bins=50)
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    plt.xlabel("Similarity Score (Top1 accepted)")
    plt.ylabel("Count")
    plt.title("Distribution of Top-1 Accepted Match Similarities for EMA Registries")
    plt.legend()
    plt.savefig(output_dir / "similarity_distribution.png")
    
    click.echo(f"[SUCCESS] Results saved to {output_dir}")
    click.echo(f"[SUCCESS] Match rate: {match_rate:.2%}")
    click.echo(f"[SUCCESS] Clusters with multiple EMA registries: {clusters_with_multiple_ema}")
    click.echo(f"[SUCCESS] Average aliases per matched EMA: {avg_aliases:.2f}")


if __name__ == "__main__":
    main()