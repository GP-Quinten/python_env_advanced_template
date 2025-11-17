#!/usr/bin/env python3
"""
Cluster unlabeled registries using predictions from R22 (Union-Find).

Inputs
------
--manifest_json : R22 manifest.json (lists per-batch predictions and pred column)
--pairs_parquet : R20 candidate_pairs.parquet (pair_hash_id, left_object_id, right_object_id)
--registry_json : R01 registry_dataset.json (provides the full set of object_id to output)
--threshold     : accept an edge if prediction >= threshold (default 0.50)
--cluster_prefix: prefix for cluster IDs (default 'C')
--pred_column   : name of prediction column in per-batch CSVs (default 'proba')

Outputs (into --output_dir)
---------------------------
- object_clusters.json : JSON array of {"object_id": <id>, "cluster_id": <str|null>}
- summary.json         : counts & diagnostics
"""

# --- Section: Imports & Dependencies -----------------------------------------
from __future__ import annotations
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json, hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set
from itertools import combinations

# --- Section: Utility Functions --------------------------------------------
def _make_pair_hash(left_id: int, right_id: int) -> str:
    """
    Create a reproducible hash for a pair of object IDs.
    Always orders (min, max) to avoid duplicates.
    """
    lid, rid = sorted((left_id, right_id))
    pair_str = f"{lid}::{rid}"
    return hashlib.md5(pair_str.encode("utf-8")).hexdigest()



# --- Section: Candidate Pair Generation ------------------------------------
def generate_candidate_pairs(data_out: list, output_dir: Path):
    """
    Generate all unique intra-cluster pairs and save full + sample files.
    """
    candidate_pairs = []
    clusters_df = pd.DataFrame(data_out)
    valid_clusters = clusters_df[clusters_df.cluster_id.notnull()]
    for cid, group in valid_clusters.groupby("cluster_id"):
        objs = group[["object_id","registry_name","acronym"]].to_dict("records")
        for left, right in combinations(objs, 2):
            lid, rid = int(left["object_id"]), int(right["object_id"])
            ph = _make_pair_hash(lid, rid)
            candidate_pairs.append({
                "pair_hash_id": ph,
                "left_name":   left.get("registry_name"),
                "left_acronym":left.get("acronym"),
                "right_name":  right.get("registry_name"),
                "right_acronym":right.get("acronym"),
                "left_object_id": lid,
                "right_object_id": rid,
                "cluster_id":   cid
            })
    pairs_df = pd.DataFrame(candidate_pairs)
    parquet_path = output_dir / "candidate_pairs.parquet"
    pairs_df.to_parquet(parquet_path, index=False)
    sample_json = output_dir / "candidate_pairs_sample.json"
    sample = pairs_df.sample(min(100, len(pairs_df)), random_state=42).to_dict("records")
    with open(sample_json, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    click.echo(f"[SUCCESS] Candidate pairs (full) → {parquet_path}")
    click.echo(f"[SUCCESS] Candidate pairs sample → {sample_json}")
    return len(candidate_pairs)


# --- Section: Threshold Analysis Plotting ----------------------------------
def threshold_analysis_plot(
    batch_files: list[Path],
    pairs_df: pd.DataFrame,
    pred_column: str,
    threshold: float,
    id_to_idx: Dict[int, int],
    all_ids: List[int],
    output_dir: Path,
):
    """
    Generate diagnostic plots showing how cluster counts vary by threshold.
    """
    thresholds = np.arange(1.0, 0.69, -0.05)
    objs_agg = []
    nclusters = []
    for thr in tqdm(thresholds, desc="Thresholds", unit="thr"):
        accepted = []
        touched = set()
        for bpath in tqdm(batch_files, desc=f"Batch files (thr={thr:.2f})", leave=False):
            df = pd.read_csv(bpath, usecols=["pair_hash_id", pred_column])
            df["pair_hash_id"] = df["pair_hash_id"].astype(str)
            df = df[df[pred_column] >= thr]
            joined = df.join(pairs_df, on="pair_hash_id", how="inner")
            for li, ri in zip(joined["left_object_id"], joined["right_object_id"]):
                if li == ri or li not in id_to_idx or ri not in id_to_idx:
                    continue
                accepted.append((int(li), int(ri)))
                touched.update((li, ri))
        uf_thr = UF(len(all_ids))
        for li, ri in tqdm(accepted, desc="Union merges", unit="edge", leave=False):
            uf_thr.union(id_to_idx[li], id_to_idx[ri])
        cluster_counts_thr = {}
        for oid in touched:
            rep = uf_thr.find(id_to_idx[oid])
            cluster_counts_thr[rep] = cluster_counts_thr.get(rep, 0) + 1
        objs_agg.append(len(touched))
        nclusters.append(len(cluster_counts_thr))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(thresholds, objs_agg, marker="o")
    ax1.set(title="Aggregated Objects by Threshold", xlabel="Threshold", ylabel="Objects")
    ax1.set_xlim(1.0, 0.7)
    ax1.set_xticks(np.arange(1.0, 0.7 - 0.01, -0.05))
    ax2.plot(thresholds, nclusters, marker="o")
    ax2.set(title="Number of Clusters by Threshold", xlabel="Threshold", ylabel="Clusters")
    ax2.set_xlim(1.0, 0.7)
    ax2.set_xticks(np.arange(1.0, 0.7 - 0.01, -0.05))
    plt.tight_layout()
    fig_path = output_dir / "threshold_analysis.png"
    fig.savefig(fig_path)
    click.echo(f"[SUCCESS] Threshold analysis plot → {fig_path}")


# --- Section: Union-Find Data Structure ------------------------------------
@dataclass
class UF:
    """
    Simple Union-Find (Disjoint Set) implementation with path compression.
    """
    p: List[int]
    r: List[int]
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


# --- Section: Command-line Interface & Main Workflow -----------------------
@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--manifest_json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Path to R22 manifest.json (lists per-batch prediction CSVs).")
@click.option("--pairs_parquet", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Path to R20 candidate_pairs.parquet (must include pair_hash_id,left_object_id,right_object_id).")
@click.option("--registry_json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Path to R01 registry_dataset.json (provides full object_id universe).")
@click.option("--output_dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Directory to write outputs.")
@click.option("--threshold", default=0.50, show_default=True, type=float,
              help="Accept an edge if prediction >= threshold.")
@click.option("--cluster_prefix", default="C", show_default=True, type=str,
              help="Prefix for cluster IDs.")
@click.option("--pred_column", default="proba", show_default=True, type=str,
              help="Name of prediction column in batch CSVs.")
@click.option("--n_batch", default=-1, show_default=True, type=int,
              help="Number of batch files to process (-1 for all).")
def main(manifest_json: Path, pairs_parquet: Path, registry_json: Path, output_dir: Path,
         threshold: float, cluster_prefix: str, pred_column: str, n_batch: int):

    # 1) Prepare output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"[INFO] Using probability threshold: {threshold}")

    # 2) Load the full registry to know all object IDs
    click.echo("[INFO] Loading registry objects...")
    # --- Load registry to get the full universe of object_ids we must output
    click.echo(f"[INFO] Loading registry objects: {registry_json}")
    reg_df = pd.read_json(registry_json)
    if "object_id" not in reg_df.columns:
        raise click.ClickException("registry_json must contain 'object_id'.")
    all_ids: List[int] = list(pd.Series(reg_df["object_id"]).astype(int).unique())
    id_to_idx: Dict[int, int] = {oid: i for i, oid in enumerate(all_ids)}
    click.echo(f"[INFO] Found {len(all_ids):,} unique object_ids.")

    # 3) Load candidate pairs mapping
    click.echo("[INFO] Loading pairs mapping...")
    # --- Load mapping pair_hash_id -> (left_object_id, right_object_id)
    click.echo(f"[INFO] Loading pairs mapping: {pairs_parquet}")
    pairs_df = pd.read_parquet(pairs_parquet, columns=["pair_hash_id", "left_object_id", "right_object_id"])
    input_pairs_count = len(pairs_df)                  # new: total input pairs
    if pairs_df.empty:
        raise click.ClickException("pairs_parquet is empty.")
    # keep minimal footprint & types
    pairs_df["left_object_id"] = pairs_df["left_object_id"].astype(int)
    pairs_df["right_object_id"] = pairs_df["right_object_id"].astype(int)
    pairs_df["pair_hash_id"] = pairs_df["pair_hash_id"].astype(str)
    pairs_df.set_index("pair_hash_id", inplace=True)
    click.echo(f"[INFO] Pairs mapping rows: {len(pairs_df):,}")

    # 4) Read manifest of batch prediction CSVs
    click.echo("[INFO] Reading manifest for batch files...")
    # --- Read manifest to get list of batch prediction CSVs
    click.echo(f"[INFO] Reading manifest: {manifest_json}")
    with open(manifest_json, "r") as f:
        manifest = json.load(f)
    batch_entries = manifest.get("batch_outputs") or []
    if not batch_entries:
        raise click.ClickException("manifest.json has no 'batch_outputs'. Did R22 run?")
    batch_files: List[Path] = [Path(d["predictions_csv"]) for d in batch_entries]
    if n_batch != -1:
        batch_files = batch_files[:n_batch]
        click.echo(f"[INFO] Only processing first {n_batch} batch files.")

    # 5) Scan predictions, filter by threshold, accumulate edges
    click.echo("[INFO] Scanning predictions and collecting edges...")
    # --- Accumulate accepted edges (object_id pairs)
    accepted_edges: List[Tuple[int, int]] = []
    touched_ids: Set[int] = set()

    click.echo(f"[INFO] Scanning predictions in {len(batch_files)} batch files...")
    for bpath in tqdm(sorted(batch_files), desc="Batches", unit="batch"):
        # minimal columns to reduce IO
        usecols = ["pair_hash_id", pred_column]
        try:
            pred_df = pd.read_csv(bpath, usecols=usecols)
        except ValueError:
            # some CSVs may include only pair_hash_id, left/right — fallback to reading all and selecting
            pred_df = pd.read_csv(bpath)
            if pred_column not in pred_df.columns:
                raise click.ClickException(f"Prediction column '{pred_column}' not found in {bpath}.")
            pred_df = pred_df[["pair_hash_id", pred_column]]

        pred_df["pair_hash_id"] = pred_df["pair_hash_id"].astype(str)
        pred_df = pred_df[pred_df[pred_column] >= threshold]
        if pred_df.empty:
            continue

        # Join to get object_ids for edges
        joined = pred_df.join(pairs_df, on="pair_hash_id", how="inner")
        if joined.empty:
            continue

        # Accumulate edges
        for li, ri in zip(joined["left_object_id"].values, joined["right_object_id"].values):
            li = int(li); ri = int(ri)
            if li == ri:
                continue
            if (li not in id_to_idx) or (ri not in id_to_idx):
                # object_id not in registry universe -> skip
                continue
            accepted_edges.append((li, ri))
            touched_ids.add(li); touched_ids.add(ri)

    click.echo(f"[INFO] Accepted edges: {len(accepted_edges):,}  |  Touched IDs: {len(touched_ids):,}")

    # 6) Perform union-find merges on accepted edges
    click.echo("[INFO] Performing Union-Find merges...")
    # --- Union-Find over the full object_id universe
    uf = UF(len(all_ids))
    # Merge only accepted edges
    for li, ri in tqdm(accepted_edges, desc="Union merges", unit="edge"):
        uf.union(id_to_idx[li], id_to_idx[ri])

    # 7) Assign cluster IDs deterministically (including singletons)
    click.echo("[INFO] Building cluster assignments...")
    # --- Build cluster IDs for all objects (including untouched ones)
    rep_to_cluster: Dict[int, str] = {}
    obj_to_cluster: Dict[int, str] = {}  # Changed type annotation - no more None

    for oid in all_ids:
        idx = id_to_idx[oid]
        rep = uf.find(idx)  # Get representative even for untouched objects
        if rep not in rep_to_cluster:
            # Deterministic ID based on representative's index
            rep_to_cluster[rep] = f"{cluster_prefix}{rep:06d}"
        obj_to_cluster[oid] = rep_to_cluster[rep]

    # compute cluster sizes (modified to handle all clusters)
    cluster_counts: Dict[str, int] = {}
    for cid in obj_to_cluster.values():
        cluster_counts[cid] = cluster_counts.get(cid, 0) + 1

    # 8) Write object_clusters.json and cluster_pairs.json
    click.echo("[INFO] Writing cluster outputs...")
    # --- Write outputs with name, acronym, and cluster_size
    clusters_out = output_dir / "object_clusters.json"
    # index registry DataFrame by object_id for lookup
    reg_indexed = reg_df.set_index("object_id")
    data_out: List[dict] = []
    for oid in all_ids:
        cid = obj_to_cluster[oid]
        entry = {
            "object_id": int(oid),
            "cluster_id": cid,  # No more None values
            "cluster_size": cluster_counts[cid],  # Always has a size (1 for singletons)
            "registry_name": reg_indexed.at[int(oid), "registry_name"]
                if "registry_name" in reg_indexed.columns else None,
            "acronym": reg_indexed.at[int(oid), "acronym"]
                if "acronym" in reg_indexed.columns else None
        }
        data_out.append(entry)

    with open(clusters_out, "w", encoding="utf-8") as f:
        json.dump(data_out, f, indent=2, ensure_ascii=False)

    # --- Build per-cluster JSON of accepted pairs
    cluster_pairs = {}
    # Build lookup for object info
    obj_info = {
        int(row["object_id"]): {
            "registry_name": row.get("registry_name"),
            "acronym": row.get("acronym"),
            "object_id": int(row["object_id"]),
            "cluster_id": row.get("cluster_id")
        }
        for row in data_out
    }
    # For each batch, collect pairs with proba and object info
    for bpath in tqdm(sorted(batch_files), desc="Collecting pairs for clusters", unit="batch"):
        try:
            pred_df = pd.read_csv(bpath, usecols=["pair_hash_id", pred_column])
        except ValueError:
            pred_df = pd.read_csv(bpath)
            if pred_column not in pred_df.columns:
                continue
            pred_df = pred_df[["pair_hash_id", pred_column]]
        pred_df["pair_hash_id"] = pred_df["pair_hash_id"].astype(str)
        pred_df = pred_df[pred_df[pred_column] >= threshold]
        if pred_df.empty:
            continue
        joined = pred_df.join(pairs_df, on="pair_hash_id", how="inner")
        for _, row in joined.iterrows():
            li = int(row["left_object_id"])
            ri = int(row["right_object_id"])
            if li == ri or (li not in obj_to_cluster) or (ri not in obj_to_cluster):
                continue
            cid = obj_to_cluster[li]
            if cid is None:
                continue
            pair_entry = {
                "proba": float(row[pred_column]),
                "left_object_id": li,
                "right_object_id": ri,
                "left_registry_name": obj_info[li]["registry_name"],
                "right_registry_name": obj_info[ri]["registry_name"],
                "left_acronym": obj_info[li]["acronym"],
                "right_acronym": obj_info[ri]["acronym"],
                "left_cluster_id": obj_info[li]["cluster_id"],
                "right_cluster_id": obj_info[ri]["cluster_id"]
            }
            cluster_pairs.setdefault(cid, []).append(pair_entry)

    cluster_pairs_path = output_dir / "cluster_pairs.json"
    with open(cluster_pairs_path, "w", encoding="utf-8") as f:
        json.dump(cluster_pairs, f, indent=2, ensure_ascii=False)
    click.echo(f"[SUCCESS] Cluster pairs JSON → {cluster_pairs_path}")

    # 9) Generate additional candidate pairs for unlabeled items
    click.echo("[INFO] Generating candidate intra-cluster pairs...")
    # --- Generate all intra-cluster candidate pairs
    n_candidate_pairs = generate_candidate_pairs(data_out, output_dir)

    # 10) Compile and write summary.json and human-readable reports
    click.echo("[INFO] Writing summary & reports...")
    # --- Prepare and write extended summary (includes input, cluster and candidate counts)

    clusters_df = pd.DataFrame(data_out)
    valid = clusters_df[clusters_df.cluster_id.notnull()]
    grp = valid.groupby("cluster_id")
    sizes = grp.size().sort_values(ascending=False)
    topN = sizes.head(20)
    bottomN = sizes[sizes > 0].sort_values().head(20)

    # --- Compute requested metrics
    n_total_registries = int(len(clusters_df))
    n_singleton_clusters = int(sum(1 for size in cluster_counts.values() if size == 1))
    n_clusters = int(len(cluster_counts))
    avg_registries_per_cluster = float(sum(cluster_counts.values()) / n_clusters)
    n_registries_in_largest_cluster = int(max(cluster_counts.values()))
    n_clusters_size_less_4 = int(sum(1 for size in cluster_counts.values() if size < 4))
    n_cluster_pairs = sum(len(pairs) for pairs in cluster_pairs.values())

    summary_extended = {
        "threshold": threshold,
        "cluster_prefix": cluster_prefix,
        "n_objects": len(all_ids),
        "n_edges": len(accepted_edges),
        "n_touched": len(touched_ids),
        "n_clusters": len({v for v in obj_to_cluster.values() if v is not None}),
        "n_input_pairs": input_pairs_count,
        "n_cluster_pairs": n_cluster_pairs,
        "n_candidate_pairs": n_candidate_pairs,
        "n_total_pairs": input_pairs_count + n_candidate_pairs,
        # --- new metrics ---
        "n_total_registries": n_total_registries,
        "n_singleton_clusters": n_singleton_clusters,  # New metric
        "n_clusters": n_clusters,
        "avg_registries_per_cluster": avg_registries_per_cluster,
        "n_registries_in_largest_cluster": n_registries_in_largest_cluster,
        "n_clusters_size_less_4": n_clusters_size_less_4,
        # --- outputs/inputs ---
        "outputs": {
            "object_clusters": str(output_dir / "object_clusters.json"),
            "cluster_pairs": str(cluster_pairs_path),
            "candidate_pairs": str(output_dir / "candidate_pairs.parquet")
        },
        "inputs": {
            "manifest_json": str(manifest_json),
            "pairs_parquet": str(pairs_parquet),
            "registry_json": str(registry_json),
            "pred_column": pred_column
        }
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_extended, f, indent=2)
    click.echo(f"[SUCCESS] Extended summary → {output_dir / 'summary.json'}")

    # --- Generate cluster report (top 20 largest with top names, and 20 smallest with names)
    report_path = output_dir / "report.md"
    clusters_df = pd.DataFrame(data_out)
    valid = clusters_df[clusters_df.cluster_id.notnull()]
    grp = valid.groupby("cluster_id")
    sizes = grp.size().sort_values(ascending=False)
    topN = sizes.head(20)
    bottomN = sizes[sizes > 0].sort_values().head(20)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Cluster Report\n\n")
        f.write("## Top 20 Largest Clusters\n\n")
        for cid, sz in topN.items():
            f.write(f"### {cid}  (size={sz})\n")
            group = grp.get_group(cid)
            names = group["registry_name"].dropna().unique()[:10]
            acronyms = group["acronym"].dropna().unique()[:10]
            for i, n in enumerate(names):
                acro = acronyms[i] if i < len(acronyms) else ""
                if acro:
                    f.write(f"- {n} ({acro})\n")
                else:
                    f.write(f"- {n}\n")
            f.write("\n")
        f.write("## 20 Smallest Clusters\n\n")
        for cid, sz in bottomN.items():
            f.write(f"### {cid}  (size={sz})\n")
            group = grp.get_group(cid)
            names = group["registry_name"].dropna().unique()[:10]
            acronyms = group["acronym"].dropna().unique()[:10]
            for i, n in enumerate(names):
                acro = acronyms[i] if i < len(acronyms) else ""
                if acro:
                    f.write(f"- {n} ({acro})\n")
                else:
                    f.write(f"- {n}\n")
            f.write("\n")
        # --- Add singleton clusters section ---
        f.write("## Sample of Singleton Clusters (up to 100)\n\n")
        singleton_clusters = [(cid, size) for cid, size in cluster_counts.items() if size == 1]
        sample_singletons = sorted(singleton_clusters)[:100]  # Take first 100 sorted by cluster ID
        for cid, _ in sample_singletons:
            group = clusters_df[clusters_df.cluster_id == cid]
            for _, row in group.iterrows():
                reg_name = row.get("registry_name") or ""
                acronym = row.get("acronym") or ""
                f.write(f"- Cluster {cid}: {reg_name} ({acronym})\n")
        f.write("\n")

    click.echo(f"[SUCCESS] Report → {report_path}")

    # --- Pairs report for top 20 clusters (after cluster_pairs is ready)
    pairs_report_path = output_dir / "pairs_report.md"
    with open(pairs_report_path, "w", encoding="utf-8") as f:
        f.write("# Pairs Report: Top 20 Largest Clusters\n\n")
        for cid in topN.index:
            f.write(f"## Cluster {cid} (size={topN[cid]})\n")
            pairs = cluster_pairs.get(cid, [])
            if not pairs:
                f.write("No pairs found for this cluster.\n\n")
                continue
            for pair in pairs:
                left_name = pair.get("left_registry_name") or str(pair["left_object_id"])
                right_name = pair.get("right_registry_name") or str(pair["right_object_id"])
                left_acro = pair.get("left_acronym") or ""
                right_acro = pair.get("right_acronym") or ""
                proba = round(pair["proba"], 2)
                f.write(f"- {left_name} ({left_acro}) ↔ {right_name} ({right_acro}) | proba={proba}\n")
            f.write("\n")
    click.echo(f"[SUCCESS] Pairs report → {pairs_report_path}")

    # 11) Export edges/nodes for Gephi visualization
    click.echo("[INFO] Exporting Gephi CSVs...")
    # --- Export for Gephi (edges & nodes)
    edges_csv = output_dir / "edges.csv"
    nodes_csv = output_dir / "nodes.csv"

    # Build edges: use accepted_edges and their probabilities
    edges_out = []
    for bpath in batch_files:
        try:
            pred_df = pd.read_csv(bpath, usecols=["pair_hash_id", pred_column])
        except ValueError:
            pred_df = pd.read_csv(bpath)
            if pred_column not in pred_df.columns:
                continue
            pred_df = pred_df[["pair_hash_id", pred_column]]
        pred_df["pair_hash_id"] = pred_df["pair_hash_id"].astype(str)
        pred_df = pred_df[pred_df[pred_column] >= threshold]
        if pred_df.empty:
            continue
        joined = pred_df.join(pairs_df, on="pair_hash_id", how="inner")
        for _, row in joined.iterrows():
            li, ri = int(row["left_object_id"]), int(row["right_object_id"])
            if li == ri: 
                continue
            if (li not in obj_to_cluster) or (ri not in obj_to_cluster):
                continue
            edges_out.append({
                "Source": li,
                "Target": ri,
                "Weight": float(row[pred_column]),
                "Cluster": obj_to_cluster.get(li) or obj_to_cluster.get(ri)
            })

    # Include generated intra-cluster candidate pairs as edges with weight=1.0
    cand_pairs_path = output_dir / "candidate_pairs.parquet"
    cand_df = pd.read_parquet(cand_pairs_path)
    for _, row in cand_df.iterrows():
        edges_out.append({
            "Source": int(row["left_object_id"]),
            "Target": int(row["right_object_id"]),
            "Weight": 1.0,
            "Cluster": row["cluster_id"]
        })

    # Build nodes
    nodes_out = []
    for entry in data_out:
        nodes_out.append({
            "Id": entry["object_id"],
            "Label": entry.get("registry_name") or str(entry["object_id"]),
            "ClusterId": entry["cluster_id"],
            "ClusterSize": entry.get("cluster_size"),
            "Acronym": entry.get("acronym")
        })

    pd.DataFrame(edges_out).to_csv(edges_csv, index=False)
    pd.DataFrame(nodes_out).to_csv(nodes_csv, index=False)

    click.echo(f"[SUCCESS] Gephi edges CSV → {edges_csv}")
    click.echo(f"[SUCCESS] Gephi nodes CSV → {nodes_csv}")


if __name__ == "__main__":
    main()
