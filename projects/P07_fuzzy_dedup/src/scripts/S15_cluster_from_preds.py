#!/usr/bin/env python3
# File: src/scripts/S15_cluster_from_preds.py
#
# Build clusters from model predictions by accepting an edge (i, j)
# when predicted probability >= threshold, then union-find to form components.

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List, Set, Tuple

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------
# IO helpers
# ---------------------------

def _read_json_any(path: Path) -> pd.DataFrame:
    """
    Read JSON or JSONL/NDJSON (auto-detect). Falls back to lines=True on ValueError.
    """
    p = Path(path)
    if p.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.read_json(p, lines=True)
    try:
        return pd.read_json(p)
    except ValueError:
        return pd.read_json(p, lines=True)

def _read_predictions(path: Path) -> pd.DataFrame:
    """
    Read predictions table.
    Expected: CSV by default, but supports JSON/JSONL if needed.
    """
    p = Path(path)
    suf = p.suffix.lower()
    if suf in {".jsonl", ".ndjson"}:
        return pd.read_json(p, lines=True)
    if suf == ".json":
        return pd.read_json(p)
    # default: CSV/TSV
    if suf == ".tsv":
        return pd.read_csv(p, sep="\t")
    return pd.read_csv(p)

# ---------------------------
# Union-Find
# ---------------------------

class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n
    def find(self, x: int) -> int:
        # path compression
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a: int, b: int) -> None:
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

# ---------------------------
# Core logic
# ---------------------------

def _build_edges_from_predictions(
    preds: pd.DataFrame,
    *,
    threshold: float,
    proba_col: str,
    idx_i_col: str,
    idx_j_col: str,
    n_rows_dataset: int,
) -> List[Tuple[int, int]]:
    """
    Accept edge (i, j) if preds[proba_col] >= threshold,
    canonicalize as (min(i,j), max(i,j)), drop self-loops and OOB indices.
    """
    req = {proba_col, idx_i_col, idx_j_col}
    if not req.issubset(preds.columns):
        missing = sorted(req - set(preds.columns))
        raise click.ClickException(f"Predictions missing columns: {missing}")

    # Filter by threshold & non-null probability
    df = preds.loc[preds[proba_col].notna() & (preds[proba_col] >= float(threshold)), [idx_i_col, idx_j_col]]
    if df.empty:
        return []

    # canonicalize, dedupe
    pairs: Set[Tuple[int, int]] = set()
    oob_count = 0
    self_count = 0

    it = tqdm(df.itertuples(index=False), total=len(df), desc="Selecting edges")
    for row in it:
        i = int(getattr(row, idx_i_col))
        j = int(getattr(row, idx_j_col))
        if i == j:
            self_count += 1
            continue
        a, b = (i, j) if i < j else (j, i)
        if a < 0 or b < 0 or a >= n_rows_dataset or b >= n_rows_dataset:
            oob_count += 1
            continue
        pairs.add((a, b))

    if oob_count:
        log.warning(f"Dropped {oob_count} prediction rows with out-of-bounds indices.")
    if self_count:
        log.info(f"Ignored {self_count} self-loop predictions (i == j).")

    return sorted(pairs)

def cluster_with_union_find(n_rows: int, edges: Iterable[Tuple[int, int]], cluster_prefix: str) -> List[str]:
    uf = UnionFind(n_rows)
    for i, j in tqdm(edges, desc="Union merges"):
        uf.union(i, j)

    # Group by representative
    rep_to_members = defaultdict(list)
    for i in range(n_rows):
        rep_to_members[uf.find(i)].append(i)

    # Deterministic cluster IDs: prefix + rep index zero-padded to 6
    rep_to_cluster = {rep: f"{cluster_prefix}{rep:06d}" for rep in sorted(rep_to_members)}
    return [rep_to_cluster[uf.find(i)] for i in range(n_rows)]

def _summarize_clusters(cluster_ids: List[str]) -> dict:
    sizes = Counter(cluster_ids)  # cluster_id -> size
    if not sizes:
        return {
            "n_rows": 0,
            "n_clusters": 0,
            "largest_cluster_size": 0,
            "size_histogram": {},
        }
    size_values = list(sizes.values())
    hist = Counter(size_values)  # size -> how many clusters of that size
    return {
        "n_rows": len(cluster_ids),
        "n_clusters": len(sizes),
        "largest_cluster_size": max(size_values),
        "size_histogram": {int(k): int(v) for k, v in sorted(hist.items())},
    }

# ---------------------------
# CLI
# ---------------------------

@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--dataset_json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Path to dataset JSON/JSONL used to determine row count; cluster_id will be appended.")
@click.option("--predictions_csv", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Predictions table with columns [idx_i, idx_j, proba] (CSV/TSV/JSON/JSONL).")
@click.option("--output_dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Directory to write predicted clusters and edges.")
@click.option("--threshold", type=float, default=0.50, show_default=True,
              help="Accept an edge (i, j) if probability >= threshold.")
@click.option("--proba_col", type=str, default="proba", show_default=True,
              help="Column in predictions with probabilities.")
@click.option("--idx_i_col", type=str, default="idx_i", show_default=True,
              help="Column name for left row index.")
@click.option("--idx_j_col", type=str, default="idx_j", show_default=True,
              help="Column name for right row index.")
@click.option("--cluster_prefix", type=str, default="C", show_default=True,
              help="Prefix for cluster IDs (e.g., 'C').")
def main(
    dataset_json: Path,
    predictions_csv: Path,
    output_dir: Path,
    threshold: float,
    proba_col: str,
    idx_i_col: str,
    idx_j_col: str,
    cluster_prefix: str,
):
    """
    Build clusters from model predictions using a fixed probability threshold.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading dataset: {dataset_json}")
    df = _read_json_any(Path(dataset_json))
    n_rows = len(df)
    log.info(f"Dataset rows: {n_rows}")

    log.info(f"Loading predictions: {predictions_csv}")
    preds = _read_predictions(Path(predictions_csv))
    log.info(f"Predictions rows: {len(preds)}")

    # Build accepted edges
    log.info(f"Selecting edges with threshold >= {threshold}")
    edges = _build_edges_from_predictions(
        preds,
        threshold=threshold,
        proba_col=proba_col,
        idx_i_col=idx_i_col,
        idx_j_col=idx_j_col,
        n_rows_dataset=n_rows,
    )
    log.info(f"Accepted edges: {len(edges)}")

    # Cluster via Union-Find
    log.info("Clustering with Union-Find…")
    cluster_ids = cluster_with_union_find(n_rows, edges, cluster_prefix=cluster_prefix)

    # Write outputs
    clusters_path = output_dir / "predicted_clusters.json"
    edges_path = output_dir / "predicted_edges.json"
    summary_path = output_dir / "summary.json"

    out_df = df.copy()
    out_df["cluster_id"] = cluster_ids
    out_df.to_json(clusters_path, orient="records", indent=3, force_ascii=False)
    log.info(f"Wrote clustered dataset: {clusters_path}")

    pd.DataFrame(edges, columns=[idx_i_col, idx_j_col]).to_json(
        edges_path, orient="records", indent=3, force_ascii=False
    )
    log.info(f"Wrote accepted edges: {edges_path}")

    summary = _summarize_clusters(cluster_ids)
    summary.update({
        "n_edges": int(len(edges)),
        "threshold": float(threshold),
        "proba_col": proba_col,
        "idx_i_col": idx_i_col,
        "idx_j_col": idx_j_col,
    })
    summary_path.write_text(json.dumps(summary, indent=3))
    log.info(f"Wrote summary: {summary_path}")

    sizes = Counter(cluster_ids)
    log.info(f"Clusters: {len(sizes)} | Largest size: {max(sizes.values()) if sizes else 0}")

if __name__ == "__main__":
    main()
