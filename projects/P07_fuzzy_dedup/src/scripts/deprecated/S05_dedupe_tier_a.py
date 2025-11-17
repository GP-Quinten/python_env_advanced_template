#!/usr/bin/env python3
# file: S05_dedupe_tier_a.py

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Set, Tuple

import click
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------
# Helpers
# ---------------------------

DEFAULT_STOPWORDS: Set[str] = {
    "the","a","an","of","for","in","on","and","with","to","from","by"
}

def _read_json_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    try:
        return pd.read_json(path)
    except ValueError:
        return pd.read_json(path, lines=True)

def _tokens(s: str) -> List[str]:
    return s.split() if s else []

@dataclass
class TierAParams:
    jw_threshold: float
    soft_tfidf_threshold: float
    soft_tfidf_min_jaccard: float
    stopwords: Set[str]

# ---------------------------
# Union-Find
# ---------------------------

class UnionFind:
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
        if ra == rb: return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1

# ---------------------------
# Tier-A rules + clustering
# ---------------------------

def apply_tier_a_rules(
    df: pd.DataFrame,
    scored_pairs: pd.DataFrame,
    target_column: str,
    p: TierAParams,
) -> List[Tuple[int, int]]:
    """
    Returns accepted edges (i, j) for Tier-A:
      R1: exact normalized match (within dataset)
      R2: jw >= jw_threshold
      R4: soft_tfidf >= soft_tfidf_threshold AND jaccard_tok >= soft_tfidf_min_jaccard
    """
    names_norm = df[target_column].fillna("").astype(str).tolist()
    accepted: Set[Tuple[int, int]] = set()

    # --- R1: exact normalized match (group inside DF) ---
    groups = defaultdict(list)
    for idx, s in enumerate(names_norm):
        if s:
            groups[s].append(idx)
    for g in groups.values():
        if len(g) >= 2:
            g = sorted(g)
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    accepted.add((g[i], g[j]))

    # --- R2 / R4 via scored_pairs ---
    req_cols = {"idx_i", "idx_j", "jw", "jaccard_tok", "soft_tfidf"}
    if not req_cols.issubset(scored_pairs.columns):
        missing = sorted(req_cols - set(scored_pairs.columns))
        raise click.ClickException(f"Scored pairs missing columns: {missing}")

    it = tqdm(scored_pairs.itertuples(index=False), total=len(scored_pairs), desc="Tier-A evaluate")
    for row in it:
        i = int(row.idx_i); j = int(row.idx_j)
        if i == j: 
            continue
        if (i, j) in accepted or (j, i) in accepted:
            continue

        # R2: Very high JW
        if row.jw is not None and row.jw >= p.jw_threshold:
            accepted.add((min(i, j), max(i, j)))
            continue

        # R4: Soft TF-IDF + Jaccard guard
        stf = getattr(row, "soft_tfidf", None)
        jac = getattr(row, "jaccard_tok", None)
        if (stf is not None) and (jac is not None):
            if (stf >= p.soft_tfidf_threshold) and (jac >= p.soft_tfidf_min_jaccard):
                accepted.add((min(i, j), max(i, j)))
                continue

    return sorted(accepted)

def cluster_with_union_find(n_rows: int, edges: List[Tuple[int, int]], cluster_prefix: str) -> List[str]:
    uf = UnionFind(n_rows)
    for i, j in tqdm(edges, desc="Union merges"):
        uf.union(i, j)

    rep_to_members = defaultdict(list)
    for i in range(n_rows):
        rep_to_members[uf.find(i)].append(i)

    # Deterministic cluster IDs
    rep_to_cluster = {rep: f"{cluster_prefix}{rep:06d}" for rep in sorted(rep_to_members)}
    return [rep_to_cluster[uf.find(i)] for i in range(n_rows)]

# ---------------------------
# CLI
# ---------------------------

@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--dataset_json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Path to dataset JSON/JSONL (must contain target_column).")
@click.option("--scored_pairs_json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Path to scored pairs JSON from step 4 (needs idx_i, idx_j, jw, jaccard_tok, soft_tfidf).")
@click.option("--output_dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Directory to write outputs.")
@click.option("--output_filename", default="tier_a_clusters.json", show_default=True,
              help="Clustered dataset output filename (adds cluster_id).")
@click.option("--edges_filename", default="tier_a_edges.json", show_default=True,
              help="Accepted Tier-A edges (pairs) filename.")
@click.option("--target_column", default="norm__registry_name", show_default=True,
              help="Normalized text column to use for rules (R1).")
@click.option("--jw_threshold", default=0.985, show_default=True,
              help="R2: Minimum JW to accept a pair.")
@click.option("--soft_tfidf_threshold", default=0.92, show_default=True,
              help="R4: Minimum Soft TF-IDF to accept a pair.")
@click.option("--soft_tfidf_min_jaccard", default=0.60, show_default=True,
              help="R4: Minimum token Jaccard to accept a pair.")
@click.option("--extra_stopwords", default="", show_default=True,
              help="Comma-separated extra stopwords to include for reference.")
@click.option("--cluster_prefix", default="C", show_default=True,
              help="Prefix for cluster IDs.")
def main(
    dataset_json: Path,
    scored_pairs_json: Path,
    output_dir: Path,
    output_filename: str,
    edges_filename: str,
    target_column: str,
    jw_threshold: float,
    soft_tfidf_threshold: float,
    soft_tfidf_min_jaccard: float,
    extra_stopwords: str,
    cluster_prefix: str,
):
    """
    Apply Tier-A duplicate rules and cluster duplicates with Union-Find.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading dataset: {dataset_json}")
    df = _read_json_any(dataset_json)
    if target_column not in df.columns:
        raise click.ClickException(f"Column '{target_column}' not found. Available: {list(df.columns)}")
    n_rows = len(df)

    log.info(f"Loading scored pairs: {scored_pairs_json}")
    scored_pairs = _read_json_any(scored_pairs_json)

    stopwords = set(DEFAULT_STOPWORDS)
    if extra_stopwords.strip():
        stopwords |= {w.strip() for w in extra_stopwords.split(",") if w.strip()}

    params = TierAParams(
        jw_threshold=jw_threshold,
        soft_tfidf_threshold=soft_tfidf_threshold,
        soft_tfidf_min_jaccard=soft_tfidf_min_jaccard,
        stopwords=stopwords,
    )

    # Apply Tier-A rules to find potential duplicate pairs
    log.info("Applying Tier-A rules…")
    edges = apply_tier_a_rules(df, scored_pairs, target_column, params)
    log.info(f"Accepted {len(edges)} edges")

    # Use Union-Find algorithm to create connected components (clusters) from the edges
    log.info("Clustering with Union-Find…")
    cluster_ids = cluster_with_union_find(n_rows, edges, cluster_prefix)

    # Create output dataframe with cluster assignments
    out_df = df.copy()
    out_df["cluster_id"] = cluster_ids

    # Save the clustered dataset as JSON
    out_path = output_dir / output_filename
    out_df.to_json(out_path, orient="records", indent=3, force_ascii=False)
    log.info(f"Wrote clustered dataset: {out_path}")

    # Save the identified duplicate edges (pairs) separately
    edges_path = output_dir / edges_filename
    pd.DataFrame(edges, columns=["idx_i", "idx_j"]).to_json(edges_path, orient="records", indent=3, force_ascii=False)
    log.info(f"Wrote accepted edges: {edges_path}")

    # Generate and log summary statistics about the clusters
    sizes = Counter(cluster_ids)
    log.info(f"Unique clusters: {len(sizes)} | Largest cluster size: {max(sizes.values()) if sizes else 0}")

if __name__ == "__main__":
    main()

