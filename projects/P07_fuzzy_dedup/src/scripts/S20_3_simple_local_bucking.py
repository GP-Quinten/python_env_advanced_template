#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S20_2_measure_semantic_bucketing_local.py
-----------------------------------------
Build a registry dataset from annotated pairs, compute local nearest neighbors
using character trigram Jaccard similarity (no Weaviate), and measure recall
of positive pairs.

Outputs (in output_dir):
 - registry.json: flattened registry table (object_id, registry_name, acronym, combined)
 - neighbors.json: mapping object_id -> list of neighbor object_ids
 - candidate_pairs.parquet (or .json fallback): candidate pair records
 - metrics.json: recall, n_pairs, n_pos, n_hit, reduction_ratio, n_objects, total_possible_pairs
 - metadata.json: basic metadata about run (topk, backend used)

Notes:
 - Local neighbor search is O(n^2) over registry size; for large n, expect longer runtimes.
"""

from pathlib import Path
import json
from typing import Any, Dict, List, Set, Tuple
import click
import pandas as pd
from tqdm import tqdm
import mmh3
import re


# -----------------------------
# Utility helpers
# -----------------------------
def unordered_pair(a: Any, b: Any) -> Tuple[str, str]:
    x, y = str(a), str(b)
    return (x, y) if x <= y else (y, x)


def pair_hash_id(left_id, right_id, *, seed: int = 13) -> str:
    a, b = (str(left_id), str(right_id))
    if a > b:
        a, b = b, a
    h = mmh3.hash128(f"{a}||{b}", seed=seed, signed=False)
    return f"{h:032x}"


def _normalize_text(s: str) -> str:
    """Basic normalization for trigram tokenization."""
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_registry_from_labeled_pairs(labeled_df: pd.DataFrame) -> pd.DataFrame:
    """Flatten labeled pair dataset into a unique registry table with object_id, registry_name, acronym."""
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


# -----------------------------
# Local similarity search
# -----------------------------
def _trigrams(s: str) -> Set[str]:
    s = _normalize_text(s)
    if not s:
        return set()
    return {s[i: i + 3] for i in range(max(0, len(s) - 2))}


def local_topk_neighbors(reg_df: pd.DataFrame, topk: int = 50) -> Dict[str, List[str]]:
    """Compute Jaccard similarity on character trigrams to return neighbors for each registry."""
    click.echo(f"[INFO] Running local neighbor search for {len(reg_df)} registries (topk={topk})")
    texts = reg_df["combined"].tolist()
    ids = reg_df["object_id"].tolist()
    grams = [_trigrams(t) for t in texts]
    neighbors: Dict[str, List[str]] = {}
    n = len(ids)

    for i in tqdm(range(n), desc="Local neighbor search"):
        gi = grams[i]
        sims: List[Tuple[str, float]] = []
        for j in range(n):
            if i == j:
                continue
            gj = grams[j]
            if not gi and not gj:
                score = 0.0
            else:
                inter = len(gi & gj)
                union = len(gi | gj) if (gi or gj) else 1
                score = inter / union
            sims.append((ids[j], score))
        sims.sort(key=lambda x: x[1], reverse=True)
        neighbors[ids[i]] = [x[0] for x in sims[:topk]]

    click.echo("[INFO] Completed local neighbor search")
    return neighbors


# -----------------------------
# Pairs & metrics
# -----------------------------
def candidate_pairs_from_neighbors(neighbors: Dict[str, List[str]], reg_df: pd.DataFrame, seed: int = 13) -> List[Dict[str, Any]]:
    info = {row["object_id"]: {"name": row["registry_name"], "acronym": row["acronym"]} for _, row in reg_df.iterrows()}
    records: Dict[str, Dict[str, Any]] = {}
    for left, neighs in neighbors.items():
        for right in neighs:
            if left == right:
                continue
            ph = pair_hash_id(left, right, seed=seed)
            if ph in records:
                continue
            linfo = info.get(left, {"name": "", "acronym": ""})
            rinfo = info.get(right, {"name": "", "acronym": ""})
            records[ph] = {
                "pair_hash_id": ph,
                "left_name": linfo.get("name", ""),
                "left_acronym": linfo.get("acronym", ""),
                "right_name": rinfo.get("name", ""),
                "right_acronym": rinfo.get("acronym", ""),
                "left_object_id": left,
                "right_object_id": right,
            }
    return list(records.values())


def compute_recall(labeled_df: pd.DataFrame, cand_set: Set[Tuple[str, str]], n_objects: int) -> Dict[str, Any]:
    pos = labeled_df[labeled_df["label"] == 1].copy()
    pos["left_object_id"] = pos["left_object_id"].astype(str)
    pos["right_object_id"] = pos["right_object_id"].astype(str)
    gt = {unordered_pair(a, b) for a, b in zip(pos["left_object_id"], pos["right_object_id"])}
    n_pos, n_hit = len(gt), len(gt & cand_set)
    recall = n_hit / n_pos if n_pos else 0.0
    n_pairs = len(cand_set)
    total = n_objects * (n_objects - 1) // 2
    reduction_ratio = 1 - (n_pairs / total) if total > 0 else 0.0
    return {
        "n_pos": n_pos,
        "n_hit": n_hit,
        "recall": recall,
        "n_pairs": n_pairs,
        "n_objects": n_objects,
        "total_possible_pairs": total,
        "reduction_ratio": reduction_ratio,
    }


# -----------------------------
# CLI
# -----------------------------
@click.command()
@click.option("--annotated_json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--output_dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--topk", type=int, default=50, help="Number of neighbors to retrieve per registry")
@click.option("--seed", type=int, default=13)
def main(annotated_json: Path, output_dir: Path, topk: int, seed: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    click.echo("\n==============================\n[START] Semantic bucketing run (LOCAL)\n==============================")
    click.echo(f"Input: {annotated_json}\nOutput dir: {output_dir}\n")

    # Load annotated pairs
    click.echo("\n========== Loading Annotated Dataset ==========")
    df = pd.read_json(annotated_json)
    click.echo(f"Loaded annotated dataset ({len(df)} pairs)")

    # Build flattened registry
    click.echo("\n========== Building Flattened Registry ==========")
    registry = build_registry_from_labeled_pairs(df)
    click.echo(f"Flattened registry has {len(registry)} objects")

    # Save flattened registry
    click.echo("\n========== Saving Flattened Registry ==========")
    try:
        p = Path(output_dir) / "registry.json"
        registry.to_json(p, orient="records", lines=False, indent=4)
        click.echo(f"Saved flattened registry to {p}")
    except Exception:
        click.echo("[WARN] Failed to save flattened registry (non-fatal)")

    # Compute neighbors locally
    click.echo(f"\n========== Retrieving Neighbors (top-{topk}, LOCAL) ==========")
    click.echo("[INFO] Local neighbor search uses character trigram Jaccard similarity over the 'combined' field.")
    neighbors = local_topk_neighbors(registry, topk=topk)
    click.echo("Neighbor retrieval complete")

    # Save neighbors
    click.echo("\n========== Saving Neighbors ==========")
    try:
        p = Path(output_dir) / "neighbors.json"
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(neighbors, fh, indent=2)
        click.echo(f"Saved neighbors mapping to {p}")
    except Exception:
        click.echo("[WARN] Failed to save neighbors mapping (non-fatal)")

    # Build candidate pairs
    click.echo("\n========== Building Candidate Pairs ==========")
    pairs = candidate_pairs_from_neighbors(neighbors, registry, seed=seed)
    cand_set = {unordered_pair(r["left_object_id"], r["right_object_id"]) for r in pairs}
    click.echo(f"Built candidate pairs: {len(pairs)}")

    # Compute recall
    click.echo("\n========== Computing Recall Metrics ==========")
    metrics = compute_recall(df, cand_set, len(registry))
    click.echo(f"Recall metrics: {metrics}")

    # Write outputs
    click.echo("\n========== Writing Outputs ==========")
    pairs_df = pd.DataFrame(pairs)
    wrote_parquet = False
    try:
        pairs_df.to_parquet(Path(output_dir) / "candidate_pairs.parquet", index=False)
        wrote_parquet = True
    except Exception:
        click.echo("[WARN] Parquet write failed; falling back to JSON")
    if not wrote_parquet:
        pairs_df.to_json(Path(output_dir) / "candidate_pairs.json", orient="records", lines=False, indent=4)

    (Path(output_dir) / "metrics.json").write_text(json.dumps(metrics, indent=4))
    metadata = {"backend": "local-trigram-jaccard", "topk": topk}
    (Path(output_dir) / "metadata.json").write_text(json.dumps(metadata, indent=4))

    click.echo(f"\n==============================\n[SUCCESS] Wrote candidate pairs ({len(pairs_df)}) and metrics to {output_dir}\n==============================")


if __name__ == "__main__":
    main()
