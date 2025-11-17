#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
S23_soft_clustering.py
Soft clustering over pairwise predictions with PageRank-based memberships.

Interface compatible with the hard-clustering UF script:
- manifest.json -> batch prediction CSVs with pair_hash_id and <pred_column>
- candidate_pairs.parquet -> maps pair_hash_id -> left/right object_id
- registry_dataset.json -> full object_id universe

Accepts --ilp_max_nodes and --ilp_time_limit for pipeline compatibility (ignored here).

Usage (example):
python S23_soft_clustering.py \
  --manifest_json data/R22_inference_unlabeled/manifest.json \
  --pairs_parquet data/R20_bucket_unlabeled_dataset/candidate_pairs.parquet \
  --registry_json data/R01_load_registry_dataset/registry_dataset.json \
  --output_dir data/R23_soft_clustering \
  --threshold 0.87 \
  --cluster_prefix C \
  --pred_column proba \
  --n_batch -1
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

# Optional: networkx for connected components (falls back to UF if missing)
try:
    import networkx as nx
except Exception:
    nx = None


# -------------------------
# Helpers
# -------------------------
def info(msg: str):
    click.echo(f"[INFO] {msg}", err=False)

def warn(msg: str):
    click.echo(f"[WARN] {msg}", err=True)

def build_edges_from_batches(
    batch_files: List[Path],
    pairs_df: pd.DataFrame,
    pred_column: str,
    threshold: float,
) -> Tuple[Dict[Tuple[int, int], float], Set[int]]:
    """
    Read all batch CSVs, keep rows with proba >= threshold, join to pairs_df to map to object_ids.
    For each pair (li, ri) keep max probability across batches.
    Returns:
        edges: {(min_id,max_id): prob}
        touched_ids: set of object_ids present in >=1 accepted edge
    """
    edges: Dict[Tuple[int, int], float] = {}
    touched_ids: Set[int] = set()

    info(f"Scanning predictions in {len(batch_files)} batch files...")
    for bpath in tqdm(sorted(batch_files), desc="Batches", unit="batch"):
        usecols = ["pair_hash_id", pred_column]
        try:
            pred_df = pd.read_csv(bpath, usecols=usecols)
        except ValueError:
            pred_df = pd.read_csv(bpath)
            if pred_column not in pred_df.columns:
                raise click.ClickException(f"Prediction column '{pred_column}' not found in {bpath}.")
            pred_df = pred_df[["pair_hash_id", pred_column]]

        pred_df["pair_hash_id"] = pred_df["pair_hash_id"].astype(str)
        pred_df = pred_df[pred_df[pred_column] >= threshold]
        if pred_df.empty:
            continue

        joined = pred_df.join(pairs_df, on="pair_hash_id", how="inner")
        if joined.empty:
            continue

        for li, ri, p in zip(joined["left_object_id"].values,
                             joined["right_object_id"].values,
                             joined[pred_column].values):
            li = int(li); ri = int(ri)
            if li == ri:
                continue
            a, b = (li, ri) if li < ri else (ri, li)
            p = float(p)
            prev = edges.get((a, b))
            if (prev is None) or (p > prev):
                edges[(a, b)] = p
            touched_ids.add(a); touched_ids.add(b)

    info(f"Accepted unique edges: {len(edges):,} | Touched IDs: {len(touched_ids):,}")
    return edges, touched_ids


def components_from_edges(edges: Dict[Tuple[int, int], float], thr: float) -> List[Set[int]]:
    """
    Connected components using only edges with weight >= thr.
    Uses networkx if available; otherwise union-find.
    """
    e2 = [(u, v) for (u, v), w in edges.items() if w >= thr]
    if not e2:
        return []

    if nx is not None:
        G = nx.Graph()
        G.add_edges_from(e2)
        return [set(c) for c in nx.connected_components(G)]

    # Fallback UF
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    nodes = set()
    for u, v in e2:
        nodes.add(u); nodes.add(v)
        union(u, v)
    comp = {}
    for n in nodes:
        r = find(n)
        comp.setdefault(r, set()).add(n)
    return list(comp.values())


def adjacency_for_component(comp_nodes: List[int],
                            edges: Dict[Tuple[int, int], float],
                            thr: float):
    """
    Build adjacency dict and degree (weighted) for a component using only edges >= thr.
    Returns:
        neighbors: {u: {v: w, ...}, ...}
        deg: {u: sum_w}
    """
    node_set = set(comp_nodes)
    neighbors: Dict[int, Dict[int, float]] = {u: {} for u in comp_nodes}
    deg: Dict[int, float] = {u: 0.0 for u in comp_nodes}
    for (a, b), w in edges.items():
        if w < thr:
            continue
        if (a in node_set) and (b in node_set):
            neighbors[a][b] = w
            neighbors[b][a] = w
            deg[a] += w
            deg[b] += w
    return neighbors, deg


def ppr_membership_for_seedset(
    comp_nodes: List[int],
    neighbors: Dict[int, Dict[int, float]],
    deg: Dict[int, float],
    seed_nodes: Set[int],
    alpha: float = 0.15,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> Dict[int, float]:
    """
    Personalized PageRank over the component graph with restarts on seed_nodes.
    Returns score per node in comp_nodes.
    """
    if not seed_nodes:
        return {u: 0.0 for u in comp_nodes}
    v = {u: (1.0 / len(seed_nodes) if u in seed_nodes else 0.0) for u in comp_nodes}
    r = v.copy()

    for _ in range(max_iter):
        r_new = {u: alpha * v[u] for u in comp_nodes}
        for u in comp_nodes:
            du = deg.get(u, 0.0)
            if du <= 0.0:
                continue
            push = (1.0 - alpha) * r[u] / du
            for vtx, w in neighbors[u].items():
                r_new[vtx] = r_new.get(vtx, 0.0) + push * w
        diff = sum(abs(r_new[u] - r[u]) for u in comp_nodes)
        r = r_new
        if diff < tol:
            break
    return r


# -------------------------
# Main
# -------------------------
@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--manifest_json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Path to R22 manifest.json (lists per-batch prediction CSVs).")
@click.option("--pairs_parquet", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Path to R20 candidate_pairs.parquet (pair_hash_id,left_object_id,right_object_id).")
@click.option("--registry_json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Path to R01 registry_dataset.json (provides full object_id universe).")
@click.option("--output_dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Directory to write outputs.")
@click.option("--threshold", default=0.50, show_default=True, type=float,
              help="Keep an edge if prediction >= threshold (base graph).")
@click.option("--cluster_prefix", default="C", show_default=True, type=str,
              help="Prefix for cluster IDs.")
@click.option("--pred_column", default="proba", show_default=True, type=str,
              help="Name of prediction column in batch CSVs.")
@click.option("--n_batch", default=-1, show_default=True, type=int,
              help="Number of batch files to process (-1 for all).")
# Soft-clustering specific knobs
@click.option("--core_threshold", default=None, type=float,
              help="Stricter threshold to define seed cores (default: min(1.0, max(threshold+0.10, 0.90))).")
@click.option("--alpha", default=0.15, show_default=True, type=float,
              help="Restart probability for Personalized PageRank.")
@click.option("--min_membership", default=0.02, show_default=True, type=float,
              help="Trim memberships below this value from the output list.")
@click.option("--topk_memberships", default=3, show_default=True, type=int,
              help="Keep at most top-K memberships per node in object_clusters.json.")
# Compatibility flags (ignored but accepted so Snakemake rules don't break)
@click.option("--ilp_max_nodes", default=60, show_default=True, type=int,
              help="[Compat only] Ignored in soft clustering.")
@click.option("--ilp_time_limit", default=0, show_default=True, type=int,
              help="[Compat only] Ignored in soft clustering (seconds).")
def main(manifest_json: Path, pairs_parquet: Path, registry_json: Path, output_dir: Path,
         threshold: float, cluster_prefix: str, pred_column: str, n_batch: int,
         core_threshold: float | None, alpha: float, min_membership: float, topk_memberships: int,
         ilp_max_nodes: int, ilp_time_limit: int):

    output_dir.mkdir(parents=True, exist_ok=True)
    info(f"Using base probability threshold: {threshold}")

    # --- Load registry (universe of object_ids) ---
    info(f"Loading registry objects: {registry_json}")
    reg_df = pd.read_json(registry_json)
    if "object_id" not in reg_df.columns:
        raise click.ClickException("registry_json must contain 'object_id'.")
    all_ids: List[int] = list(pd.Series(reg_df["object_id"]).astype(int).unique())
    info(f"Found {len(all_ids):,} unique object_ids.")
    reg_indexed = reg_df.set_index("object_id")

    # --- Load pairs mapping ---
    info(f"Loading pairs mapping: {pairs_parquet}")
    pairs_df = pd.read_parquet(pairs_parquet, columns=["pair_hash_id", "left_object_id", "right_object_id"])
    if pairs_df.empty:
        raise click.ClickException("pairs_parquet is empty.")
    pairs_df["left_object_id"] = pairs_df["left_object_id"].astype(int)
    pairs_df["right_object_id"] = pairs_df["right_object_id"].astype(int)
    pairs_df["pair_hash_id"] = pairs_df["pair_hash_id"].astype(str)
    pairs_df.set_index("pair_hash_id", inplace=True)
    info(f"Pairs mapping rows: {len(pairs_df):,}")

    # --- Read manifest for batch CSVs ---
    info(f"Reading manifest: {manifest_json}")
    with open(manifest_json, "r") as f:
        manifest = json.load(f)
    batch_entries = manifest.get("batch_outputs") or []
    if not batch_entries:
        raise click.ClickException("manifest.json has no 'batch_outputs'. Did R22 run?")
    batch_files: List[Path] = [Path(d["predictions_csv"]) for d in batch_entries]
    if n_batch != -1:
        batch_files = batch_files[:n_batch]
        info(f"Only processing first {n_batch} batch files.")

    # --- Build weighted edges & touched nodes from batches ---
    edges, touched_ids = build_edges_from_batches(batch_files, pairs_df, pred_column, threshold)

    # If nothing touched, still emit outputs with null clusters
    if not touched_ids:
        info("No edges above threshold; emitting empty clustering outputs.")
        _emit_empty_outputs(all_ids, reg_indexed, output_dir)
        return

    # --- Define seed cores at a stricter threshold ---
    if core_threshold is None:
        core_threshold = min(1.0, max(threshold + 0.10, 0.90))
    info(f"Core threshold for seeds: {core_threshold}")

    seed_components = components_from_edges(edges, core_threshold)
    seed_components = [set(c) & touched_ids for c in seed_components if len(c & touched_ids) > 0]
    info(f"Seed components (cores): {len(seed_components)}")

    # Assign cluster IDs to seed components
    seed_id_list: List[Tuple[str, Set[int]]] = []
    for idx, core_nodes in enumerate(sorted(seed_components, key=lambda s: (-len(s), min(s)))):
        cid = f"{cluster_prefix}{idx+1:06d}"
        seed_id_list.append((cid, core_nodes))

    # --- Base components at threshold for membership diffusion ---
    base_components = components_from_edges(edges, threshold)
    info(f"Base components: {len(base_components)}")

    # Compute memberships
    node_memberships: Dict[int, List[Tuple[str, float]]] = {}
    cluster_effective_size: Dict[str, float] = {cid: 0.0 for cid, _ in seed_id_list}

    def adjacency_for_component(comp_nodes: List[int],
                                edges: Dict[Tuple[int, int], float],
                                thr: float):
        node_set = set(comp_nodes)
        neighbors: Dict[int, Dict[int, float]] = {u: {} for u in comp_nodes}
        deg: Dict[int, float] = {u: 0.0 for u in comp_nodes}
        for (a, b), w in edges.items():
            if w < thr:
                continue
            if (a in node_set) and (b in node_set):
                neighbors[a][b] = w
                neighbors[b][a] = w
                deg[a] += w
                deg[b] += w
        return neighbors, deg

    for comp in tqdm(base_components, desc="Soft clustering over components"):
        comp_nodes = sorted(list(comp))
        neighbors, deg = adjacency_for_component(comp_nodes, edges, threshold)

        # Find seeds inside this component
        seeds_in_comp: List[Tuple[str, Set[int]]] = []
        for cid, core_nodes in seed_id_list:
            inter = core_nodes & comp
            if inter:
                seeds_in_comp.append((cid, inter))

        # If none, synthesize one (hub-based)
        if not seeds_in_comp:
            # pick highest-degree node as seed (or singleton)
            if len(comp_nodes) == 1:
                cid = f"{cluster_prefix}{len(seed_id_list)+1:06d}"
                seeds_in_comp.append((cid, set(comp_nodes)))
                seed_id_list.append((cid, set(comp_nodes)))
                cluster_effective_size[cid] = 0.0
            else:
                deg_items = sorted(((u, deg.get(u, 0.0)) for u in comp_nodes), key=lambda x: (-x[1], x[0]))
                hub = deg_items[0][0]
                # small local core
                local_core = {hub} | {v for v, w in neighbors[hub].items() if w >= max(threshold, (core_threshold + threshold)/2.0)}
                cid = f"{cluster_prefix}{len(seed_id_list)+1:06d}"
                seeds_in_comp.append((cid, local_core))
                seed_id_list.append((cid, local_core))
                cluster_effective_size[cid] = 0.0

        # PPR per seed
        seed_scores: Dict[str, Dict[int, float]] = {}
        for cid, seed_nodes in seeds_in_comp:
            r = ppr_membership_for_seedset(comp_nodes, neighbors, deg, seed_nodes, alpha=alpha)
            seed_scores[cid] = r

        # Normalize per-node across seeds
        for u in comp_nodes:
            denom = sum(seed_scores[cid][u] for cid, _ in seeds_in_comp)
            if denom <= 0:
                memb = [(cid, 1.0 / len(seeds_in_comp)) for cid, _ in seeds_in_comp]
            else:
                memb = [(cid, seed_scores[cid][u] / denom) for cid, _ in seeds_in_comp]
            memb = [(cid, s) for cid, s in memb if s >= min_membership]
            memb.sort(key=lambda x: -x[1])
            if topk_memberships > 0:
                memb = memb[:topk_memberships]
            node_memberships[u] = memb
            for cid, s in memb:
                cluster_effective_size[cid] = cluster_effective_size.get(cid, 0.0) + s

    # --- Build object_clusters.json (top hard cluster + memberships list) ---
    objects_out: List[dict] = []
    for oid in all_ids:
        entry = {
            "object_id": int(oid),
            "cluster_id": None,
            "cluster_size": None,
            "cluster_size_effective": None,
            "memberships": None,
            "registry_name": reg_indexed.at[int(oid), "registry_name"] if "registry_name" in reg_indexed.columns else None,
            "acronym": reg_indexed.at[int(oid), "acronym"] if "acronym" in reg_indexed.columns else None,
        }
        if oid in node_memberships:
            memb = node_memberships[oid]
            entry["memberships"] = [{"cluster_id": cid, "score": float(s)} for cid, s in memb]
            if memb:
                top_cid, _ = memb[0]
                entry["cluster_id"] = top_cid
                eff = cluster_effective_size.get(top_cid, 0.0)
                entry["cluster_size_effective"] = float(eff)
                entry["cluster_size"] = int(round(eff))
        objects_out.append(entry)

    clusters_out_path = output_dir / "object_clusters.json"
    with open(clusters_out_path, "w", encoding="utf-8") as f:
        json.dump(objects_out, f, indent=2, ensure_ascii=False)
    info(f"object_clusters.json → {clusters_out_path}")

    # --- Summary ---
    summary = {
        "threshold": threshold,
        "core_threshold": core_threshold,
        "alpha": alpha,
        "min_membership": min_membership,
        "topk_memberships": topk_memberships,
        "n_objects": len(all_ids),
        "n_edges": len(edges),
        "n_touched": len(touched_ids),
        "n_seed_clusters": len({m['cluster_id'] for rec in objects_out for m in (rec.get('memberships') or [])}),
        "inputs": {
            "manifest_json": str(manifest_json),
            "pairs_parquet": str(pairs_parquet),
            "registry_json": str(registry_json),
            "pred_column": pred_column,
        },
        "params": {
            "cluster_prefix": cluster_prefix,
            "n_batch": n_batch,
            "ilp_max_nodes": ilp_max_nodes,   # recorded for compatibility
            "ilp_time_limit": ilp_time_limit  # recorded for compatibility
        },
        "outputs": {"object_clusters": str(clusters_out_path)},
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    info(f"Summary → {output_dir / 'summary.json'}")

    # --- Report (by effective size) ---
    report_path = output_dir / "report.md"
    cluster_effective_size = {  # recompute quickly from objects_out
        cid: sum(m["score"] for rec in objects_out for m in (rec.get("memberships") or []) if m["cluster_id"] == cid)
        for cid in {m["cluster_id"] for rec in objects_out for m in (rec.get("memberships") or [])}
    }
    cluster_size = {
        cid: sum(1 for rec in objects_out if rec.get("cluster_id") == cid)
        for cid in cluster_effective_size.keys()
    }
    eff_sizes = sorted(cluster_effective_size.items(), key=lambda x: -x[1])
    top20 = eff_sizes[:20]
    bottom20 = [x for x in sorted(eff_sizes, key=lambda x: x[1]) if x[1] > 0][:20]

    # Prepare DataFrame for grouping
    df_report = pd.DataFrame(objects_out)
    df_report = df_report[df_report["cluster_id"].notnull()]
    grp = df_report.groupby("cluster_id")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Cluster Report\n\n")
        f.write("## Top 20 Largest Clusters (by effective size)\n\n")
        for cid, sz in top20:
            size = cluster_size.get(cid, 0)
            f.write(f"### {cid}  (effective_size={sz:.1f}, size={size})\n")
            if cid in grp.groups:
                group = grp.get_group(cid)
                names = group["registry_name"].dropna().unique()[:20]
                acronyms = group["acronym"].dropna().unique()[:20]
                for i, n in enumerate(names):
                    acro = acronyms[i] if i < len(acronyms) else ""
                    if acro:
                        f.write(f"- {n} ({acro})\n")
                    else:
                        f.write(f"- {n}\n")
            f.write("\n")
        f.write("## 20 Smallest Clusters (by effective size)\n\n")
        for cid, sz in bottom20:
            size = cluster_size.get(cid, 0)
            f.write(f"### {cid}  (effective_size={sz:.1f}, size={size})\n")
            if cid in grp.groups:
                group = grp.get_group(cid)
                names = group["registry_name"].dropna().unique()[:20]
                acronyms = group["acronym"].dropna().unique()[:20]
                for i, n in enumerate(names):
                    acro = acronyms[i] if i < len(acronyms) else ""
                    if acro:
                        f.write(f"- {n} ({acro})\n")
                    else:
                        f.write(f"- {n}\n")
            f.write("\n")
    info(f"Report → {report_path}")

    # --- Pairs report for top 20 clusters (after object_clusters is ready)
    pairs_report_path = output_dir / "pairs_report.md"
    with open(pairs_report_path, "w", encoding="utf-8") as f:
        f.write("# Pairs Report: Top 20 Largest Clusters\n\n")
        for cid, sz in top20:
            f.write(f"## Cluster {cid} (effective_size={sz:.1f}, size={cluster_size.get(cid, 0)})\n")
            group = grp.get_group(cid) if cid in grp.groups else pd.DataFrame()
            pairs = []
            # For each object in cluster, find other objects in same cluster and their membership scores
            members = group["object_id"].tolist() if not group.empty else []
            for i, oid1 in enumerate(members):
                memb1 = node_memberships.get(oid1, [])
                for oid2 in members[i+1:]:
                    memb2 = node_memberships.get(oid2, [])
                    # Find shared cluster membership
                    score1 = next((m[1] for m in memb1 if m[0] == cid), None)
                    score2 = next((m[1] for m in memb2 if m[0] == cid), None)
                    if score1 is not None and score2 is not None:
                        pairs.append({
                            "left_object_id": oid1,
                            "right_object_id": oid2,
                            "left_registry_name": group[group["object_id"] == oid1]["registry_name"].values[0] if "registry_name" in group.columns else str(oid1),
                            "right_registry_name": group[group["object_id"] == oid2]["registry_name"].values[0] if "registry_name" in group.columns else str(oid2),
                            "left_acronym": group[group["object_id"] == oid1]["acronym"].values[0] if "acronym" in group.columns else "",
                            "right_acronym": group[group["object_id"] == oid2]["acronym"].values[0] if "acronym" in group.columns else "",
                            "score_left": score1,
                            "score_right": score2
                        })
            if not pairs:
                f.write("No pairs found for this cluster.\n\n")
                continue
            for pair in pairs[:100]:  # limit to 100 pairs per cluster
                left_name = pair.get("left_registry_name") or str(pair["left_object_id"])
                right_name = pair.get("right_registry_name") or str(pair["right_object_id"])
                left_acro = pair.get("left_acronym") or ""
                right_acro = pair.get("right_acronym") or ""
                score_l = round(pair["score_left"], 3)
                score_r = round(pair["score_right"], 3)
                f.write(f"- {left_name} ({left_acro}) ↔ {right_name} ({right_acro}) | scores=({score_l}, {score_r})\n")
            f.write("\n")
    info(f"Pairs report → {pairs_report_path}")

    # --- Export edges/nodes for Gephi visualization
    edges_csv = output_dir / "edges.csv"
    nodes_csv = output_dir / "nodes.csv"

    # Build edges: use all pairs with membership scores above min_membership
    edges_out = []
    for oid, membs in node_memberships.items():
        for cid, score in membs:
            # Find other objects in same cluster
            for oid2, membs2 in node_memberships.items():
                if oid2 <= oid:
                    continue
                for cid2, score2 in membs2:
                    if cid2 == cid and score >= min_membership and score2 >= min_membership:
                        edges_out.append({
                            "Source": oid,
                            "Target": oid2,
                            "Weight": round((score + score2) / 2, 4),
                            "Cluster": cid
                        })
    # Also include hard cluster edges (top membership only)
    # (optional, can be skipped if above is sufficient)

    # Build nodes
    nodes_out = []
    for entry in objects_out:
        nodes_out.append({
            "Id": entry["object_id"],
            "Label": entry.get("registry_name") or str(entry["object_id"]),
            "ClusterId": entry.get("cluster_id"),
            "ClusterSize": entry.get("cluster_size"),
            "Acronym": entry.get("acronym")
        })

    pd.DataFrame(edges_out).to_csv(edges_csv, index=False)
    pd.DataFrame(nodes_out).to_csv(nodes_csv, index=False)
    info(f"Gephi edges CSV → {edges_csv}")
    info(f"Gephi nodes CSV → {nodes_csv}")

    info("[SUCCESS] Soft clustering complete.")


def _emit_empty_outputs(all_ids: List[int], reg_indexed: pd.DataFrame, output_dir: Path):
    objects_out = []
    for oid in all_ids:
        objects_out.append({
            "object_id": int(oid),
            "cluster_id": None,
            "cluster_size": None,
            "cluster_size_effective": None,
            "memberships": None,
            "registry_name": reg_indexed.at[int(oid), "registry_name"] if "registry_name" in reg_indexed.columns else None,
            "acronym": reg_indexed.at[int(oid), "acronym"] if "acronym" in reg_indexed.columns else None,
        })
    clusters_out_path = output_dir / "object_clusters.json"
    with open(clusters_out_path, "w", encoding="utf-8") as f:
        json.dump(objects_out, f, indent=2, ensure_ascii=False)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"note": "no edges above threshold; empty clustering"}, f, indent=2)

    with open(output_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("# Soft Clustering Report\n\nNo edges above threshold.\n")


if __name__ == "__main__":
    main()
