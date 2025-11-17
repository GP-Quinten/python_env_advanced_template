#!/usr/bin/env python3
# file: S06_explore_results.py
#
# This script generates human-readable reports and statistics from the deduplication results.
# It helps analysts understand the clusters of duplicate records that were found.
#
# Key outputs:
# 1. clusters.csv - Complete dataset sorted by cluster for detailed inspection
# 2. clusters_summary.csv - Statistical overview of each cluster (size, samples)
# 3. report.md - Human-readable Markdown report with top clusters expanded
# 4. stats.json - Overall statistics (total counts, distribution) for programmatic use

import json
import logging
from pathlib import Path
from typing import List, Optional

import click
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _read_json_any(path: Path) -> pd.DataFrame:
    """Reads JSON or JSONL into a DataFrame. Tries different formats to be robust."""
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    try:
        return pd.read_json(path)
    except ValueError:
        return pd.read_json(path, lines=True)


def _safe_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Return the first existing column from candidates, else None.
    This helps handle different naming conventions in input datasets.
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--clustered_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to clustered dataset (from S05; must contain 'cluster_id').",
)
@click.option(
    "--edges_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False,
    help="Optional: accepted edges JSON from S05 (idx_i, idx_j).",
)
@click.option(
    "--output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory to write outputs.",
)
@click.option(
    "--clusters_csv_filename",
    default="clusters.csv",
    show_default=True,
    help="Filename for full per-row export, sorted by cluster.",
)
@click.option(
    "--summary_csv_filename",
    default="clusters_summary.csv",
    show_default=True,
    help="Filename for per-cluster summary CSV.",
)
@click.option(
    "--report_md_filename",
    default="report.md",
    show_default=True,
    help="Filename for human-readable Markdown report.",
)
@click.option(
    "--stats_json_filename",
    default="stats.json",
    show_default=True,
    help="Filename for overall statistics JSON.",
)
@click.option(
    "--min_cluster_size",
    default=2,
    show_default=True,
    help="Only include clusters of this size or larger in the summary/report.",
)
@click.option(
    "--top_k",
    default=30,
    show_default=True,
    help="Number of largest clusters to expand in the Markdown report.",
)
@click.option(
    "--members_limit",
    default=50,
    show_default=True,
    help="Max members to list per cluster in the Markdown report.",
)
def main(
    clustered_json: Path,
    edges_json: Optional[Path],
    output_dir: Path,
    clusters_csv_filename: str,
    summary_csv_filename: str,
    report_md_filename: str,
    stats_json_filename: str,
    min_cluster_size: int,
    top_k: int,
    members_limit: int,
):
    """
    Explore deduplication results and save them in readable formats.
    Produces:
      - clusters.csv               (all rows, sorted)
      - clusters_summary.csv       (one row per cluster with size & samples)
      - report.md                  (stats + top clusters expanded)
      - stats.json                 (counts, distribution)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Load data ----------
    log.info(f"Loading clustered dataset: {clustered_json}")
    df = _read_json_any(clustered_json)
    if "cluster_id" not in df.columns:
        raise click.ClickException("Input clustered dataset must contain 'cluster_id'.")

    # Heuristic name columns - looks for common column names for registry names
    # and uses the first one it finds for reporting
    raw_col = _safe_col(df, ["registry_name", "name", "title"])
    norm_col = _safe_col(df, ["norm__registry_name", "normalized_name", "norm_name"])
    acronym_col = _safe_col(df, ["acronym", "abbr"])

    n_rows = len(df)
    log.info(f"Loaded {n_rows} rows.")

    # ---------- Sort & export full rows ----------
    # This creates the complete clusters.csv with all rows sorted by cluster_id
    # Helpful for detailed inspection of each cluster's contents
    sort_keys = ["cluster_id"]
    if raw_col:
        sort_keys.append(raw_col)
    df_sorted = df.sort_values(sort_keys)
    clusters_csv_path = output_dir / clusters_csv_filename
    df_sorted.to_csv(clusters_csv_path, index=False)
    log.info(f"Wrote full export: {clusters_csv_path}")

    # ---------- Build per-cluster summary ----------
    # Count how many records are in each cluster and filter small clusters
    sizes = df.groupby("cluster_id").size().rename("size").sort_values(ascending=False)
    sizes_filt = sizes[sizes >= min_cluster_size]

    def _examples(series: pd.Series, k: int = 5) -> List[str]:
        """Get up to k unique examples from a series, for reporting."""
        uniq = series.dropna().astype(str).unique().tolist()
        return uniq[:k]

    # Create a summary row for each cluster with size and example values
    summary_rows = []
    for cid, size in sizes_filt.items():
        rows = df[df["cluster_id"] == cid]
        ex_raw = _examples(rows[raw_col], 5) if raw_col else []
        ex_norm = _examples(rows[norm_col], 5) if norm_col else []
        ex_acr = _examples(rows[acronym_col], 5) if acronym_col else []
        summary_rows.append(
            {
                "cluster_id": cid,
                "size": int(size),
                "sample_raw": ex_raw,
                "sample_norm": ex_norm,
                "sample_acronym": ex_acr,
            }
        )

    # Create and save the clusters_summary.csv (statistical overview)
    summary_df = pd.DataFrame(summary_rows).sort_values(["size", "cluster_id"], ascending=[False, True])
    summary_csv_path = output_dir / summary_csv_filename
    summary_df.to_csv(summary_csv_path, index=False)
    log.info(f"Wrote cluster summary: {summary_csv_path}")

    # ---------- Edges stats (optional) ----------
    # If edges_json was provided, calculate some network statistics
    # This shows how the records are connected in the deduplication graph
    edges_stats = {}
    if edges_json:
        log.info(f"Loading edges: {edges_json}")
        e_df = _read_json_any(edges_json)
        if not {"idx_i", "idx_j"}.issubset(e_df.columns):
            log.warning("Edges file doesn't have 'idx_i' and 'idx_j'; skipping edge stats.")
        else:
            m = len(e_df)
            # Approx unique nodes that appear in edges
            touched = pd.Index(e_df["idx_i"]).union(pd.Index(e_df["idx_j"]))
            edges_stats = {
                "edges_count": int(m),
                "unique_nodes_in_edges": int(len(touched)),
                "avg_degree_estimate": float(2 * m / max(1, len(touched))),
            }

    # ---------- Global stats ----------
    # Calculate overall statistics about the clustering results
    n_clusters = int(sizes.shape[0])
    n_clusters_filt = int(sizes_filt.shape[0])
    largest = int(sizes.iloc[0]) if n_clusters > 0 else 0
    singles = int((df["cluster_id"].value_counts() == 1).sum())

    # Create a stats dictionary to be saved as JSON
    stats = {
        "rows": int(n_rows),
        "clusters_total": n_clusters,
        "clusters_with_size_ge_min": n_clusters_filt,
        "largest_cluster_size": largest,
        "singleton_clusters": singles,
        "min_cluster_size_filter": int(min_cluster_size),
        "name_columns": {
            "raw_col": raw_col,
            "norm_col": norm_col,
            "acronym_col": acronym_col,
        },
        **({"edges": edges_stats} if edges_stats else {}),
    }

    # Save the statistics as JSON for programmatic use
    stats_path = output_dir / stats_json_filename
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=3, ensure_ascii=False)
    log.info(f"Wrote stats: {stats_path}")

    # ---------- Markdown report ----------
    # Generate a human-readable Markdown report focusing on the top_k largest clusters
    top = summary_df.head(top_k).copy()

    def _fmt_list(lst: List[str], limit: int) -> str:
        """Format a list of strings with a limit, adding '... (+X more)' if needed."""
        if not lst:
            return "-"
        if len(lst) <= limit:
            return ", ".join(map(str, lst))
        return ", ".join(map(str, lst[:limit])) + f" … (+{len(lst) - limit} more)"

    # Build the Markdown report with sections for overview and top clusters
    report_lines = []
    report_lines.append("# Deduplication Report (Tier-A)")
    report_lines.append("")
    report_lines.append("## Overview")
    report_lines.append("")
    report_lines.append(f"- Total rows: **{n_rows}**")
    report_lines.append(f"- Total clusters: **{n_clusters}**")
    report_lines.append(f"- Singleton clusters (size = 1): **{singles}**")
    report_lines.append(f"- Largest cluster size: **{largest}**")
    report_lines.append(f"- Clusters with size ≥ {min_cluster_size}: **{n_clusters_filt}**")
    if edges_stats:
        report_lines.append(f"- Accepted edges: **{edges_stats['edges_count']}**, "
                            f"unique nodes in edges: **{edges_stats['unique_nodes_in_edges']}**, "
                            f"avg degree (est): **{edges_stats['avg_degree_estimate']:.2f}**")
    report_lines.append("")
    report_lines.append("## Top clusters")
    report_lines.append("")
    
    # For each top cluster, create a section with details and members
    for _, row in top.iterrows():
        cid = row["cluster_id"]
        size = int(row["size"])
        report_lines.append(f"### {cid} (size {size})")
        # Gather members (limited)
        members = df[df["cluster_id"] == cid]
        # Prefer raw names if present, else normalized
        if raw_col:
            names_series = members[raw_col].astype(str)
            report_lines.append(f"- Sample raw: {_fmt_list(row.get('sample_raw', []), 5)}")
        if norm_col:
            report_lines.append(f"- Sample norm: {_fmt_list(row.get('sample_norm', []), 5)}")
        if acronym_col:
            acr = row.get("sample_acronym", [])
            if acr:
                report_lines.append(f"- Sample acronym: {_fmt_list(acr, 5)}")
        # Expanded member list (limited)
        list_col = raw_col or norm_col
        if list_col:
            all_members = members[list_col].astype(str).tolist()
            shown = all_members[:members_limit]
            if list_col == raw_col:
                report_lines.append(f"- Members (first {min(members_limit, len(all_members))}):")
            else:
                report_lines.append(f"- Members (normalized, first {min(members_limit, len(all_members))}):")
            for v in shown:
                report_lines.append(f"  - {v}")
            if len(all_members) > members_limit:
                report_lines.append(f"  - … (+{len(all_members) - members_limit} more)")
        report_lines.append("")

    # Save the Markdown report
    report_md_path = output_dir / report_md_filename
    report_md_path.write_text("\n".join(report_lines), encoding="utf-8")
    log.info(f"Wrote Markdown report: {report_md_path}")

    log.info("Done.")


if __name__ == "__main__":
    main()
