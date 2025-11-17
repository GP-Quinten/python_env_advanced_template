#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build 1st-round search-engine datasets (queries & annotations) from Excel files.

This script mirrors the notebook you shared:
- Queries:
    * Read Excel with headers, expecting columns: 'id', 'query'
    * Rename to ['query_id', 'query_text']
    * Format query_id as '1st-XXX' (zero-padded)
    * Output JSON list of records
- Annotations:
    * Read all sheets from the Excel file
    * For each sheet:
        - Keep rows with numeric 'query_id'
        - Keep rows where 'registry_name' is present and != 'Not available'
        - Drop columns starting with 'Unnamed:'
    * Concatenate sheets
    * Sort by ['query_id', 'registry_name']
    * Create integer index 'annotation_id' (0..N-1)
    * Map each integer 'annotation_id' to a list of registry IDs via the provided mapping
    * Compose IDs:
        - query_id => '1st-XXX'
        - annotation_id => '1st-XXX' where XXX = index+1
        - explode registry_ids to 'registry_id'
        - add per-registry suffix to annotation_id: '1st-XXX-NNN'
    * annotation_label is always 'YES'
    * Select ['annotation_id', 'query_id', 'registry_id', 'annotation_label']
    * Output JSON list of records
- Writes a metadata.json with basic stats in --output_dir.

Dependencies:
    click, pandas, openpyxl

Example:
    python build_first_round_dataset.py \
      --queries_xlsx /path/queries_JP20250718.xlsx \
      --annotations_xlsx /path/annotations_SZ20250721.xlsx \
      --output_queries /path/out/1st_queries.json \
      --output_annotations /path/out/1st_annotations.json \
      --output_dir /path/out
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import click
import pandas as pd


def _ensure_parent_dirs(*paths: Path) -> None:
    for p in paths:
        parent = p if p.is_dir() else p.parent
        parent.mkdir(parents=True, exist_ok=True)


def _format_id(prefix: str, n: int) -> str:
    return f"{prefix}{n:03d}"


def build_queries(queries_xlsx: Path) -> pd.DataFrame:
    """Replicate the notebook logic for 1st-round queries."""
    df = pd.read_excel(queries_xlsx)

    if not {"id", "query"}.issubset(df.columns):
        missing = {"id", "query"} - set(df.columns)
        raise ValueError(f"Missing expected columns in queries file: {sorted(missing)}")

    df = (
        df.rename(columns={"id": "query_id", "query": "query_text"})[["query_id", "query_text"]]
        .assign(
            query_id=lambda d: pd.to_numeric(d["query_id"], errors="raise").astype(int),
            query_text=lambda d: d["query_text"].astype(str).str.strip(),
        )
        .assign(query_id=lambda d: d["query_id"].apply(lambda q: _format_id("1st-", q)))
        .reset_index(drop=True)
    )
    return df


# ---- mapping from notebook ----
MAPPING: Dict[int, List[int]] = {
    0: [15266, 28296],
    1: [23122, 54418],
    2: [24400],
    3: [134, 3929, 12678, 939, 11671, 12736, 4255, 12570, 11323, 4141, 20272, 30259, 7524, 13642, 12467, 24836, 14877, 26025, 28487, 29634, 33493, 35694, 41743, 42097, 43737, 45170, 46019, 46823, 48138],
    4: [385, 5233, 10289, 19471, 17240, 22221, 34135, 39412, 48710, 53360],
    5: [2891, 26243, 10277, 5494, 19231, 15398, 20581, 23900, 345943],
    6: [],  # FIXME https://sci-hub.st/https://doi.org/10.1111/j.1399-3046.2004.00232.x
    7: [4956],
    8: [3202, 1863, 3426, 2138, 18618, 27081, 2062, 24872, 22093, 27183, 50958, 12470, 11153, 26903, 20522, 11690, 28422, 29986, 30149, 33217, 29829, 36920, 38096, 41742, 39261, 43434, 46114, 46770, 53439],
    9: [],  # FIXME https://catalogues.ema.europa.eu/node/2880/administrative-details
    10: [36354],
    11: [19908],
    12: [274, 7627, 54583],
    13: [15826, 43304, 51529],
    14: [54345],
    15: [39792, 54367],
    16: [195, 1028, 22976, 34002, 35105, 48757, 54564],
    17: [54374, 54475, 280, 44056, 54413],
    18: [54430],
    19: [54490],
    20: [],  # FIXME not found
    21: [],  # FIXME not found
    22: [54550],
    23: [5013, 27367, 50509, 47395, 27366],  # maybe others
    24: [5298],
    25: [],  # TODO
    26: [54373],
    27: [10567, 44394],
    28: [54431],
    29: [],  # FIXME not found
    30: [12419],
    31: [54511],
    32: [54518],
    33: [],  # FIXME not found
    34: [28804],
    35: [2087, 37688, 2978, 33829, 7810, 27508, 41707, 16550, 26107, 10999, 4467, 18034, 16791, 22718, 20672, 24332, 24147, 25461, 18547, 17003, 17581, 32885, 32086, 28343, 33779, 33830, 35635, 35636, 32924, 37054, 38299, 40135, 41295, 43541, 47800, 49159, 52978],
    36: [4036, 8825],
    37: [3326],
    38: [1373],
    39: [54464],
    40: [212],
    41: [23620],
    42: [],
    43: [],
    44: [1459, 11322, 8527, 7922, 32926, 36701, 48081],
    45: [8318, 1106, 7892, 17480, 41840],
    46: [6641, 3422, 608, 29589],
    47: [5765, 24233],
    48: [26854, 51467],
    49: [2041, 1326, 131, 530, 18222, 2473, 10459, 13468, 8330, 5027, 3273, 48171],
    50: [8834, 18404, 33761, 17137, 7389, 19108, 19897, 39981, 42002, 52335, 53289],
    51: [220, 4614],
    52: [1541, 44081],
    53: [7520, 43336],
    54: [10838],
    # 55: [280, 44056, 907, 52483, 3656, 40406, 49796, 54413],
    56: [30280, 44627],
    57: [54545],
    58: [44476],
    59: [10491],
    60: [6465],
    61: [22531],
    62: [4982],
    63: [6020, 11948],
    64: [38781, 54216],
    65: [13784, 13522, 30629],
}


def _read_annotation_sheets(annotations_xlsx: Path) -> Tuple[List[pd.DataFrame], List[str], List[str]]:
    """Read and clean all sheets; return (dfs_kept, sheet_names_kept, sheet_names_skipped)."""
    xls = pd.read_excel(annotations_xlsx, sheet_name=None)
    dfs: List[pd.DataFrame] = []
    kept, skipped = [], []

    for name, df in xls.items():
        # Normalize columns; skip if required fields missing
        cols = set(df.columns)
        if not {"query_id", "registry_name"}.issubset(cols):
            skipped.append(name)
            continue

        # Clean rows
        qid_num = pd.to_numeric(df["query_id"], errors="coerce")
        df_clean = df.loc[
            (qid_num.notna())
            & df["registry_name"].notna()
            & (df["registry_name"].astype(str).str.strip() != "Not available"),
            ~df.columns.str.startswith("Unnamed:"),
        ].copy()

        if df_clean.empty:
            skipped.append(name)
            continue

        # ensure types
        df_clean["query_id"] = qid_num.loc[df_clean.index].astype(int)
        df_clean["registry_name"] = df_clean["registry_name"].astype(str).str.strip()

        dfs.append(df_clean)
        kept.append(name)

    return dfs, kept, skipped


def build_annotations(annotations_xlsx: Path) -> pd.DataFrame:
    """Replicate the notebook logic for 1st-round annotations."""
    dfs, kept, skipped = _read_annotation_sheets(annotations_xlsx)

    if not dfs:
        logging.warning("No valid annotation sheets found. Skipped: %s", skipped)
        return pd.DataFrame(columns=["annotation_id", "query_id", "registry_id", "annotation_label"])

    df = pd.concat(dfs, ignore_index=True)

    # Sort and create integer index 'annotation_id' (0..N-1)
    df = df.sort_values(["query_id", "registry_name"]).reset_index(drop=True)
    df = df.reset_index().rename(columns={"index": "annotation_id"})  # 0..N-1

    # Attach registry_ids list via mapping on integer annotation_id
    df["registry_ids"] = df["annotation_id"].apply(lambda i: MAPPING.get(int(i), []))

    # Compose IDs before exploding
    df["query_id"] = df["query_id"].apply(lambda q: _format_id("1st-", int(q)))
    df["annotation_id"] = df["annotation_id"].apply(lambda i: _format_id("1st-", int(i) + 1))

    # Explode registry_ids -> registry_id
    df = (
        df.explode("registry_ids", ignore_index=True)
        .rename(columns={"registry_ids": "registry_id"})
    )

    # Drop rows where mapping produced an empty list (explode yields NaN)
    df = df.loc[df["registry_id"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["annotation_id", "query_id", "registry_id", "annotation_label"])

    # Per-registry sequence within each (query_id, annotation_id)
    df = df.sort_values(["query_id", "annotation_id", "registry_id"]).reset_index(drop=True)
    df["n"] = df.groupby(["query_id", "annotation_id"]).cumcount() + 1
    df["annotation_id"] = df.apply(lambda r: f"{r['annotation_id']}-{int(r['n']):03d}", axis=1)

    # Final shape
    df["annotation_label"] = "YES"
    df = df[["annotation_id", "query_id", "registry_id", "annotation_label"]].reset_index(drop=True)
    return df


def save_json_records(df: pd.DataFrame, path: Path) -> None:
    records = df.to_dict(orient="records")
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)


def write_metadata(
    output_dir: Path,
    queries_xlsx: Path,
    annotations_xlsx: Path,
    output_queries: Path,
    output_annotations: Path,
    df_queries: pd.DataFrame,
    df_annotations: pd.DataFrame,
    sheets_kept: List[str],
    sheets_skipped: List[str],
    extra: dict | None = None,  # <-- add extra argument
) -> None:
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "queries_xlsx": str(queries_xlsx),
            "annotations_xlsx": str(annotations_xlsx),
        },
        "outputs": {
            "queries_json": str(output_queries),
            "annotations_json": str(output_annotations),
        },
        "counts": {
            "queries": int(len(df_queries)),
            "annotations_rows": int(len(df_annotations)),
            "unique_annotation_ids": int(df_annotations["annotation_id"].nunique() if not df_annotations.empty else 0),
        },
        "sheets": {
            "kept": sheets_kept,
            "skipped": sheets_skipped,
        },
        "notes": [
            "Queries expect columns: 'id', 'query'.",
            "Annotations filtered to numeric query_id and registry_name != 'Not available'.",
            "annotation_id formatted as '1st-XXX-NNN' after explode; query_id as '1st-XXX'.",
            "annotation_label set to 'YES' for all exploded rows.",
            "Mapping dictionary copied from the notebook; empty lists are dropped after explode.",
        ],
        "version": "1.0.0",
    }
    if extra:
        meta.update(extra)
    meta_path = output_dir / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)


@click.command()
@click.option(
    "--queries_xlsx",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the Excel file containing queries.",
)
@click.option(
    "--annotations_xlsx",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the Excel file containing annotations.",
)
@click.option(
    "--output_queries",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Path to output JSON file for processed queries.",
)
@click.option(
    "--output_annotations",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Path to output JSON file for processed annotations.",
)
@click.option(
    "--output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory to store output files and metadata.",
)
def main(
    queries_xlsx: Path,
    annotations_xlsx: Path,
    output_queries: Path,
    output_annotations: Path,
    output_dir: Path,
):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logging.info("Ensuring output directories exist…")
    _ensure_parent_dirs(output_dir, output_queries, output_annotations)

    logging.info("Reading & building queries from: %s", queries_xlsx)
    df_queries = build_queries(queries_xlsx)
    save_json_records(df_queries, output_queries)
    logging.info("Saved queries JSON -> %s (%d rows)", output_queries, len(df_queries))

    logging.info("Reading & building annotations from: %s", annotations_xlsx)
    # Collect sheet stats for metadata in a lightweight pass
    dfs, kept, skipped = _read_annotation_sheets(annotations_xlsx)
    if dfs:
        df_annotations = pd.concat(dfs, ignore_index=True)
        # Re-run the remainder of the pipeline on the concatenated df
        df_annotations = (
            df_annotations.sort_values(["query_id", "registry_name"])
            .reset_index(drop=True)
            .reset_index()
            .rename(columns={"index": "annotation_id"})
        )
        df_annotations["registry_ids"] = df_annotations["annotation_id"].apply(lambda i: MAPPING.get(int(i), []))
        df_annotations["query_id"] = df_annotations["query_id"].apply(lambda q: _format_id("1st-", int(q)))
        df_annotations["annotation_id"] = df_annotations["annotation_id"].apply(lambda i: _format_id("1st-", int(i) + 1))
        df_annotations = (
            df_annotations.explode("registry_ids", ignore_index=True)
            .rename(columns={"registry_ids": "registry_id"})
        )
        df_annotations = df_annotations.loc[df_annotations["registry_id"].notna()].copy()
        if not df_annotations.empty:
            df_annotations = df_annotations.sort_values(["query_id", "annotation_id", "registry_id"]).reset_index(drop=True)
            df_annotations["n"] = df_annotations.groupby(["query_id", "annotation_id"]).cumcount() + 1
            df_annotations["annotation_id"] = df_annotations.apply(lambda r: f"{r['annotation_id']}-{int(r['n']):03d}", axis=1)
            df_annotations["annotation_label"] = "YES"
            df_annotations = df_annotations[["annotation_id", "query_id", "registry_id", "annotation_label"]]
        else:
            df_annotations = pd.DataFrame(columns=["annotation_id", "query_id", "registry_id", "annotation_label"])
    else:
        kept, skipped = [], list(pd.read_excel(annotations_xlsx, sheet_name=None).keys())
        df_annotations = pd.DataFrame(columns=["annotation_id", "query_id", "registry_id", "annotation_label"])

    save_json_records(df_annotations, output_annotations)
    logging.info("Saved annotations JSON -> %s (%d rows)", output_annotations, len(df_annotations))

    # compute label counts dict
    label_counts = (
        df_annotations["annotation_label"].value_counts().to_dict()
        if not df_annotations.empty else {}
    )

    # pass into write_metadata
    write_metadata(
        output_dir=output_dir,
        queries_xlsx=queries_xlsx,
        annotations_xlsx=annotations_xlsx,
        output_queries=output_queries,
        output_annotations=output_annotations,
        df_queries=df_queries,
        df_annotations=df_annotations,
        sheets_kept=kept,
        sheets_skipped=skipped,
        extra={"annotation_label_counts": label_counts},
    )
    logging.info("Metadata written -> %s", output_dir / "metadata.json")
    logging.info("Done.")


if __name__ == "__main__":
    main()
