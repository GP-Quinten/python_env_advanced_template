#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build 2nd-round search-engine datasets (queries & annotations) from Excel files.

This script replicates the notebook logic you shared:
- Reads queries from an Excel file with no header (column 0 = 'query_text'),
  trims whitespace, sorts by text, and assigns query IDs like '2nd-001'.
- Reads annotations by concatenating all sheets except 'Data', then:
    * keeps rows with a non-null 'query_text'
    * merges on 'query_text' to bring 'query_id'
    * selects the first non-null value across several "fit to request" columns
      (with 'Maybe' treated as missing)
    * maps Yes→YES, No→NO
    * assigns sequential annotation IDs like '2nd-001'
    * outputs only: query_id, registry_id, annotation_id, annotation_label
- Saves both datasets as JSON, plus a small metadata JSON in the output dir.

Dependencies:
    pandas, click, openpyxl

Usage example:
    python build_second_round_dataset.py \
        --queries_xlsx /path/queries_GH20250725.xlsx \
        --annotations_xlsx /path/annotations_RR20250804.xlsx \
        --output_queries /path/out/queries.json \
        --output_annotations /path/out/annotations.json \
        --output_dir /path/out
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import click
import pandas as pd


FIT_TO_REQUEST_CANDIDATES: List[str] = [
    "fit to request_Ghinwa",
    "Soso fit for request",
    "Sonia fit for request",
    "fit to request_Ryane",
    "fit to request",
    "fit to request.1",
]


def _ensure_parent_dirs(*paths: Path) -> None:
    for p in paths:
        if p is None:
            continue
        parent = p if p.is_dir() else p.parent
        parent.mkdir(parents=True, exist_ok=True)


def _format_id(prefix: str, n: int) -> str:
    return f"{prefix}{n:03d}"


def build_queries(queries_xlsx: Path) -> pd.DataFrame:
    """
    Replicates notebook logic:
      - read Excel with header=None
      - rename col 0 -> 'query_text'
      - strip, sort
      - query_id via rank().astype('int') -> '2nd-XXX'
      - keep ['query_id','query_text']
    """
    df = (
        pd.read_excel(queries_xlsx, header=None)
        .rename(columns={0: "query_text"})
        .assign(query_text=lambda d: d["query_text"].astype(str).str.strip())
        .loc[lambda d: d["query_text"].notna() & (d["query_text"] != "")]
        .sort_values("query_text", kind="mergesort")  # stable
        .assign(
            query_id=lambda d: d["query_text"]
            .rank()  # default: average ranking
            .astype("int")
            .apply(lambda r: _format_id("2nd-", int(r)))
        )
        [["query_id", "query_text"]]
        .reset_index(drop=True)
    )
    return df


def _first_non_null(series: pd.Series):
    for v in series:
        if pd.notna(v):
            return v
    return pd.NA


def build_annotations(annotations_xlsx: Path, df_queries: pd.DataFrame) -> pd.DataFrame:
    """
    Replicates notebook logic across sheets (skipping 'Data'):
      - concat sheets (only rows with non-null 'query_text')
      - left-merge on 'query_text' to bring 'query_id'
      - pick first non-null across fit-to-request columns after trimming and masking "Maybe"
      - map Yes→YES, No→NO
      - assign sequential annotation_id ('2nd-XXX') using expanding().count equivalent
      - select ['query_id','registry_id','annotation_id','annotation_label']
    """
    # read all sheets
    all_sheets = pd.read_excel(annotations_xlsx, sheet_name=None)
    dfs = []
    for sheet_name, df in all_sheets.items():
        if sheet_name == "Data":
            continue
        if "query_text" not in df.columns:
            # Skip sheets without the expected column
            continue
        df_local = df.loc[lambda d: d["query_text"].notna()].copy()
        dfs.append(df_local)

    if not dfs:
        return pd.DataFrame(columns=["query_id", "registry_id", "annotation_id", "annotation_label"])

    df_ann = pd.concat(dfs, ignore_index=True)

    # Merge with queries to get query_id
    df_ann = df_ann.merge(df_queries, on="query_text", how="left")

    # keep only if any of the fit columns exists and is non-null
    present_cols = [c for c in FIT_TO_REQUEST_CANDIDATES if c in df_ann.columns]
    if not present_cols:
        # No fit columns found -> empty result
        return pd.DataFrame(columns=["query_id", "registry_id", "annotation_id", "annotation_label"])

    # Normalize: strip; mask 'Maybe' (case-insensitive)
    for col in present_cols:
        df_ann[col] = (
            df_ann[col]
            .astype("string")
            .str.strip()
            .mask(df_ann[col].astype("string").str.strip().str.casefold() == "maybe")
        )

    df_ann = df_ann.loc[lambda d: d[present_cols].notna().any(axis=1)].copy()

    # First non-null across present fit columns
    df_ann["fit_to_request"] = df_ann[present_cols].apply(_first_non_null, axis=1)

    # Keep only rows with a usable label
    def _map_label(v) -> str | pd._libs.missing.NAType:
        if pd.isna(v):
            return pd.NA
        s = str(v).strip()
        if s.lower() == "yes":
            return "YES"
        if s.lower() == "no":
            return "NO"
        return pd.NA

    df_ann["annotation_label"] = df_ann["fit_to_request"].apply(_map_label)
    df_ann = df_ann.loc[df_ann["annotation_label"].notna()].copy()

    # Assign annotation_id using the same expanding().count idea from the notebook
    # (equivalent here to a simple 1..N sequence over current order)
    df_ann["__tmp_rownum"] = range(1, len(df_ann) + 1)
    df_ann["annotation_id"] = df_ann["__tmp_rownum"].apply(lambda n: _format_id("2nd-", int(n)))
    df_ann.drop(columns="__tmp_rownum", inplace=True)

    # Final projection
    wanted_cols = ["query_id", "registry_id", "annotation_id", "annotation_label"]
    # Ensure registry_id exists (if not, create empty column to match notebook selection)
    if "registry_id" not in df_ann.columns:
        df_ann["registry_id"] = pd.NA

    df_ann = df_ann[wanted_cols].reset_index(drop=True)
    return df_ann


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
    present_fit_cols: List[str],
) -> None:
    # Ensure output_dir exists
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_label_value_counts = df_annotations["annotation_label"].value_counts(dropna=False).to_dict()
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
            "annotations": int(len(df_annotations)),
        },
        "fit_to_request_columns_used": present_fit_cols,
        "annotation_label_value_counts": annotation_label_value_counts,
        "notes": [
            "Query IDs are formatted as '2nd-XXX' using pandas rank() as in the notebook.",
            "Annotation IDs are sequential '2nd-XXX' over the filtered annotation rows.",
            "Values equal to 'Maybe' (case-insensitive) were treated as missing.",
            "annotation_label is YES/NO based on fit_to_request."
        ],
        "version": "1.0.0"
    }
    meta_path = output_dir / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)


@click.command()
@click.option(
    "--queries_xlsx",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the Excel file containing 2nd round queries.",
)
@click.option(
    "--annotations_xlsx",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the Excel file containing 2nd round annotations.",
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

    logging.info("Building queries dataset from %s", queries_xlsx)
    df_queries = build_queries(queries_xlsx)
    save_json_records(df_queries, output_queries)
    logging.info("Saved queries JSON: %s (%d rows)", output_queries, len(df_queries))

    logging.info("Building annotations dataset from %s", annotations_xlsx)
    df_annotations = build_annotations(annotations_xlsx, df_queries)
    save_json_records(df_annotations, output_annotations)
    logging.info("Saved annotations JSON: %s (%d rows)", output_annotations, len(df_annotations))

    # For metadata, re-detect which fit columns were present in annotations input
    all_sheets = pd.read_excel(annotations_xlsx, sheet_name=None)
    present_fit_cols = sorted({
        col for df in all_sheets.values() for col in FIT_TO_REQUEST_CANDIDATES if col in df.columns
    })

    write_metadata(
        output_dir=output_dir,
        queries_xlsx=queries_xlsx,
        annotations_xlsx=annotations_xlsx,
        output_queries=output_queries,
        output_annotations=output_annotations,
        df_queries=df_queries,
        df_annotations=df_annotations,
        present_fit_cols=present_fit_cols,
    )
    logging.info("Metadata written to: %s", output_dir / "metadata.json")
    logging.info("Done.")


if __name__ == "__main__":
    main()
