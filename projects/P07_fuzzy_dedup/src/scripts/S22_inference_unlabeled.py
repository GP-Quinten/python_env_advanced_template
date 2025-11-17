#!/usr/bin/env python3
"""
Run inference on precomputed feature batches (from R21) using ONLY the 'model'
step in a fitted sklearn Pipeline. Assumes the model supports predict_proba.

Inputs
------
--pipeline       : joblib path to fitted Pipeline (must have step named 'model')
--features_dir   : directory with feature batches from R21 (CSR .npz)
--summary_json   : R21 summary.json (lists 'features_batches' and totals)
--row_index_csv  : R21 row_index.csv.gz (pair_hash_id,left_name,right_name,...), rows align with batches concatenation

Outputs (all in --output_dir)
-----------------------------
- batches/predictions_batch_XXXX.csv.gz : per-batch predictions with columns:
      pair_hash_id,left_name,right_name,proba
- prediction_samples.parquet            : random sample from all batches (pair_hash_id, proba, batch_idx)
- prediction_distribution.png           : histogram of sampled probabilities
- manifest.json                         : metadata, list of batch outputs, counts, etc.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any

import click
import joblib
import numpy as np
import pandas as pd
from scipy import sparse as sp
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


IDX_COL = "pair_hash_id"
LEFT_COL = "left_name"
RIGHT_COL = "right_name"
# add acronym constants
LEFT_ACRONYM = "left_acronym"
RIGHT_ACRONYM = "right_acronym"
# add object_id constants
LEFT_OBJECT_ID = "left_object_id"
RIGHT_OBJECT_ID = "right_object_id"
# add embedding constants
LEFT_EMBEDDING = "left_embedding"
RIGHT_EMBEDDING = "right_embedding"


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--pipeline", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Path to fitted sklearn Pipeline (joblib). Must contain step named 'model'.")
@click.option("--features_dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Directory containing R21 outputs (feature batches).")
@click.option("--summary_json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Path to R21 summary.json (contains 'features_batches').")
@click.option("--row_index_csv", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Path to R21 row_index.csv.gz (aligned with batches).")
# add optional candidate_pairs parquet
@click.option("--candidate_pairs_parquet", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False,
              help="Optional candidate_pairs.parquet to merge additional columns (acronyms, IDs).")
@click.option("--output_dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Directory to write predictions artifacts.")
@click.option("--sample_rows", default=200000, show_default=True, type=int,
              help="Total number of rows to sample across all batches for distribution plot.")
@click.option("--seed", default=42, show_default=True, type=int,
              help="Random seed for reproducibility.")
# new options for diagnostic sampling
@click.option("--sample_n", default=100, show_default=True, type=int,
              help="Number of top/mid/bottom samples to save")
@click.option("--mid_center", default=0.5, show_default=True, type=float,
              help="Center probability for mid‐samples")
def main(pipeline: Path, features_dir: Path, summary_json: Path, row_index_csv: Path,
         candidate_pairs_parquet: Path, output_dir: Path, sample_rows: int, seed: int, sample_n: int, mid_center: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    batches_dir = output_dir / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"[INFO] Loading pipeline: {pipeline}")
    pipe = joblib.load(pipeline)

    if "model" not in pipe.named_steps:
        raise click.ClickException(f"Pipeline has no step named 'model'. Steps: {list(pipe.named_steps.keys())}")

    est = pipe.named_steps["model"]
    click.echo(f"[INFO] Using model step: {type(est).__name__}")

    click.echo(f"[INFO] Reading summary: {summary_json}")
    with open(summary_json, "r") as f:
        summary = json.load(f)
    # summary.json now nests batch list under "outputs"
    outputs = summary.get("outputs", {})

    features_batches: List[str] = outputs.get("features_batches", []) or []
    if not features_batches:
        features_batches = sorted(str(p) for p in features_dir.glob("features_batch_*.npz"))
    if not features_batches:
        raise click.ClickException("No feature batches found.")

    total_rows = int(summary.get("n_rows", 0))
    click.echo(f"[INFO] Found {len(features_batches)} feature batches; total_rows={total_rows:,}")

    # Read the entire row index CSV once
    click.echo(f"[INFO] Reading full row index CSV: {row_index_csv}")
    idx_df_full = pd.read_csv(row_index_csv)

    # if provided, merge in only the object_id cols from candidate_pairs.parquet
    if candidate_pairs_parquet:
        click.echo(f"[INFO] Reading candidate pairs parquet: {candidate_pairs_parquet}")
        cp_df = pd.read_parquet(candidate_pairs_parquet)[[IDX_COL, LEFT_OBJECT_ID, RIGHT_OBJECT_ID]]
        idx_df_full = idx_df_full.merge(cp_df, on=IDX_COL, how="left")

    if len(idx_df_full) < total_rows:
        click.echo(f"[WARN] Row index CSV has fewer rows ({len(idx_df_full)}) than total_rows ({total_rows})")

    rng = np.random.default_rng(seed)
    sample_target = int(sample_rows)
    sample_accum: List[pd.DataFrame] = []
    remaining = sample_target

    manifest_batches: List[Dict[str, Any]] = []

    start_idx = 0
    for b_idx, batch_path in enumerate(tqdm(sorted(features_batches), desc="Batches", unit="batch")):
        batch_path = Path(batch_path)
        Xb = sp.load_npz(batch_path)
        n_rows_b = Xb.shape[0]

        end_idx = start_idx + n_rows_b
        if end_idx > len(idx_df_full):
            raise click.ClickException("Row index exhausted before feature batches; alignment mismatch.")

        idx_df = idx_df_full.iloc[start_idx:end_idx].copy()
        start_idx = end_idx

        # include acronyms in the column‐presence check
        for col in (IDX_COL, LEFT_COL, RIGHT_COL, LEFT_ACRONYM, RIGHT_ACRONYM, LEFT_OBJECT_ID, RIGHT_OBJECT_ID):
            if col not in idx_df.columns:
                raise click.ClickException(f"Missing column in row_index: {col}")

        # Predict probabilities
        proba = est.predict_proba(Xb)[:, 1]

        # Output per-batch, now including acronyms and object_ids
        out_df = idx_df[
            [IDX_COL, LEFT_COL, RIGHT_COL, LEFT_ACRONYM, RIGHT_ACRONYM, LEFT_OBJECT_ID, RIGHT_OBJECT_ID]
        ].copy()
        out_df["proba"] = proba
        batch_out = batches_dir / f"predictions_batch_{b_idx:04d}.csv.gz"
        out_df.to_csv(batch_out, index=False, compression="gzip")

        # Sample
        take = 0
        if sample_target > 0 and total_rows > 0:
            take = int(np.ceil(sample_target * (n_rows_b / total_rows)))
            take = min(take, remaining, n_rows_b)
        if take > 0:
            sample_idx = rng.choice(n_rows_b, size=take, replace=False) if n_rows_b > take else np.arange(n_rows_b)
            sample_df = out_df.iloc[sample_idx].copy()
            sample_df["batch_idx"] = b_idx
            sample_accum.append(sample_df)
            remaining -= take

        manifest_batches.append({
            "batch_index": b_idx,
            "features_batch": str(batch_path),
            "predictions_csv": str(batch_out),
            "rows": int(n_rows_b),
        })

    # Save sample and plot
    sample_out = output_dir / "prediction_samples.parquet"
    if sample_accum:
        sample_all = pd.concat(sample_accum, axis=0, ignore_index=True)
        sample_all = sample_all[
            [IDX_COL, LEFT_COL, RIGHT_COL, LEFT_ACRONYM, RIGHT_ACRONYM, LEFT_OBJECT_ID, RIGHT_OBJECT_ID, "proba", "batch_idx"]
        ]
        try:
            sample_all.to_parquet(sample_out, index=False)
        except Exception:
            sample_all.to_csv(str(sample_out).replace(".parquet", ".csv.gz"), index=False, compression="gzip")
            sample_out = Path(str(sample_out).replace(".parquet", ".csv.gz"))

        hist_png = output_dir / "prediction_distribution.png"
        plt.figure(figsize=(8, 5))
        plt.hist(sample_all["proba"].astype(float).values, bins=50)
        plt.xlabel("proba")
        plt.ylabel("Count")
        plt.title(f"Sampled probability distribution ({len(sample_all):,} rows)")
        plt.tight_layout()
        plt.savefig(hist_png)
        plt.close()
        click.echo(f"[INFO] Saved prediction distribution plot → {hist_png}")
    else:
        hist_png = None
        click.echo("[WARN] No samples collected; skipping sample save and plot.")

    # Manifest
    manifest = {
        "pipeline": str(pipeline),
        "estimator": type(est).__name__,
        "features_dir": str(features_dir),
        "row_index_csv": str(row_index_csv),
        "batches_written": len(manifest_batches),
        "pred_column": "proba",
        "batch_outputs": manifest_batches,
        "sample_file": str(sample_out) if sample_accum else None,
        "hist_png": str(hist_png) if hist_png else None,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # --- new diagnostics samples: top / mid / bottom
    click.echo("[INFO] Generating top/mid/bottom diagnostic samples...")
    # load all batch outputs
    all_preds = pd.concat(
        [pd.read_csv(b["predictions_csv"], compression="gzip") for b in manifest_batches],
        ignore_index=True
    )
    # pick bottom, top, and mid-center samples
    bottom = all_preds.nsmallest(sample_n, "proba").sort_values("proba")
    top = all_preds.nlargest(sample_n, "proba").sort_values("proba", ascending=False)
    mid_idx = (all_preds["proba"] - mid_center).abs().sort_values().head(sample_n).index
    mid = all_preds.loc[mid_idx].sort_values("proba")

    # save diagnostics (include object_id columns)
    bottom.to_parquet(output_dir / "bottom_samples.parquet", index=False)
    mid.to_parquet(output_dir / "mid_samples.parquet", index=False)
    top.to_parquet(output_dir / "top_samples.parquet", index=False)
    click.echo(f"[SUCCESS] Saved bottom {sample_n} → bottom_samples.parquet")
    click.echo(f"[SUCCESS] Saved mid {sample_n} → mid_samples.parquet")
    click.echo(f"[SUCCESS] Saved top {sample_n} → top_samples.parquet")

    # new: also save JSONL for easy inspection
    bottom.to_json(output_dir / "bottom_samples.jsonl", orient="records", lines=True)
    mid.to_json(output_dir / "mid_samples.jsonl", orient="records", lines=True)
    top.to_json(output_dir / "top_samples.jsonl", orient="records", lines=True)
    click.echo(f"[SUCCESS] Saved bottom {sample_n} → bottom_samples.jsonl")
    click.echo(f"[SUCCESS] Saved mid {sample_n} → mid_samples.jsonl")
    click.echo(f"[SUCCESS] Saved top {sample_n} → top_samples.jsonl")

    click.echo(f"[SUCCESS] Wrote manifest → {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
