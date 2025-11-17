#!/usr/bin/env python3
"""
Featurize unlabeled candidate pairs using ONLY the preprocessing/featurization
part of a fitted sklearn pipeline (no model inference).

Inputs
------
- --pipeline: joblib path to fitted Pipeline from R12.
- --pairs_json: JSON with at least:
      pair_hash_id, left_name, right_name
  (produced by R20_bucket_unlabeled_dataset)

Outputs (into --output_dir)
---------------------------
- features_csr.npz            : CSR sparse matrix (scipy.sparse), rows aligned to row_index.csv.gz
- feature_names.json          : list of feature names (if available; else f0..f{n-1})
- row_index.csv.gz            : pair_hash_id order (+ left/right names for convenience)
- features_sample.parquet     : small sample (rows x <=512 features) for ad-hoc inspection
- feature_distributions.png   : density bars (sparse) or histograms (dense) for sampled features
- summary.json                : metadata (n_rows, n_features, sparsity, etc.)

Notes
-----
- We try to slice the pipeline to exclude the final estimator (pipe[:-1]).
- If slicing is not supported, we look for a step named 'preprocess'/'features'/'featurizer'.
- If the resulting object doesn't implement .transform, we raise a clear error.
"""

from __future__ import annotations
import json
import math
import warnings
from pathlib import Path

import click
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm as tqdm
from scipy import sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gzip
import shutil

from typing import Any, List, Tuple, Optional

LEFT_OBJECT_ID = "left_object_id"
RIGHT_OBJECT_ID = "right_object_id"
LEFT_COL = "left_name"
RIGHT_COL = "right_name"
IDX_COL = "pair_hash_id"
LEFT_ACRONYM = "left_acronym"
RIGHT_ACRONYM = "right_acronym"
LEFT_EMBEDDING="left_embedding"
RIGHT_EMBEDDING="right_embedding"


def _extract_featurizer(pipe) -> Any:
    """Return a fitted transformer that exposes .transform(X)."""
    # 1) sklearn Pipeline supports slicing in >=1.1 (common)
    try:
        sub = pipe[:-1]
        if hasattr(sub, "transform"):
            return sub
    except Exception:
        pass

    # 2) Look in named_steps for likely featurizer keys
    try:
        steps = getattr(pipe, "named_steps", {})
        for key in ("preprocess", "features", "featurizer", "vectorizer", "transform"):
            if key in steps and hasattr(steps[key], "transform"):
                return steps[key]
    except Exception:
        pass

    # 3) If the whole pipeline is itself a transformer (rare), use it
    if hasattr(pipe, "transform"):
        return pipe

    raise click.ClickException(
        "Could not locate a featurizer/transformer in the pipeline. "
        "Expected a Pipeline with preprocessing steps before the final estimator."
    )


def _get_feature_names(featurizer, input_cols: Optional[List[str]] = None, n_features: Optional[int] = None) -> List[str]:
    """Best-effort extraction of feature names."""
    try:
        # Some transformers want input_features=...
        try:
            names = featurizer.get_feature_names_out(input_cols)  # type: ignore
        except TypeError:
            names = featurizer.get_feature_names_out()  # type: ignore
        names = list(map(str, names))
        if len(names) == 0 and n_features is not None:
            return [f"f{i}" for i in range(n_features)]
        return names
    except Exception:
        if n_features is None:
            return []
        return [f"f{i}" for i in range(n_features)]


def _plot_sparse_density(X: sp.csr_matrix, feature_names: List[str], out_png: Path, max_features: int, seed: int):
    """Plot density (non-zero ratio) for a sampled subset of features."""
    rng = np.random.default_rng(seed)
    n_features = X.shape[1]
    k = min(max_features, n_features)
    # sample features uniformly
    feat_idx = np.sort(rng.choice(n_features, size=k, replace=False)) if n_features > k else np.arange(n_features)
    # compute density for sampled features
    # nnz per feature from CSR: use CSC for fast per-column counts
    Xc = X.tocsc()
    counts = np.diff(Xc.indptr)[feat_idx]
    density = counts / X.shape[0]
    labels = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in feat_idx]

    plt.figure(figsize=(max(8, k * 0.3), 5))
    plt.bar(range(k), density)
    plt.xticks(range(k), labels, rotation=90)
    plt.ylabel("Non-zero ratio")
    plt.title(f"Feature density for {k} sampled features")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def _plot_dense_hists(X_sample: np.ndarray, feature_names: List[str], out_png: Path, max_features: int, seed: int):
    """Plot histograms for a sampled subset of features."""
    rng = np.random.default_rng(seed)
    n_features = X_sample.shape[1]
    k = min(max_features, n_features)
    cols = np.sort(rng.choice(n_features, size=k, replace=False)) if n_features > k else np.arange(n_features)

    # grid layout
    ncols = int(math.ceil(math.sqrt(k)))
    nrows = int(math.ceil(k / ncols))
    fig_w = max(8, ncols * 3)
    fig_h = max(6, nrows * 2.5)

    plt.figure(figsize=(fig_w, fig_h))
    for idx, j in enumerate(cols, start=1):
        ax = plt.subplot(nrows, ncols, idx)
        ax.hist(X_sample[:, j], bins=50)
        title = feature_names[j] if j < len(feature_names) else f"f{j}"
        ax.set_title(str(title), fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--pipeline", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Path to fitted sklearn Pipeline (joblib).")
@click.option("--pairs_parquet", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Path to candidate_pairs.parquet from R20.")
@click.option("--preloaded_embeddings", type=click.Path(exists=True, dir_okay=False, path_type=Path),
                help="Optional: local embeddings parquet from S10a_preload_embeddings.py")
@click.option("--new_preloaded_embeddings", type=click.Path(exists=True, dir_okay=False, path_type=Path),
                help="Optional: local embeddings parquet from S10a_preload_embeddings.py for new registries to inference. Necessary if there are saved in another file")
@click.option("--output_dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Directory for outputs.")
@click.option("--batch_size", default=200000, show_default=True, type=int)
@click.option("--sample_rows", default=100000, show_default=True, type=int)
@click.option("--max_plot_features", default=48, show_default=True, type=int)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--n_batches", default=-1, show_default=True, type=int,
              help="Number of batches to process (-1 for all batches).")
def main(pipeline: Path, pairs_parquet: Path, preloaded_embeddings: Path, new_preloaded_embeddings: Path, output_dir: Path,
         batch_size: int, sample_rows: int, max_plot_features: int, seed: int, n_batches: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"[INFO] Loading pipeline: {pipeline}")
    pipe = joblib.load(pipeline)
    click.echo(f"[INFO] Pipeline loaded successfully.")
    featurizer = _extract_featurizer(pipe)
    click.echo(f"[INFO] Featurizer extracted: {type(featurizer).__name__}")

    click.echo(f"[INFO] Streaming unlabeled pairs from Parquet: {pairs_parquet}")
    pqf = pq.ParquetFile(pairs_parquet)
    click.echo(f"[INFO] Parquet file opened. Total rows: {pqf.metadata.num_rows}")

    # --- Embeddings handling (simplified: no intermediate Parquet write) ---
    have_embeddings = False
    emb_map = None
    if preloaded_embeddings:
        click.echo("[INFO] Loading embeddings into memory (dict map)...")
        emb = pd.read_parquet(preloaded_embeddings)[["object_id", "registry_name_embedding"]]
        if new_preloaded_embeddings:
            click.echo("[INFO] Loading new embeddings into memory and concatenating...")
            emb_new = pd.read_parquet(new_preloaded_embeddings)[["object_id", "registry_name_embedding"]]
            emb = pd.concat([emb, emb_new], ignore_index=True)
        emb["object_id"] = emb["object_id"].astype(str)
        # Build dictionary: str(object_id) -> embedding (list / vector)
        emb_map = dict(zip(emb["object_id"].values, emb["registry_name_embedding"].values))
        have_embeddings = True
        click.echo(f"[INFO] Embeddings loaded: {len(emb_map):,} ids. Will attach via fast map (no per-batch merge).")
        del emb  # free original frame

    total_rows = pqf.metadata.num_rows
    pbar = tqdm(total=total_rows, unit="rows", desc="Transforming")

    n_features = None
    row_index_csv = output_dir / "row_index.csv"
    row_index_gz = output_dir / "row_index.csv.gz"
    batch_feature_files = []
    batch_row_counts = []
    batch_nnz_counts = []
    sample_rows_accum = []
    sample_feats_accum = []
    rng = np.random.default_rng(seed)
    sample_feat_cap = None

    LOG_EVERY = 10  # only emit a progress log every N batches (reduce noise)

    # streaming batches: now read only raw pair columns (object ids retained for merging)
    base_cols = [IDX_COL, LEFT_OBJECT_ID, RIGHT_OBJECT_ID, LEFT_COL, RIGHT_COL, LEFT_ACRONYM, RIGHT_ACRONYM]
    batch_iter = pqf.iter_batches(batch_size, columns=base_cols)
    batch_count = 0
    click.echo(f"[INFO] Processing batches of size {batch_size}...")
    for batch in batch_iter:
        if n_batches != -1 and batch_count >= n_batches:
            click.echo(f"[INFO] Reached n_batches={n_batches}; stopping batch processing.")
            break

        df_batch = batch.to_pandas()

        if have_embeddings:
            # Cast ids to str once, then map to embeddings (fast; avoids large merge)
            df_batch[LEFT_OBJECT_ID] = df_batch[LEFT_OBJECT_ID].astype(str)
            df_batch[RIGHT_OBJECT_ID] = df_batch[RIGHT_OBJECT_ID].astype(str)
            df_batch[LEFT_EMBEDDING] = df_batch[LEFT_OBJECT_ID].map(emb_map)
            df_batch[RIGHT_EMBEDDING] = df_batch[RIGHT_OBJECT_ID].map(emb_map)

        pbar.update(len(df_batch))

        # Columns required by featurizer
        feature_input_cols = [LEFT_COL, RIGHT_COL, LEFT_ACRONYM, RIGHT_ACRONYM]
        if have_embeddings:
            feature_input_cols += [LEFT_EMBEDDING, RIGHT_EMBEDDING]

        if batch_count == 0:
            df_batch[[IDX_COL, LEFT_COL, RIGHT_COL, LEFT_ACRONYM, RIGHT_ACRONYM]].to_csv(row_index_csv, index=False)
        else:
            df_batch[[IDX_COL, LEFT_COL, RIGHT_COL, LEFT_ACRONYM, RIGHT_ACRONYM]].to_csv(
                row_index_csv, index=False, header=False, mode="a"
            )

        Xb = featurizer.transform(df_batch[feature_input_cols])
        if not sp.issparse(Xb):
            Xb = sp.csr_matrix(np.asarray(Xb))
        else:
            Xb = Xb.tocsr()

        if n_features is None:
            n_features = Xb.shape[1]
            sample_feat_cap = min(512, n_features)

        batch_feat_path = output_dir / f"features_batch_{batch_count:04d}.npz"
        sp.save_npz(batch_feat_path, Xb)
        batch_feature_files.append(str(batch_feat_path))
        batch_row_counts.append(Xb.shape[0])
        batch_nnz_counts.append(Xb.nnz)

        if len(sample_rows_accum) < sample_rows:
            rows_needed = sample_rows - len(sample_rows_accum)
            take_rows = min(rows_needed, Xb.shape[0])
            if take_rows > 0:
                sample_rows_idx = (
                    rng.choice(Xb.shape[0], size=take_rows, replace=False)
                    if Xb.shape[0] > take_rows else np.arange(Xb.shape[0])
                )
                sample_feat_idx = (
                    rng.choice(n_features, size=sample_feat_cap, replace=False)
                    if n_features > sample_feat_cap else np.arange(n_features)
                )
                X_sample = Xb[sample_rows_idx][:, sample_feat_idx].toarray()
                sample_rows_accum.append(X_sample)
                sample_feats_accum.append(sample_feat_idx)

        if batch_count % LOG_EVERY == 0:
            click.echo(f"[INFO] Batch {batch_count}: rows {Xb.shape[0]}, features {Xb.shape[1]} (sparse nnz={Xb.nnz})")

        batch_count += 1

    pbar.close()

    click.echo(f"[INFO] Compressing row_index.csv → {row_index_gz}")
    with open(row_index_csv, "rb") as f_in:
        with gzip.open(row_index_gz, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    row_index_csv.unlink()
    click.echo(f"[INFO] Compressed row index saved → {row_index_gz}")

    # Feature names (best effort)
    click.echo(f"[INFO] Extracting feature names...")
    feature_names = _get_feature_names(featurizer, input_cols=[LEFT_COL, RIGHT_COL, 
                                                               LEFT_ACRONYM, RIGHT_ACRONYM, 
                                                               LEFT_EMBEDDING, RIGHT_EMBEDDING
                                                               ], n_features=n_features)
    with open(output_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)
    click.echo(f"[INFO] Feature names saved to {output_dir / 'feature_names.json'}")

    # Save sample for inspection
    click.echo(f"[INFO] Creating sample for inspection...")
    if sample_rows_accum:
        X_sample_full = np.vstack(sample_rows_accum)
        # Use the first batch's sampled feature indices for column names
        sample_feat_idx = sample_feats_accum[0] if sample_feats_accum else np.arange(sample_feat_cap)
        sample_cols = [feature_names[j] if j < len(feature_names) else f"f{j}" for j in sample_feat_idx]
        sample_df = pd.DataFrame(X_sample_full, columns=sample_cols)
        try:
            sample_df.to_parquet(output_dir / "features_sample.parquet", index=False)
            click.echo(f"[INFO] Sample features saved to {output_dir / 'features_sample.parquet'}")
        except Exception as e:
            warnings.warn(f"Parquet write failed ({e}); falling back to CSV.gz")
            click.echo(f"[WARN] Parquet write failed, saving sample to CSV.gz instead.")
            sample_df.to_csv(output_dir / "features_sample.csv.gz", index=False, compression="gzip")
    else:
        click.echo(f"[WARN] No sample rows accumulated; skipping sample save.")

    # Plots (use sample for dense, first batch for sparse)
    plot_path = output_dir / "feature_distributions.png"
    click.echo(f"[INFO] Generating feature distribution plots...")
    total_n_rows = sum(batch_row_counts)
    total_nnz = sum(batch_nnz_counts)
    density = total_nnz / (total_n_rows * max(1, n_features)) if total_n_rows and n_features else 0.0
    if density < 0.2:
        # Use first batch for sparse plot
        X_sparse_plot = sp.load_npz(batch_feature_files[0])
        _plot_sparse_density(X_sparse_plot, feature_names, plot_path, max_features=max_plot_features, seed=seed)
        click.echo(f"[INFO] Sparse density plot saved to {plot_path}")
    else:
        # Use sample for dense plot (sample from all batches)
        if sample_rows_accum:
            _plot_dense_hists(X_sample_full, sample_cols, plot_path, max_features=max_plot_features, seed=seed)
            click.echo(f"[INFO] Dense histogram plot saved to {plot_path}")
        else:
            click.echo(f"[WARN] No sample available for dense plot.")

    # Summary
    click.echo(f"[INFO] Writing summary metadata...")
    summary = {
        "n_rows": int(total_n_rows),
        "n_features": int(n_features),
        "nnz": int(total_nnz),
        "density": float(density),
        "batch_size": int(batch_size),
        "sample_rows": int(X_sample_full.shape[0]) if sample_rows_accum else 0,
        "sample_features": int(len(sample_cols)) if sample_rows_accum else 0,
        "outputs": {
            "features_batches": batch_feature_files,
            "feature_names": str(output_dir / "feature_names.json"),
            "row_index": str(row_index_gz),
            "features_sample": str(output_dir / "features_sample.parquet"),
            "plots": str(plot_path),
        }
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    click.echo(f"[INFO] Summary saved to {output_dir / 'summary.json'}")

    click.echo(f"[SUCCESS] Saved features (by batch), names, row_index, sample, plots, and summary → {output_dir}")


if __name__ == "__main__":
    main()
