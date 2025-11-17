#!/usr/bin/env python3
# File: src/scripts/S07_make_features.py
#
# CLI to turn a labeled pair dataset (left_name, right_name, label)
# into numeric features and train/test splits saved to disk.

import json
from pathlib import Path
import click
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from P07_fuzzy_dedup.pair_features import (
    PairStringFeatures,
    DEFAULT_STOPWORDS,
)
# Added for normalization
import re
import unicodedata
from typing import List, Iterable, Set, Dict

# ---- Normalization Helpers (same as in S02_normalize_dataset.py) ----
DEFAULT_DROP_TERMS: Set[str] = {
    "registry", "register", "database", "program", "project", "initiative",
    "study", "system", "network", "society", "group", "center", "centre",
    "institute", "institution", "unit", "department", "service",
    "clinic", "hospital", "university", "research", "data",
}

_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^0-9a-z ]+")
_MULTISPACE_RE = re.compile(r"\s{2,}")

_ROMAN_MAP = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9,
    "x": 10, "xi": 11, "xii": 12, "xiii": 13, "xiv": 14, "xv": 15, "xvi": 16, "xvii": 17,
    "xviii": 18, "xix": 19, "xx": 20
}
_ROMAN_RE = re.compile(r"\b(" + "|".join(sorted(_ROMAN_MAP.keys(), key=len, reverse=True)) + r")\b", re.IGNORECASE)

def _strip_accents(text: str) -> str:
    nkfd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nkfd if not unicodedata.combining(ch))

def _roman_to_arabic(text: str) -> str:
    def repl(m):
        key = m.group(1).lower()
        return str(_ROMAN_MAP.get(key, key))
    return _ROMAN_RE.sub(repl, text)

CANONICAL_SPELLINGS: Dict[str, str] = {
    "centre": "center",
    "centres": "centers",
    "programme": "program",
    "tumour": "tumor",
    "tumours": "tumors",
    "paediatric": "pediatric",
    "haematology": "hematology",
    "haemorrhage": "hemorrhage",
    "oedema": "edema",
    "oesophagus": "esophagus",
    "oesophageal": "esophageal",
    "organisation": "organization",
    "organisations": "organizations",
}

EXPANSIONS: Dict[str, str] = {
    "reg": "registry",
    "reg.": "registry",
    "db": "database",
    "prog": "program",
    "dept": "department",
    "ctr": "center",
    "int'l": "international",
    "intl": "international",
    "natl": "national",
    "assoc": "association",
    "soc": "society",
    "univ": "university",
    "hosp": "hospital",
    "med": "medical",
}

def _expand_and_canonicalize(tokens: List[str]) -> List[str]:
    out = []
    for t in tokens:
        t0 = CANONICAL_SPELLINGS.get(t, t)
        t1 = EXPANSIONS.get(t0, t0)
        out.append(t1)
    return out

def normalize_registry_names(
    names: Iterable[str],
    *,
    drop_terms: Set[str] = DEFAULT_DROP_TERMS,
    stopwords: Set[str] = DEFAULT_STOPWORDS,
    remove_drop_terms: bool = True,
    remove_stopwords: bool = True,
    sort_tokens: bool = True,
) -> List[str]:
    normalized = []
    for s in names:
        if s is None:
            normalized.append("")
            continue
        s = unicodedata.normalize("NFKC", str(s))
        s = _strip_accents(s).lower()
        s = s.replace("&", " and ").replace("/", " ")
        s = s.replace("-", " ").replace("_", " ")
        s = _roman_to_arabic(s)
        s = _NON_ALNUM_RE.sub(" ", s)
        s = _WS_RE.sub(" ", s).strip()
        if not s:
            normalized.append("")
            continue
        tokens = s.split(" ")
        tokens = _expand_and_canonicalize(tokens)
        if remove_drop_terms:
            tokens = [t for t in tokens if t not in DEFAULT_DROP_TERMS]
        if remove_stopwords:
            tokens = [t for t in tokens if t not in DEFAULT_STOPWORDS]
        tokens = [t for t in tokens if t]
        if sort_tokens:
            tokens = sorted(tokens)
        s_out = " ".join(tokens)
        s_out = _MULTISPACE_RE.sub(" ", s_out).strip()
        normalized.append(s_out)
    return normalized

def _read_any_table(path: Path) -> pd.DataFrame:
    """
    Read a table from various formats (CSV, TSV, JSON, JSONL).
    """
    p = Path(path)
    if p.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.read_json(p, lines=True)
    if p.suffix.lower() == ".json":
        return pd.read_json(p)
    if p.suffix.lower() in {".csv", ".tsv"}:
        sep = "," if p.suffix.lower() == ".csv" else "\t"
        return pd.read_csv(p, sep=sep)
    # Fallback
    return pd.read_json(p)

@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--dataset_json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Path to labeled pairs table (CSV/TSV/JSON/JSONL) with left/right/label columns.")
@click.option("--left_col", default="left_name", show_default=True)
@click.option("--right_col", default="right_name", show_default=True)
@click.option("--label_col", default="label", show_default=True)
@click.option("--test_size", default=0.2, show_default=True, type=float)
@click.option("--random_state", default=42, show_default=True, type=int)
@click.option("--output_dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--jw_prefix_weight", default=0.1, show_default=True, type=float)
@click.option("--jw_prefix_max", default=4, show_default=True, type=int)
@click.option("--save_debug_csv", is_flag=True, help="Also write a CSV with features + label for inspection.")
def main(
    dataset_json: Path,
    left_col: str,
    right_col: str,
    label_col: str,
    test_size: float,
    random_state: int,
    output_dir: Path,
    jw_prefix_weight: float,
    jw_prefix_max: int,
    save_debug_csv: bool,
):
    """
    Main CLI entrypoint for feature extraction and train/test split.
    """
    click.echo(f"[INFO] Creating output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"[INFO] Reading input table: {dataset_json}")
    df = _read_any_table(dataset_json)
    click.echo(f"[INFO] Loaded {len(df)} rows.")

    # Ensure a stable pair_id exists for each row
    if "pair_id" not in df.columns:
        df["pair_id"] = df.index

    # Check for required columns
    missing = [c for c in (left_col, right_col, label_col) if c not in df.columns]
    if missing:
        raise click.ClickException(f"Missing expected columns: {missing}")

    # ---- Apply normalization as in S02_normalize_dataset.py ----
    click.echo("[INFO] Normalizing text columns...")
    df[left_col] = normalize_registry_names(df[left_col].fillna(""), remove_drop_terms=True, remove_stopwords=True, sort_tokens=True)
    df[right_col] = normalize_registry_names(df[right_col].fillna(""), remove_drop_terms=True, remove_stopwords=True, sort_tokens=True)

    # Build and apply all features via PairStringFeatures
    click.echo(f"[INFO] Extracting features using PairStringFeatures")
    extractor = PairStringFeatures(
        left_col=left_col,
        right_col=right_col,
        stopwords=DEFAULT_STOPWORDS,
        jw_prefix_weight=jw_prefix_weight,
        jw_prefix_max=jw_prefix_max,
    )
    X = extractor.fit_transform(df[[left_col, right_col]])
    y = df[label_col].astype(int).values

    # Split indices together with features and labels
    click.echo(f"[INFO] Splitting data into train/test sets (test_size={test_size}, random_state={random_state})")
    # Split the array of pair_ids along with X and y.
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df["pair_id"].values, test_size=test_size, random_state=random_state, stratify=y
    )

    # Save arrays to disk
    click.echo(f"[INFO] Saving arrays to {output_dir}")
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_test.npy", y_test)
    np.save(output_dir / "train_idx.npy", idx_train)
    np.save(output_dir / "test_idx.npy", idx_test)

    # Save feature names for reference
    feat_names = extractor.get_feature_names_out().tolist()
    (output_dir / "feature_names.json").write_text(json.dumps(feat_names, indent=2))
    click.echo(f"[INFO] Feature names: {feat_names}")

    # Optionally save a full meta file with descriptive columns
    meta_cols = [ "pair_id", left_col, right_col, label_col ]
    try:
        import pyarrow
    except ImportError:
        raise click.ClickException("pyarrow is required for parquet support. Please install it via 'pip install pyarrow'.")
    (output_dir / "pairs_meta.parquet").write_bytes(
        df[meta_cols].to_parquet(index=False)
    )
    click.echo(f"[INFO] Saved pairs meta data with columns: {meta_cols}")

    # Always write debug CSV if requested (regardless of tqdm usage)
    if save_debug_csv:
        click.echo(f"[INFO] Writing debug CSV with features and labels: {output_dir / 'pairs_with_features.csv'}")
        debug_df = pd.DataFrame(X, columns=feat_names)
        debug_df[label_col] = y
        debug_df[left_col] = df[left_col].astype(str)
        debug_df[right_col] = df[right_col].astype(str)
        debug_df.to_csv(output_dir / "pairs_with_features.csv", index=False)

    # Save metadata about the dataset
    metadata = {
        "n_samples": len(df),
        "n_features": X.shape[1] if X.ndim > 1 else 1,
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
        "feature_names": feat_names,
        "label_distribution_train": {
            str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))
        },
        "label_distribution_test": {
            str(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))
        },
    }
    click.echo(f"[INFO] Writing metadata JSON: {output_dir / 'metadata.json'}")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=4))

    click.echo(f"[SUCCESS] Wrote: {output_dir}/X_train.npy, X_test.npy, y_train.npy, y_test.npy, metadata.json")
    click.echo(f"[SUCCESS] Features: {feat_names}")

if __name__ == "__main__":
    main()
