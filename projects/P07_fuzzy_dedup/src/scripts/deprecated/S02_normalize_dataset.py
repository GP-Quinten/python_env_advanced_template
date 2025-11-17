#!/usr/bin/env python3
# file: normalize_registries.py

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Set

import click
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---- Config knobs (same spirit as your notebook) ----
DEFAULT_DROP_TERMS: Set[str] = {
    "registry", "register", "database", "program", "project", "initiative",
    "study", "system", "network", "society", "group", "center", "centre",
    "institute", "institution", "unit", "department", "service",
    "clinic", "hospital", "university", "research", "data",
}

DEFAULT_STOPWORDS: Set[str] = {
    "the", "a", "an", "of", "for", "in", "on", "and", "with", "to", "from", "by",
}

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

_ROMAN_MAP = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9,
    "x": 10, "xi": 11, "xii": 12, "xiii": 13, "xiv": 14, "xv": 15, "xvi": 16, "xvii": 17,
    "xviii": 18, "xix": 19, "xx": 20
}
_ROMAN_RE = re.compile(r"\b(" + "|".join(sorted(_ROMAN_MAP.keys(), key=len, reverse=True)) + r")\b", re.IGNORECASE)

_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^0-9a-z ]+")
_MULTISPACE_RE = re.compile(r"\s{2,}")

def _strip_accents(text: str) -> str:
    nkfd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nkfd if not unicodedata.combining(ch))

def _expand_and_canonicalize(tokens: List[str]) -> List[str]:
    out = []
    for t in tokens:
        t0 = CANONICAL_SPELLINGS.get(t, t)
        t1 = EXPANSIONS.get(t0, t0)
        out.append(t1)
    return out

def _roman_to_arabic(text: str) -> str:
    def repl(m):
        key = m.group(1).lower()
        return str(_ROMAN_MAP.get(key, key))
    return _ROMAN_RE.sub(repl, text)

def normalize_registry_names(
    names: Iterable[str],
    *,
    drop_terms: Set[str] = DEFAULT_DROP_TERMS,
    stopwords: Set[str] = DEFAULT_STOPWORDS,
    remove_drop_terms: bool = True,
    remove_stopwords: bool = True,
    sort_tokens: bool = True,
) -> List[str]:
    """
    Normalize registry names for deduplication.
    """
    normalized = []
    for s in names:
        if s is None:
            normalized.append("")
            continue

        # 1) Unicode normalize, accent strip, lowercase
        s = unicodedata.normalize("NFKC", str(s))
        s = _strip_accents(s).lower()

        # 2) Normalize joiners/punct
        s = s.replace("&", " and ").replace("/", " ")
        s = s.replace("-", " ").replace("_", " ")

        # 3) Roman numerals -> arabic
        s = _roman_to_arabic(s)

        # 4) Remove non-alnum (keep spaces, digits)
        s = _NON_ALNUM_RE.sub(" ", s)

        # 5) Collapse whitespace
        s = _WS_RE.sub(" ", s).strip()
        if not s:
            normalized.append("")
            continue

        # 6) Token-level canonicalization/expansion
        tokens = s.split(" ")
        tokens = _expand_and_canonicalize(tokens)

        # 7) Optional removals
        if remove_drop_terms:
            tokens = [t for t in tokens if t not in drop_terms]
        if remove_stopwords:
            tokens = [t for t in tokens if t not in stopwords]

        # 8) Clean empties
        tokens = [t for t in tokens if t]

        # 9) Optional sort to reduce order variance
        if sort_tokens:
            tokens = sorted(tokens)

        s_out = " ".join(tokens)
        s_out = _MULTISPACE_RE.sub(" ", s_out).strip()
        normalized.append(s_out)

    return normalized

def _read_json_any(path: Path) -> pd.DataFrame:
    """
    Reads JSON or JSONL into a DataFrame, trying sensible defaults.
    """
    # Heuristics: jsonl by extension or when lines look JSONL-like
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    # Try standard JSON (array or object)
    try:
        return pd.read_json(path)
    except ValueError:
        # Fallback: try as JSONL
        return pd.read_json(path, lines=True)

@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--dataset_json", "dataset_json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Path to input JSON/JSONL file containing a 'registry_name' column/field.")
@click.option("--output_dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Directory to write outputs.")
@click.option("--target_column", default="registry_name", show_default=True,
              help="Column name that contains the raw registry names.")
@click.option("--output_filename", default="normalized_registries.json", show_default=True,
              help="Output JSON filename.")
def main(dataset_json: Path, output_dir: Path, target_column: str, output_filename: str):
    """
    Load a JSON/JSONL dataset, normalize registry names, and save a new DataFrame.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading: {dataset_json}")
    df = _read_json_any(dataset_json)

    if target_column not in df.columns:
        raise click.ClickException(
            f"Column '{target_column}' not found in data. Available columns: {list(df.columns)}"
        )

    log.info("Normalizing registry names…")
    df["norm__registry_name"] = normalize_registry_names(
        df[target_column].fillna(""),
        drop_terms=DEFAULT_DROP_TERMS,
        stopwords=DEFAULT_STOPWORDS,
        remove_drop_terms=True,
        remove_stopwords=True,
        sort_tokens=True,
    )

    # Save as JSON with indent=3
    out_json = output_dir / output_filename
    df.to_json(out_json, orient="records", indent=3, force_ascii=False)
    log.info(f"Wrote JSON: {out_json}")

    log.info("Done.")

if __name__ == "__main__":
    main()
    log.info("Done.")
