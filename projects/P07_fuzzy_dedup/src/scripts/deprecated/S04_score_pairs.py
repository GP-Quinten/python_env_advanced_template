#!/usr/bin/env python3
# file: S04_score_pairs.py

# This script calculates similarity scores between candidate pairs of registry names
# It takes pairs identified in the blocking step and computes different similarity metrics
# to help determine which pairs are likely to be actual duplicates

import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import click
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ----------------------------
# Similarity functions
# ----------------------------

def jaro_winkler(s1: str, s2: str, p: float = 0.1, max_l: int = 4) -> float:
    """
    Calculate Jaro-Winkler similarity between two strings.
    
    This metric favors strings that match from the beginning (common prefixes).
    Good for detecting typos and small edits in names.
    
    Args:
        s1, s2: Strings to compare
        p: Prefix scaling factor (how much to favor matching prefixes)
        max_l: Maximum prefix length to consider
        
    Returns:
        Similarity score between 0 (completely different) and 1 (identical)
    """
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    match_distance = max(len1, len2) // 2 - 1
    s1_flags = [False] * len1
    s2_flags = [False] * len2
    matches = 0
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if not s2_flags[j] and s1[i] == s2[j]:
                s1_flags[i] = s2_flags[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0
    s1_m = [s1[i] for i in range(len1) if s1_flags[i]]
    s2_m = [s2[j] for j in range(len2) if s2_flags[j]]
    transpositions = sum(c1 != c2 for c1, c2 in zip(s1_m, s2_m)) // 2
    jaro = ((matches / len1) + (matches / len2) + ((matches - transpositions) / matches)) / 3.0
    prefix = 0
    for c1, c2 in zip(s1, s2):
        if c1 == c2:
            prefix += 1
            if prefix == max_l:
                break
        else:
            break
    return jaro + prefix * p * (1.0 - jaro)

def _levenshtein_distance(a: str, b: str) -> int:
    """
    Calculate the Levenshtein (edit) distance between two strings.
    
    This is the minimum number of single-character insertions, deletions, 
    and substitutions required to change one string into another.
    
    Args:
        a, b: Strings to compare
        
    Returns:
        Edit distance as an integer
    """
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    if la > lb:
        a, b, la, lb = b, a, lb, la
    prev = list(range(la + 1))
    for j in range(1, lb + 1):
        curr = [j] + [0] * la
        bj = b[j - 1]
        for i in range(1, la + 1):
            ai = a[i - 1]
            cost = 0 if ai == bj else 1
            curr[i] = min(prev[i] + 1, curr[i - 1] + 1, prev[i - 1] + cost)
        prev = curr
    return prev[la]

def levenshtein_norm_sim(a: str, b: str) -> float:
    """
    Normalized Levenshtein similarity between two strings.
    
    Converts the edit distance to a similarity score by normalizing by the maximum length
    and converting to a similarity (1 - normalized_distance).
    
    Args:
        a, b: Strings to compare
        
    Returns:
        Similarity score between 0 (completely different) and 1 (identical)
    """
    if not a and not b:
        return 1.0
    d = _levenshtein_distance(a, b)
    m = max(len(a), len(b))
    return 1.0 - (d / m if m else 0.0)

def jaccard_tokens(s1: str, s2: str) -> float:
    """
    Jaccard similarity between tokenized strings (words).
    
    Measures the overlap between the sets of words in each string.
    Jaccard = |intersection| / |union|
    
    Args:
        s1, s2: Strings to tokenize and compare
        
    Returns:
        Similarity score between 0 (no words in common) and 1 (identical word sets)
    """
    A, B = set(s1.split()), set(s2.split())
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0

# --- Soft TF-IDF ---
def compute_idf(corpus: Iterable[str]) -> Dict[str, float]:
    """
    Compute Inverse Document Frequency (IDF) weights for all tokens in corpus.
    
    IDF measures how important/unique a word is across the entire corpus.
    Words that appear in few documents get higher IDF values.
    
    Args:
        corpus: Collection of strings to analyze
        
    Returns:
        Dictionary mapping each token to its IDF weight
    """
    from collections import Counter
    docs = [set(s.split()) for s in corpus]
    N = len(docs)
    df = Counter()
    for toks in docs:
        df.update(toks)
    return {t: math.log((N + 1) / (df[t] + 1)) + 1.0 for t in df}

def _tfidf_weights(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    """
    Compute TF-IDF weights for tokens in a document.
    
    TF-IDF combines the frequency of a term in a document (TF) with
    its importance across all documents (IDF).
    
    Args:
        tokens: List of tokens in a document
        idf: IDF values for all tokens in corpus
        
    Returns:
        Dictionary mapping each token to its TF-IDF weight
    """
    from collections import Counter
    tf = Counter(tokens)
    return {t: (1.0 + math.log(c)) * idf.get(t, 0.0) for t, c in tf.items()}

def soft_tfidf(
    s1: str,
    s2: str,
    idf: Dict[str, float],
    *,
    sim_func=jaro_winkler,
    tau: float = 0.9
) -> float:
    """
    Soft TF-IDF similarity between two strings.
    
    This is an advanced hybrid metric that combines:
    1. TF-IDF weighting (importance of words)
    2. Soft matching between tokens (using sim_func)
    
    It handles scenarios where tokens are similar but not exact matches,
    making it robust for entity names with spelling variations.
    
    Args:
        s1, s2: Strings to compare
        idf: IDF values for all tokens
        sim_func: Function to compare individual tokens
        tau: Threshold for token similarity (only consider tokens with similarity ≥ tau)
        
    Returns:
        Similarity score between 0 (completely different) and 1 (identical)
    """
    toks1, toks2 = s1.split(), s2.split()
    if not toks1 and not toks2:
        return 1.0
    if not toks1 or not toks2:
        return 0.0
    w1 = _tfidf_weights(toks1, idf)
    w2 = _tfidf_weights(toks2, idf)
    candidates: List[Tuple[float, str, str]] = []
    for t1 in w1:
        for t2 in w2:
            s = sim_func(t1, t2)
            if s >= tau:
                candidates.append((s, t1, t2))
    if not candidates:
        return 0.0
    candidates.sort(reverse=True)
    used1, used2 = set(), set()
    num = 0.0
    for s, t1, t2 in candidates:
        if t1 in used1 or t2 in used2:
            continue
        used1.add(t1); used2.add(t2)
        num += w1[t1] * w2[t2] * s
    denom = math.sqrt(sum(v*v for v in w1.values())) * math.sqrt(sum(v*v for v in w2.values()))
    return (num / denom) if denom > 0 else 0.0

# ----------------------------
# IO helper
# ----------------------------
def _read_json_any(path: Path) -> pd.DataFrame:
    """
    Flexibly read a JSON file, handling both regular JSON and JSON Lines formats.
    
    Args:
        path: Path to JSON file
        
    Returns:
        DataFrame containing the data
    """
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    try:
        return pd.read_json(path)
    except ValueError:
        return pd.read_json(path, lines=True)

# ----------------------------
# CLI
# ----------------------------
@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--dataset_json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Path to dataset JSON/JSONL used to create pairs (must contain target_column).")
@click.option("--pairs_json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Path to candidate pairs JSON (fields: idx_i, idx_j).")
@click.option("--output_dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Directory to write outputs.")
@click.option("--output_filename", default="scored_pairs.json", show_default=True,
              help="Output JSON filename for similarity scores.")
@click.option("--target_column", default="norm__registry_name", show_default=True,
              help="Text column to score (should match what you blocked on).")
@click.option("--jw_prefix_weight", default=0.1, show_default=True, help="Jaro–Winkler prefix weight p.")
@click.option("--jw_prefix_max", default=4, show_default=True, help="Jaro–Winkler max prefix length.")
@click.option("--soft_tfidf_tau", default=0.9, show_default=True, help="Soft TF-IDF token similarity threshold.")
def main(
    dataset_json: Path,
    pairs_json: Path,
    output_dir: Path,
    output_filename: str,
    target_column: str,
    jw_prefix_weight: float,
    jw_prefix_max: int,
    soft_tfidf_tau: float,
):
    """Compute JW, Levenshtein, Jaccard, Soft TF-IDF similarity scores for candidate pairs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load the main dataset containing the registry names
    log.info(f"Loading dataset: {dataset_json}")
    df = _read_json_any(dataset_json)
    if target_column not in df.columns:
        raise click.ClickException(f"Column '{target_column}' not found in dataset.")
    names = df[target_column].fillna("").astype(str).tolist()

    # Step 2: Load the candidate pairs generated by the blocking step
    log.info(f"Loading pairs: {pairs_json}")
    pairs_df = _read_json_any(pairs_json)
    if not {"idx_i", "idx_j"}.issubset(pairs_df.columns):
        raise click.ClickException("Pairs file must contain 'idx_i' and 'idx_j'.")
    pairs = list(pairs_df[["idx_i", "idx_j"]].itertuples(index=False, name=None))

    # Step 3: Precompute IDF values for all tokens in the dataset (needed for Soft TF-IDF)
    log.info("Computing IDF for Soft TF-IDF...")
    idf = compute_idf(names)

    # Step 4: Calculate all similarity scores for each candidate pair
    rows = []
    for i, j in tqdm(pairs, desc="Scoring pairs"):
        s1 = names[i]
        s2 = names[j]
        
        # Calculate four different similarity metrics
        jw = jaro_winkler(s1, s2, p=jw_prefix_weight, max_l=jw_prefix_max)
        lev = levenshtein_norm_sim(s1, s2)
        jac = jaccard_tokens(s1, s2)
        stf = soft_tfidf(s1, s2, idf,
                         sim_func=lambda a, b: jaro_winkler(a, b, p=jw_prefix_weight, max_l=jw_prefix_max),
                         tau=soft_tfidf_tau)
        
        # Store results along with the original names for reference
        rows.append({
            "idx_i": i, "idx_j": j,
            "left_name": s1, "right_name": s2,
            "jw": jw, "lev_sim": lev, "jaccard_tok": jac, "soft_tfidf": stf
        })

    # Step 5: Save the scored pairs to a JSON file
    out_path = output_dir / output_filename
    pd.DataFrame(rows).to_json(out_path, orient="records", indent=3, force_ascii=False)
    log.info(f"Wrote scored pairs: {out_path}")

if __name__ == "__main__":
    main()
