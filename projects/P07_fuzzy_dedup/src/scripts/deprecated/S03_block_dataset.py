#!/usr/bin/env python3
# file: S03_block_dataset.py

import logging
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Set
from collections import defaultdict
from itertools import combinations

import click
import pandas as pd
import mmh3  # MurmurHash3
from tqdm import tqdm  # Import tqdm for progress bars

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Large prime for modular hashing (2^61 - 1 is prime)
# Using a large prime helps ensure good distribution of hash values
_P = (1 << 61) - 1


def _char_ngrams(s: str, n: int = 3) -> Sequence[str]:
    """
    Generate character n-grams from a string with start/end padding.

    Args:
        s: Input string
        n: Size of n-grams (default: 3 for trigrams)

    Returns:
        List of n-grams with padding

    Example:
        For s="abc" with n=3:
        - Adds "^^" prefix and "$$" suffix
        - Returns ["^^a", "^ab", "abc", "bc$", "c$$"]
    """
    if not s:
        return []
    pad = "^" * (n - 1)  # Start padding
    tail = "$" * (n - 1)  # End padding
    s2 = f"{pad}{s}{tail}"
    return [s2[i:i + n] for i in range(len(s2) - n + 1)]


def _shingle_to_int(g: str, seed: int) -> int:
    """
    Convert a shingle (n-gram) to a 64-bit integer using MurmurHash3.

    Args:
        g: Input shingle (n-gram string)
        seed: Hash seed for deterministic results

    Returns:
        64-bit unsigned integer hash of the shingle
    """
    h64_lo, _h64_hi = mmh3.hash64(g, seed=seed, signed=False)
    return h64_lo  # already unsigned (0..2^64-1)


def _make_hash_funcs(num_perm: int, seed: int) -> List[tuple]:
    """
    Generate hash function parameters for MinHash.

    Creates parameters for hash functions of the form: h(x) = (a*x + b) mod P
    where P is a large prime.

    Args:
        num_perm: Number of hash permutations
        seed: Random seed for reproducibility

    Returns:
        List of (a,b) tuples for hash functions
    """
    import random
    rng = random.Random(seed)
    funcs = []
    for _ in range(num_perm):
        a = rng.randrange(1, _P - 1)  # a must be non-zero
        b = rng.randrange(0, _P - 1)
        funcs.append((a, b))
    return funcs


def _band_hash(band_idx: int, band_values: Sequence[int], seed: int) -> str:
    """
    Hash a band of MinHash values to a bucket ID.

    Args:
        band_idx: Index of the band
        band_values: Sequence of MinHash values in this band
        seed: Hash seed for deterministic results

    Returns:
        String representation of the band hash (16 hex digits)
    """
    # Serialize band index + values + seed into bytes deterministically
    data = bytearray()
    data += band_idx.to_bytes(4, "little", signed=False)
    for v in band_values:
        data += v.to_bytes(8, "little", signed=False)
    data += seed.to_bytes(8, "little", signed=False)

    # 128-bit MurmurHash3; get a positive int and format as hex
    h128 = mmh3.hash128(bytes(data), seed=seed, signed=False)
    return f"{h128:032x}"[:16]  # keep first 16 hex chars for brevity


def lsh_blocks(
    names: Iterable[str],
    *,
    ngram: int = 3,
    num_perm: int = 128,
    bands: int = 32,
    seed: int = 13
) -> List[List[str]]:
    """
    Compute LSH block IDs for each string using MinHash over character n-grams.

    This function implements Locality-Sensitive Hashing with MinHash to group
    similar strings together. Similar strings will likely share at least one block ID.

    Args:
        names: Iterable of strings to process
        ngram: Size of character n-grams
        num_perm: Number of MinHash permutations (signature length)
        bands: Number of LSH bands to use
        seed: Random seed for reproducibility

    Returns:
        List of lists, where each inner list contains block IDs for a string

    Algorithm Overview:
        1. Convert each string to character n-grams
        2. Hash n-grams to integers using MurmurHash3
        3. Apply MinHash to create a compact signature for each string
        4. Split each signature into bands and hash bands to buckets
        5. Return bucket IDs as block IDs for each string

    LSH Parameters:
        - The (num_perm, bands) combination determines the LSH threshold.
        - Threshold t ≈ (1/bands)^(1/rows) where rows = num_perm/bands
        - Strings with similarity > t are likely to share at least one bucket
        - Lower threshold (more bands) = higher recall, lower precision
        - Higher threshold (fewer bands) = lower recall, higher precision
    """
    # Validate input parameters
    if num_perm <= 0:
        raise ValueError("num_perm must be > 0")
    if bands <= 0:
        raise ValueError("bands must be > 0")

    # Calculate rows per band (may have uneven last band)
    rows = math.ceil(num_perm / bands)  # allow uneven last band

    # Generate hash function parameters (a,b) for MinHash
    funcs = _make_hash_funcs(num_perm, seed)

    # Convert input to list if it's a generator to enable progress tracking
    names_list = list(names)
    log.info(f"Processing {len(names_list)} strings...")

    out_blocks: List[List[str]] = []
    for s in tqdm(names_list, desc="Computing LSH blocks"):
        s = (s or "").strip()
        # Generate character n-grams from the string
        grams = _char_ngrams(s, n=ngram)
        if not grams:
            # Handle empty strings by assigning empty block list
            out_blocks.append([])
            continue

        # Convert n-grams to integer hashes for more efficient processing
        shingles = {_shingle_to_int(g, seed) for g in grams}

        # MinHash signature generation:
        # For each hash function (a*x + b) % P, find the minimum hash value
        # This approximates Jaccard similarity between sets of n-grams
        sig = []
        for (a, b) in funcs:
            m = min(((a * x + b) % _P) for x in shingles)
            sig.append(m)

        # LSH banding strategy:
        # Split signature into bands and hash each band to a bucket
        # Similar items are likely to hash to the same bucket in at least one band
        blocks = []
        for b_idx in range(bands):
            start = b_idx * rows
            if start >= num_perm:
                break
            end = min(start + rows, num_perm)
            band_vals = sig[start:end]
            # Hash the band values to a bucket ID
            bucket = _band_hash(b_idx, band_vals, seed)
            # Format block ID with metadata for traceability
            # mh = MinHash, {ngram}g = n-gram size, {num_perm}x{bands} = LSH parameters
            blocks.append(f"mh{ngram}g:{num_perm}x{bands}:{b_idx}:{bucket}")

        out_blocks.append(blocks)

    return out_blocks


def _read_json_any(path: Path) -> pd.DataFrame:
    """
    Flexibly read a JSON file regardless of format (JSON or JSONL).

    Args:
        path: Path to JSON/JSONL file

    Returns:
        Pandas DataFrame with the loaded data
    """
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    try:
        # First try regular JSON format
        return pd.read_json(path)
    except ValueError:
        # Fall back to JSONL format
        return pd.read_json(path, lines=True)


def make_candidate_pairs_via_lsh(
    df: pd.DataFrame,
    col_norm: str = "norm__registry_name",
    *,
    # lsh params (must match/plug into your lsh_blocks)
    ngram: int = 3,
    num_perm: int = 128,
    bands: int = 32,
    seed: int = 13,
    # block pruning
    min_block_size: int = 2,
    max_block_size: int = 200
) -> List[Tuple[int, int]]:
    """
    Use LSH block IDs to generate unique candidate pairs (i < j).
    Prunes very small and very large blocks to avoid noise and blowups.
    """
    # Use precomputed blocks if available to avoid redundant computation
    if "lsh_blocks" in df.columns:
        all_blocks = df["lsh_blocks"].tolist()
    else:
        names = df[col_norm].fillna("").astype(str).tolist()
        all_blocks = lsh_blocks(
            names,
            ngram=ngram,
            num_perm=num_perm,
            bands=bands,
            seed=seed,
        )

    block2ids = defaultdict(list)
    for idx, blocks in enumerate(all_blocks):
        for b in blocks:
            block2ids[b].append(idx)

    # prune and generate pairs
    pairs: Set[Tuple[int, int]] = set()
    for bid, ids in block2ids.items():
        L = len(ids)
        if L < min_block_size or L > max_block_size:
            continue
        ids_sorted = sorted(ids)
        for i, j in combinations(ids_sorted, 2):
            pairs.add((i, j))

    return sorted(pairs)


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--dataset_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to input JSON/JSONL file.",
)
@click.option(
    "--output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory to write outputs.",
)
@click.option(
    "--target_column",
    default="norm__registry_name",
    show_default=True,
    help="Column to block on (e.g., a normalized name column).",
)
@click.option(
    "--output_filename",
    default="blocked_registries.json",
    show_default=True,
    help="Output JSON filename.",
)
@click.option(
    "--pairs_filename",
    default="candidate_pairs.json",
    show_default=True,
    help="Candidate pairs JSON filename.",
)
@click.option("--ngram", default=3, show_default=True, help="Character n-gram size.")
@click.option("--num_perm", default=128, show_default=True, help="MinHash signature length.")
@click.option("--bands", default=32, show_default=True, help="Number of LSH bands.")
@click.option("--seed", default=13, show_default=True, help="Seed for Murmur and permutations.")
@click.option("--min_block_size", default=2, show_default=True, help="Minimum block size to consider.")
@click.option("--max_block_size", default=200, show_default=True, help="Maximum block size to consider.")
def main(
    dataset_json: Path,
    output_dir: Path,
    target_column: str,
    output_filename: str,
    pairs_filename: str,
    ngram: int,
    num_perm: int,
    bands: int,
    seed: int,
    min_block_size: int,
    max_block_size: int,
):
    """
    Load JSON/JSONL, compute MurmurHash3-based LSH blocking IDs for the chosen column,
    and save the augmented DataFrame as JSON.

    This script implements the LSH (Locality-Sensitive Hashing) algorithm to find
    potential duplicates in a dataset. It works by:
    1. Loading data from a JSON/JSONL file
    2. Computing LSH blocks based on character n-grams and MinHash
    3. Saving the data with block IDs for later processing
    4. Generating and saving candidate pairs for comparison
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading: {dataset_json}")
    df = _read_json_any(dataset_json)
    log.info(f"Loaded {len(df)} records")

    if target_column not in df.columns:
        raise click.ClickException(
            f"Column '{target_column}' not found in data. "
            f"Available columns: {list(df.columns)}"
        )

    log.info(
        f"Computing LSH blocks on '{target_column}' "
        f"(ngram={ngram}, num_perm={num_perm}, bands={bands}, seed={seed})…"
    )

    # Fill NAs and convert to strings
    values = df[target_column].fillna("").astype(str).tolist()

    # Compute LSH blocks with progress tracking
    df["lsh_blocks"] = lsh_blocks(
        values,
        ngram=ngram,
        num_perm=num_perm,
        bands=bands,
        seed=seed,
    )

    # Count non-empty block lists
    non_empty_blocks = sum(1 for blocks in df["lsh_blocks"] if blocks)
    log.info(f"{non_empty_blocks} out of {len(df)} records have block assignments")

    # Write output
    out_json = output_dir / output_filename
    log.info(f"Writing output to: {out_json}")
    df.to_json(out_json, orient="records", indent=3, force_ascii=False)
    log.info(f"Wrote JSON: {out_json}")
    
    # Generate and save candidate pairs
    log.info(f"Generating candidate pairs (min_block_size={min_block_size}, max_block_size={max_block_size})...")
    pairs = make_candidate_pairs_via_lsh(
        df, 
        col_norm=target_column,
        ngram=ngram,
        num_perm=num_perm,
        bands=bands,
        seed=seed,
        min_block_size=min_block_size,
        max_block_size=max_block_size
    )
    
    log.info(f"Found {len(pairs)} candidate pairs")
    pairs_json = output_dir / pairs_filename
    log.info(f"Writing candidate pairs to: {pairs_json}")
    pairs_df = pd.DataFrame(pairs, columns=["idx_i", "idx_j"])
    pairs_df.to_json(pairs_json, orient="records", indent=3, force_ascii=False)
    log.info(f"Wrote pairs JSON: {pairs_json}")
    
    log.info("Done.")


if __name__ == "__main__":
    main()
