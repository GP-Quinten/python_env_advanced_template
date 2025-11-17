#!/usr/bin/env python3
"""
S20_1_find_bucket_params.py
---------------------------
Grid-search LSH parameters on an annotated (labeled) dataset to maximize recall
on positive pairs while minimizing the number of generated candidate pairs.

Input (annotated pairs) must have columns:
- left_object_id, right_object_id, label
- left_name, left_acronym, right_name, right_acronym (used to reconstruct registry)

Outputs:
- metrics.csv: per-configuration metrics
- best_params.json: best LSH params found
- summary.json: overall best metrics summary
"""

import itertools
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import click
import pandas as pd
import mmh3
from tqdm import tqdm
from functools import partial
from tqdm.contrib.concurrent import process_map
import matplotlib.pyplot as plt
import seaborn as sns

from P07_fuzzy_dedup.bucketer import LSHBucketer
from deduplication_model.normalization.utils import normalize_registry_names


# ---------------- Utilities ---------------- #

def unordered_pair(a: Any, b: Any) -> Tuple[str, str]:
    x, y = str(a), str(b)
    return (x, y) if x <= y else (y, x)

def candidate_pairs_set_from_records(pairs_out: Sequence[Dict[str, Any]]) -> Set[Tuple[str, str]]:
    return {unordered_pair(r["left_object_id"], r["right_object_id"]) for r in pairs_out}

def pair_hash_id(left_id, right_id, *, seed: int = 13) -> str:
    a, b = (str(left_id), str(right_id))
    if a > b:
        a, b = b, a
    h = mmh3.hash128(f"{a}||{b}", seed=seed, signed=False)
    return f"{h:032x}"

def build_registry_from_labeled_pairs(labeled_df: pd.DataFrame) -> pd.DataFrame:
    left = labeled_df[["left_object_id","left_name","left_acronym"]].rename(columns={
        "left_object_id":"object_id", "left_name":"registry_name", "left_acronym":"acronym"})
    right = labeled_df[["right_object_id","right_name","right_acronym"]].rename(columns={
        "right_object_id":"object_id", "right_name":"registry_name", "right_acronym":"acronym"})
    reg = pd.concat([left,right], ignore_index=True)
    reg["object_id"] = reg["object_id"].astype(str)
    return reg.groupby("object_id",as_index=False).agg({"registry_name":"first","acronym":"first"})

# Pass keys define which text variants are used when generating candidate pairs
# - raw_name: original registry_name
# - norm_name: normalized registry_name
# - acronym: acronym only
# - raw_full: "registry_name + acronym"
# - norm_full: normalized("registry_name + acronym")
PASS_KEYS = ("raw_name", "norm_name", "acronym", "raw_full", "norm_full")

def all_nonempty_pass_combos(keys=PASS_KEYS):
    """Return all non-empty combinations of pass keys as tuples, e.g. ('raw_name',), ('norm_name','acronym'), ..."""
    from itertools import combinations
    combos = []
    for r in range(1, len(keys) + 1):
        for c in combinations(keys, r):
            combos.append(c)
    return combos

def run_lsh_on_registry(df: pd.DataFrame, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    info = {row["object_id"]: {"name": row["registry_name"], "acronym": row["acronym"]}
            for _, row in df.iterrows()}
    object_ids = df["object_id"].astype(str).tolist()

    # collect unique matches across both passes
    records = {}

    # Only pass valid kwargs to LSHBucketer (avoid passing 'passes' etc.)
    bucketer_kwargs = {k: params[k] for k in ("ngram","num_perm","bands","seed","min_block_size","max_block_size") if k in params}

    def run_pass(texts: List[str]):
        bucketer = LSHBucketer(**bucketer_kwargs)
        docs = list(zip(object_ids, texts))
        bucketer.build(docs)
        for left_id, right_id in bucketer.pairs():
            ph = pair_hash_id(left_id, right_id, seed=params["seed"])
            if ph not in records:
                left, right = info[left_id], info[right_id]
                records[ph] = {
                    "pair_hash_id":      ph,
                    "left_name":         left["name"],
                    "left_acronym":      left["acronym"],
                    "right_name":        right["name"],
                    "right_acronym":     right["acronym"],
                    "left_object_id":    left_id,
                    "right_object_id":   right_id,
                }

    # Precompute all text variants once
    raw_names = df["registry_name"].fillna("").astype(str).tolist()
    norm_names = normalize_registry_names(raw_names)
    acronyms  = df["acronym"].fillna("").astype(str).tolist()
    raw_full  = [f"{n} {a}".strip() for n, a in zip(raw_names, acronyms)]
    norm_full = normalize_registry_names(raw_full)

    text_variants = {
        "raw_name":  raw_names,
        "norm_name": norm_names,
        "acronym":   acronyms,
        "raw_full":  raw_full,
        "norm_full": norm_full,
    }

    # Which passes to use this run? default = all
    selected_passes = params.get("passes", PASS_KEYS)
    for key in selected_passes:
        run_pass(text_variants[key])

    return list(records.values())

def compute_recall(labeled_df: pd.DataFrame, cand_set: Set[Tuple[str,str]], n_objects:int) -> Dict[str,Any]:
    pos = labeled_df[labeled_df["label"]==1].copy()
    pos["left_object_id"]=pos["left_object_id"].astype(str)
    pos["right_object_id"]=pos["right_object_id"].astype(str)
    gt = {unordered_pair(a,b) for a,b in zip(pos["left_object_id"],pos["right_object_id"])}
    n_pos, n_hit = len(gt), len(gt & cand_set)
    recall = n_hit/n_pos if n_pos else 0.0
    n_pairs = len(cand_set)
    total = n_objects*(n_objects-1)//2
    reduction_ratio = 1 - (n_pairs/total) if total>0 else 0.0
    return {"n_pos":n_pos,"n_hit":n_hit,"recall":recall,"n_pairs":n_pairs,"n_objects":n_objects,
            "total_possible_pairs":total,"reduction_ratio":reduction_ratio}


# ---------------- CLI ---------------- #

@click.command()
@click.option("--annotated_json", type=click.Path(exists=True,dir_okay=False,path_type=Path), required=True)
@click.option("--output_dir", type=click.Path(file_okay=False,path_type=Path), required=True)
@click.option("--seed", default=13, type=int)
def main(annotated_json: Path, output_dir: Path, seed: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_json(annotated_json)
    registry = build_registry_from_labeled_pairs(df)

    # ---------------- Param grids ---------------- #
    ngram_vals      = [3, 4]
    num_perm_vals   = [128]
    bands_vals      = [28, 32, 36, 40]
    min_block_vals  = [2]
    max_block_vals  = [50, 100, 200, 500, 1000]

    # Explicit pass combinations instead of generating all subsets
    pass_combinations = [
        ["raw_name"],                                   # raw only
        ["norm_name"],                                  # normalized only
        ["acronym"],                                    # acronym only
        ["raw_name", "acronym"],                        # raw + acronym
        ["norm_name", "acronym"],                       # normalized + acronym
        ["raw_name", "norm_name", "acronym"],           # three-way
        ["raw_name", "norm_name", "acronym",
         "raw_full", "norm_full"],                      # all passes
    ]

    grid = itertools.product(
        ngram_vals, num_perm_vals, bands_vals,
        min_block_vals, max_block_vals, pass_combinations
    )

    param_dicts = [
        dict(
            ngram=ngram,
            num_perm=num_perm,
            bands=bands,
            seed=seed,
            min_block_size=minb,
            max_block_size=maxb,
            passes=passes  # <- pass list chosen here
        )
        for ngram, num_perm, bands, minb, maxb, passes in grid
    ]
    evaluate = partial(evaluate_params, labeled_df=df, registry=registry)
    metrics = process_map(
        evaluate,
        param_dicts,
        max_workers=None,
        desc="Grid search",
        total=len(param_dicts)
    )
    best = max(metrics, key=lambda row: (row["recall"], -row["n_pairs"]))

    metrics_df = pd.DataFrame(metrics)
    # helpful readable label for plotting/filtering
    metrics_df["passes_label"] = metrics_df["passes"].apply(lambda p: "+".join(p))
    metrics_df.to_csv(output_dir/"metrics.csv", index=False)
    (output_dir/"best_params.json").write_text(json.dumps(best,indent=2))
    (output_dir/"summary.json").write_text(json.dumps({"best":best},indent=2))

    # Generate plots
    make_plots(metrics_df, output_dir)

    # Generate markdown report
    make_report(metrics_df, best, output_dir)

    click.echo(f"[SUCCESS] Best params: {best}")

    # --- report a random sample of up to 100 positive pairs missed by LSH ---
    best_params = {k: best[k] for k in ("ngram","num_perm","bands","seed","min_block_size","max_block_size","passes")}
    pairs = run_lsh_on_registry(registry, best_params)
    cand_set = candidate_pairs_set_from_records(pairs)
    # build ground‐truth positives
    pos = df[df["label"]==1].copy()
    pos["left_object_id"]  = pos["left_object_id"].astype(str)
    pos["right_object_id"] = pos["right_object_id"].astype(str)
    gt = {unordered_pair(a,b) for a,b in zip(pos["left_object_id"], pos["right_object_id"])}
    missed = list(gt - cand_set)
    sample = random.sample(missed, min(100, len(missed)))
    # write markdown table
    md = [
        "# Random Sample of Missed Positive Pairs\n",
        "The following positive pairs were not retrieved by LSH.\n\n",
        "| left_object_id | left_name | left_acronym | right_object_id | right_name | right_acronym |\n",
        "|---|---|---|---|---|---|\n"
    ]
    # map each ID to its registry name & acronym
    id2name    = registry.set_index("object_id")["registry_name"].to_dict()
    id2acronym = registry.set_index("object_id")["acronym"].to_dict()
    for l, r in sample:
        md.append(
            f"| {l} | {id2name.get(l, '')} | {id2acronym.get(l, '')} | "
            f"{r} | {id2name.get(r, '')} | {id2acronym.get(r, '')} |\n"
        )
    (output_dir/"missed_pairs.md").write_text("".join(md), encoding="utf-8")
    click.echo(f"[SUCCESS] Wrote missed pairs → {output_dir/'missed_pairs.md'}")

def evaluate_params(params: Dict[str, Any], labeled_df: pd.DataFrame, registry: pd.DataFrame) -> Dict[str, Any]:
    try:
        pairs = run_lsh_on_registry(registry, params)
        cand_set = candidate_pairs_set_from_records(pairs)
        m = compute_recall(labeled_df, cand_set, len(registry))
    except Exception as e:
        # fallback: still supply all metrics keys so reduction_ratio exists
        default = compute_recall(labeled_df, set(), len(registry))
        m = {**default, "error": str(e)}
    return {**params, **m}


# ---------------- Reporting & Visualization ---------------- #
def make_plots(metrics_df: pd.DataFrame, outdir: Path):
    # 1. Recall vs. #Pairs (log scale)
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=metrics_df, x="n_pairs", y="recall", hue="bands", style="ngram", s=60)
    plt.xscale("log")
    plt.xlabel("# Candidate pairs (log scale)")
    plt.ylabel("Recall")
    plt.title("Recall vs Candidate Pairs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir/"recall_vs_pairs.png", dpi=150)
    plt.close()

    # 2. Recall vs Reduction Ratio
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=metrics_df, x="reduction_ratio", y="recall", hue="bands", style="ngram", s=60)
    plt.xlabel("Reduction Ratio")
    plt.ylabel("Recall")
    plt.title("Recall vs Reduction Ratio")
    plt.tight_layout()
    plt.savefig(outdir/"recall_vs_reduction.png", dpi=150)
    plt.close()

    # 3. Heatmap of recall by bands vs max_block_size (fixing ngram=3)
    pivot = metrics_df[metrics_df["ngram"]==3].pivot_table(
        index="bands", columns="max_block_size", values="recall", aggfunc="max"
    )
    plt.figure(figsize=(8,6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Recall by bands vs max_block_size (ngram=3)")
    plt.savefig(outdir/"heatmap_recall.png", dpi=150)
    plt.close()

    # 4. Recall distribution by ngram
    plt.figure(figsize=(8,6))
    sns.boxplot(data=metrics_df, x="ngram", y="recall")
    plt.title("Recall distribution by ngram")
    plt.savefig(outdir/"recall_by_ngram.png", dpi=150)
    plt.close()

def make_report(metrics_df: pd.DataFrame, best: Dict[str,Any], outdir: Path):
    lines = ["# LSH Grid Search Report\n", "## Best Parameters\n"]
    lines.append("```json\n" + json.dumps(best, indent=2) + "\n```\n")

    lines.append("## Summary Statistics\n")
    lines.append(f"- Grid size: {len(metrics_df)} configs\n")
    lines.append(f"- Recall range: {metrics_df['recall'].min():.3f} – {metrics_df['recall'].max():.3f}\n")
    lines.append(f"- Candidate pairs range: {metrics_df['n_pairs'].min():,} – {metrics_df['n_pairs'].max():,}\n")
    lines.append(f"- Reduction ratio range: {metrics_df['reduction_ratio'].min():.5f} – {metrics_df['reduction_ratio'].max():.5f}\n")

    lines.append("## Plots\n")
    lines.append("![Recall vs Pairs](recall_vs_pairs.png)\n")
    lines.append("![Recall vs Reduction](recall_vs_reduction.png)\n")
    lines.append("![Heatmap Recall](heatmap_recall.png)\n")
    lines.append("![Recall by Ngram](recall_by_ngram.png)\n")

    report_path = outdir/"report.md"
    report_path.write_text("".join(lines), encoding="utf-8")
    print(f"[SUCCESS] Wrote markdown report → {report_path}")

if __name__=="__main__":
    main()
