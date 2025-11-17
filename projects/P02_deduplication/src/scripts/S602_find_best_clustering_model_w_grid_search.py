#!/usr/bin/env python3
import sys
import time
import logging
import numpy as np
import pandas as pd

from pathlib import Path

# Set project root and update system path for local modules
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from tqdm import tqdm
from itertools import product
import click

# Adjust imports as needed in your environment
from p02_deduplication.config import config
from p02_deduplication.logger import init_logger
from sklearn.metrics import precision_recall_fscore_support
from p02_deduplication.utils.utils import run_dbscan


###############################################
# 1) CLUSTERING / EVALUATION HELPERS
###############################################


def global_clustering(df, eps1, min_samples=2):
    """
    Run DBSCAN once for the entire dataset with eps1.
    Return (df_global, global_time).
    df_global['cluster'] has labels (0 for noise, 1.. for clusters).
    """
    df_local = df.copy(deep=True)
    embeddings = np.vstack(df_local["embedding"].values)
    labels, comp_time = run_dbscan(embeddings, eps=eps1, min_samples=min_samples)
    df_local["cluster"] = labels
    return df_local, comp_time


def subcluster_noise(df_global, eps2, min_samples=2):
    """
    From df_global, subcluster noise points (cluster=0) with eps2.
    Return (df_noise, noise_time).
    df_noise['Subcluster'] has sub-labels for noise points only.
    """
    df_local = df_global.copy(deep=True)
    df_local["Subcluster"] = None

    # noise mask, cluster col = 0
    noise_mask = df_local["cluster"] == 0
    if not noise_mask.any():
        return df_local, 0.0  # no noise -> skip

    embeddings_noise = np.vstack(df_local.loc[noise_mask, "embedding"])
    sub_labels, comp_time = run_dbscan(
        embeddings_noise, eps=eps2, min_samples=min_samples
    )
    df_local.loc[noise_mask, "Subcluster"] = sub_labels
    return df_local, comp_time


def subcluster_big_clusters(df_noise, eps3, min_samples=2, size_threshold=10):
    """
    From df_noise, subcluster any big cluster (>=size_threshold) with eps3.
    Return (df_final, final_time).
    """
    df_local = df_noise.copy(deep=True)
    total_time = 0.0

    # We already have df_local['cluster'] from global, and 'Subcluster' from noise subclustering.
    # We will re-use df_local['Subcluster'] for big clusters as well.
    clusters = df_local["cluster"].unique()
    for c in clusters:
        if c == 0:
            continue  # skip noise
        mask = df_local["cluster"] == c
        if mask.sum() < size_threshold:
            continue
        embeddings_cluster = np.vstack(df_local.loc[mask, "embedding"])
        sub_labels, comp_time = run_dbscan(
            embeddings_cluster, eps=eps3, min_samples=min_samples
        )
        df_local.loc[mask, "Subcluster"] = sub_labels
        total_time += comp_time

    return df_local, total_time


def build_final_cluster_col(df_local):
    """
    Combine 'cluster' and 'Subcluster' into one column 'Final_Cluster'.
    - If cluster==0 (noise) and Subcluster is not None, do '0_sub'
    - If cluster!=0 and Subcluster is not None, do 'c_sub'
    - Otherwise just use cluster
    """
    df_local["Final_Cluster"] = df_local["cluster"].astype(str)

    # For noise
    noise_mask = (df_local["cluster"] == 0) & (df_local["Subcluster"].notnull())
    df_local.loc[noise_mask, "Final_Cluster"] = (
        df_local.loc[noise_mask, "cluster"].astype(str)
        + "_"
        + df_local.loc[noise_mask, "Subcluster"].astype(str)
    )

    # For big clusters
    big_mask = (df_local["cluster"] != 0) & (df_local["Subcluster"].notnull())
    df_local.loc[big_mask, "Final_Cluster"] = (
        df_local.loc[big_mask, "cluster"].astype(str)
        + "_"
        + df_local.loc[big_mask, "Subcluster"].astype(str)
    )
    return df_local


###############################################
# 2) METRICS CALCULATION
###############################################
def evaluate_stage(df_local, df_eval, cluster_col="cluster"):
    """
    For each row in df_eval, check if the two data sources share the same cluster label.
    Then compute precision, recall, f1, plus some 'noise' metrics.

    Returns a dict with:
      { "f1", "precision", "recall",
        "N_single_occ_registries", "N_multi_occ_registries",
        "N_distinct_registries", "Noise_perc" }
    """
    # 1) Pairwise cluster matching
    cluster_map = df_local.set_index("data_source_name")[cluster_col].to_dict()

    df_tmp = df_eval.copy()
    df_tmp["cluster_match_label"] = 0
    for idx, row in df_tmp.iterrows():
        ds1 = row["data_source_name_1"]
        ds2 = row["data_source_name_2"]
        label1 = cluster_map.get(ds1, -9999)
        label2 = cluster_map.get(ds2, -9999)
        # if ds1 and ds2 are in the same cluster, and not noise (string does not end with '0')
        if (
            label1 == label2
            and label1 != -9999
            and ((label1 != 0) or (not str(label1).endswith("_0")))
        ):
            df_tmp.at[idx, "cluster_match_label"] = 1

    y_true = df_tmp["LLM_label"].values
    y_pred = df_tmp["cluster_match_label"].values
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    # 2) Additional cluster stats
    #   single-occ = final cluster name ends with "_0" or equals to 0
    #   multi-occ = all other cluster labels
    condition = (df_local[cluster_col].astype(str).str.endswith("_0")) | (
        df_local[cluster_col] == 0
    )
    N_single_occ_registries = df_local[condition].shape[0]
    N_multi_occ_registries = df_local[~condition][cluster_col].nunique()
    N_distinct_registries = N_single_occ_registries + N_multi_occ_registries
    Noise_perc = 0.0
    if N_distinct_registries > 0:
        Noise_perc = (N_single_occ_registries / df_local.shape[0]) * 100

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "N_single_occ_registries": int(N_single_occ_registries),
        "N_multi_occ_registries": int(N_multi_occ_registries),
        "N_distinct_registries": int(N_distinct_registries),
        "Noise_perc": Noise_perc,
    }


###############################################
# 3) MAIN: NESTED LOOPS, CACHE PARTIAL RESULTS
###############################################
@click.command()
@click.option(
    "--training_dataset",
    type=str,
    help="Path to the training dataset.",
)
@click.option(
    "--evaluation_dataset",
    type=str,
    help="Path to the evaluation dataset.",
)
@click.option(
    "--output_dir",
    type=str,
    help="Path to the output directory.",
)
def main(training_dataset, evaluation_dataset, output_dir):
    # track time
    start_time = time.time()
    init_logger(
        stream=True,
        file=True,
        file_path="logs/optimized_training_logs.txt",
        level="WARNING",
    )

    # ------------- LOAD DATA -------------
    df = pd.read_parquet(training_dataset)
    df_eval = pd.read_parquet(evaluation_dataset)
    # Filter/rename as needed
    df_eval = df_eval[df_eval["to_include"] == "Yes"].copy()
    # df_eval = df_eval[
    #     ~(
    #         df_eval["explanation"].str.startswith("Inclusion-uncertainty")
    #         & (df_eval["diff"] != 0)
    #     )
    # ]
    df_eval.rename(columns={"final_label": "LLM_label"}, inplace=True)

    # ------------- DEFINE RANGES -------------
    eps1_values = np.arange(0.305, 0.360001, 0.005)  # for global
    eps2_values = np.arange(0.30, 0.500001, 0.05)  # for noise
    eps3_values = np.arange(0.265, 0.280001, 0.005)  # for big clusters

    results_list = []

    # ------------- CACHES -------------
    # Store partial results so we only do each step once per param
    global_cache = {}  # key=eps1 -> (df_global, global_metrics, global_time)
    noise_cache = (
        {}
    )  # key=(eps1, eps2) -> (df_noise, sub_noise_metrics, sub_noise_time)

    # ------------- NESTED LOOPS -------------
    logging.warning("Starting nested grid search ...")

    outer_pbar = tqdm(eps1_values, desc="eps1 loop", position=0)
    for eps1 in outer_pbar:
        # -- Step 1: Global, done once per eps1.
        if eps1 not in global_cache:
            # Compute global DBSCAN
            df_global, global_time = global_clustering(df, eps1=eps1, min_samples=2)
            # Evaluate
            global_metrics = evaluate_stage(df_global, df_eval, cluster_col="cluster")
            global_cache[eps1] = (df_global, global_metrics, global_time)

        # Unpack from cache
        df_global, global_metrics, global_time = global_cache[eps1]

        # Middle loop: eps2
        middle_pbar = tqdm(
            eps2_values, desc=f"eps2 loop (eps1={eps1:.3f})", position=1, leave=False
        )
        for eps2 in middle_pbar:
            # -- Step 2: Noise subclustering
            if (eps1, eps2) not in noise_cache:
                df_sub_noise, noise_time = subcluster_noise(
                    df_global, eps2=eps2, min_samples=2
                )
                # Build "Final_Cluster" for noise subclustering stage
                df_sub_noise_copy = df_sub_noise.copy()
                df_sub_noise_copy = build_final_cluster_col(
                    df_sub_noise_copy.rename(
                        columns={"cluster": "cluster", "Subcluster": "Subcluster"}
                    )
                )

                sub_noise_metrics = evaluate_stage(
                    df_sub_noise_copy, df_eval, cluster_col="Final_Cluster"
                )
                noise_cache[(eps1, eps2)] = (
                    df_sub_noise,
                    sub_noise_metrics,
                    noise_time,
                )

            df_sub_noise, sub_noise_metrics, noise_time = noise_cache[(eps1, eps2)]

            # Inner loop: eps3
            inner_pbar = tqdm(
                eps3_values,
                desc=f"eps3 loop (eps1={eps1:.3f}, eps2={eps2:.3f})",
                position=2,
                leave=False,
            )
            for eps3 in inner_pbar:
                # -- Step 3: Big cluster subclustering
                df_sub_final, final_time = subcluster_big_clusters(
                    df_sub_noise, eps3=eps3, min_samples=2, size_threshold=10
                )
                # Build "Final_Cluster"
                df_sub_final_copy = build_final_cluster_col(df_sub_final)

                sub_final_metrics = evaluate_stage(
                    df_sub_final_copy, df_eval, cluster_col="Final_Cluster"
                )

                # --------------- COMBINE METRICS ---------------
                row_dict = {
                    # Params
                    "eps1": round(eps1, 4),
                    "eps2": round(eps2, 4),
                    "eps3": round(eps3, 4),
                    # Global metrics
                    "global_f1": global_metrics["f1"],
                    "global_precision": global_metrics["precision"],
                    "global_recall": global_metrics["recall"],
                    "global_N_single_occ_reg": global_metrics[
                        "N_single_occ_registries"
                    ],
                    "global_N_multi_occ_reg": global_metrics["N_multi_occ_registries"],
                    "global_N_distinct_reg": global_metrics["N_distinct_registries"],
                    "global_Noise_perc": global_metrics["Noise_perc"],
                    "global_compute_time": global_time,
                    # Sub-noise metrics
                    "sub_noise_f1": sub_noise_metrics["f1"],
                    "sub_noise_precision": sub_noise_metrics["precision"],
                    "sub_noise_recall": sub_noise_metrics["recall"],
                    "sub_noise_N_single_occ_reg": sub_noise_metrics[
                        "N_single_occ_registries"
                    ],
                    "sub_noise_N_multi_occ_reg": sub_noise_metrics[
                        "N_multi_occ_registries"
                    ],
                    "sub_noise_N_distinct_reg": sub_noise_metrics[
                        "N_distinct_registries"
                    ],
                    "sub_noise_Noise_perc": sub_noise_metrics["Noise_perc"],
                    "sub_noise_compute_time": noise_time,
                    # Sub-final metrics
                    "sub_final_f1": sub_final_metrics["f1"],
                    "sub_final_precision": sub_final_metrics["precision"],
                    "sub_final_recall": sub_final_metrics["recall"],
                    "sub_final_N_single_occ_reg": sub_final_metrics[
                        "N_single_occ_registries"
                    ],
                    "sub_final_N_multi_occ_reg": sub_final_metrics[
                        "N_multi_occ_registries"
                    ],
                    "sub_final_N_distinct_reg": sub_final_metrics[
                        "N_distinct_registries"
                    ],
                    "sub_final_Noise_perc": sub_final_metrics["Noise_perc"],
                    "sub_final_compute_time": final_time,
                }

                total_comp_time = global_time + noise_time + final_time
                row_dict["total_compute_time"] = total_comp_time

                results_list.append(row_dict)

    # ------------- CREATE DATAFRAME -------------
    results_df = pd.DataFrame(results_list).where(pd.notnull, np.nan)

    # ------------- SAVE TO DISK -------------
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "grid_search_results.xlsx"
    results_df.to_excel(output_file, index=False)
    logging.warning("Results uploaded to S3 as grid_search_results.xlsx")

    # compute mean time of 'total_compute_time'
    mean_time = results_df["total_compute_time"].mean()
    logging.warning(
        f"Mean total computation time for a single triplet of parameters: {mean_time:.2f} seconds."
    )
    # track total time computation in minutes
    total_time = (time.time() - start_time) / 60
    logging.warning(f"Total time taken: {total_time:.2f} minutes.")


if __name__ == "__main__":
    main()
