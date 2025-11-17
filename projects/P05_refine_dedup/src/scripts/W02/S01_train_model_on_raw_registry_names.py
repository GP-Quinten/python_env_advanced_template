#!/usr/bin/env python3
import sys
import time
import logging
import numpy as np
import pandas as pd
import json
import os
from math import sqrt

from pathlib import Path

# print working directory
print(Path.cwd())

from tqdm import tqdm
from itertools import product
import click

# Adjust imports as needed in your environment
from sklearn.metrics import precision_recall_fscore_support
from src.p05_refine_dedup.utils.utils import (
    global_clustering,
    subcluster_noise,
    subcluster_big_clusters,
    build_final_cluster_col,
)
from src.p05_refine_dedup import config
from src.p05_refine_dedup.utils.s3_io_functions import (
    load_parquet_from_s3,
    upload_parquet_to_s3,
)
from etc.param_config import PARAMS_RANGES_CONFIG

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


###############################################
# 2) METRICS CALCULATION
###############################################


def compute_custom_score(precision, recall, beta=sqrt(1 / 2)):
    if precision + recall == 0:
        return 0
    return ((1 + beta**2) * (precision * recall)) / ((beta**2 * precision) + recall)


def retrieve_registry_name(df, col_to_extract, col_registry_name="registry_name"):
    """
    Extract the registry name from the given column in the DataFrame.
    If the column is not present, return None.
    Only removes the parenthetical part if it appears at the end of the string.
    """
    if col_to_extract in df.columns:
        return df[col_to_extract].apply(
            lambda x: (
                x[: x.rfind(" (")]
                if isinstance(x, str) and " (" in x and x.endswith(")")
                else x
            )
        )
    return None


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
    # rename 'alias' by 'alias_full_name'
    df_eval = df_eval.rename(columns={"alias": "alias_full_name"})
    # Create registry_name column
    df_eval["registry_name"] = retrieve_registry_name(
        df_eval, "full_name", col_registry_name="registry_name"
    )
    # create alias column
    df_eval["alias"] = retrieve_registry_name(
        df_eval, "alias_full_name", col_registry_name="alias"
    )
    cluster_map = df_local.set_index("registry_name")[cluster_col].to_dict()

    df_tmp = df_eval.copy()
    df_tmp["cluster_match_label"] = 0
    for idx, row in df_tmp.iterrows():
        ds1 = row["registry_name"]
        ds2 = row["alias"]
        label1 = cluster_map.get(ds1, -9999)
        label2 = cluster_map.get(ds2, -9999)
        # if ds1 and ds2 are in the same cluster, and not noise (string does not end with '0')
        if (
            label1 == label2
            and label1 != -9999
            and ((label1 != 0) or (not str(label1).endswith("_0")))
        ):
            df_tmp.at[idx, "cluster_match_label"] = 1

    y_true = df_tmp["final_label"].values
    y_pred = df_tmp["cluster_match_label"].values
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    beta = sqrt(1 / 2)  # 0.707  #
    custom_score = compute_custom_score(precision, recall, beta=beta)

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
        "custom_score": custom_score,
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
    "--s3_input_embeddings",
    type=str,
    help="Path to the embeddings in s3 bucket.",
)
@click.option(
    "--evaluation_dataset",
    type=str,
    help="Path to the evaluation dataset.",
)
@click.option(
    "--best_config_json",
    type=str,
    help="Path to the best configuration JSON file.",
)
@click.option(
    "--grid_search_results_xlsx",
    type=str,
    help="Path to save the grid search results in Excel format.",
)
def main(
    s3_input_embeddings,
    evaluation_dataset,
    best_config_json,
    grid_search_results_xlsx,
    params_ranges_config=PARAMS_RANGES_CONFIG,
):
    # track time
    start_time = time.time()

    # ------------- LOAD DATA -------------
    bucket_name = config.BUCKET_NAME_DEV
    # folder_path = s3_input_embeddings, file_name = last part
    folder_path = s3_input_embeddings.rsplit("/", 1)[0]
    file_name = s3_input_embeddings.rsplit("/", 1)[-1]
    df = load_parquet_from_s3(
        bucket_name=bucket_name,
        folder_path=folder_path,
        file_name=file_name,
    )
    # # slect first 1000 for testing
    # df = df.head(1000)  # Uncomment for testing with a smaller subset

    df_eval = pd.read_excel(
        evaluation_dataset, engine="openpyxl"
    )  # Load evaluation dataset from Excel
    # # test on 10 pairs
    # df_eval = df_eval.head(10)  # Uncomment for testing with a smaller subset
    # Filter/rename as needed
    # df_eval.rename(columns={"final_label": "LLM_label"}, inplace=True)

    # ------------- DEFINE RANGES -------------
    # use params_ranges_config
    # start with primary
    eps1_values = params_ranges_config["primary"]["eps"]
    min_samples1_values = params_ranges_config["primary"]["min_samples"]
    eps2_values = params_ranges_config["noise"]["eps"]
    min_samples2_values = params_ranges_config["noise"]["min_samples"]
    eps3_values = params_ranges_config["big"]["eps"]
    min_samples3_values = params_ranges_config["big"]["min_samples"]

    results_list = []
    best_custom_score = -1
    best_f1 = -1
    best_precision = -1
    best_config = None

    # ------------- CACHES -------------
    # Store partial results so we only do each step once per param
    global_cache = {}  # key=eps1 -> (df_global, global_metrics, global_time)
    noise_cache = (
        {}
    )  # key=(eps1, eps2) -> (df_noise, sub_noise_metrics, sub_noise_time)

    # ------------- NESTED LOOPS -------------
    logging.warning("Starting nested grid search ...")
    cpu_count = os.cpu_count()
    logging.warning(f"Number of CPUs found/used for DBSCAN: {cpu_count}")
    outer_pbar = tqdm(eps1_values, desc="eps1 loop", position=0)
    for eps1 in outer_pbar:
        # -- Step 1: Global, done once per eps1.
        if eps1 not in global_cache:
            # Compute global DBSCAN
            df_global, global_time = global_clustering(
                df, eps1=eps1, min_samples=2, col_embedding="registry_name_embedding"
            )
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
            # Enforce eps2 > eps1
            if eps2 <= eps1:
                continue
            # -- Step 2: Noise subclustering
            if (eps1, eps2) not in noise_cache:
                df_sub_noise, noise_time = subcluster_noise(
                    df_global,
                    eps2=eps2,
                    min_samples=2,
                    col_embedding="registry_name_embedding",
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
                # Enforce eps3 < eps1
                if eps3 >= eps1:
                    continue
                # -- Step 3: Big cluster subclustering
                df_sub_final, final_time = subcluster_big_clusters(
                    df_sub_noise,
                    eps3=eps3,
                    min_samples=2,
                    size_threshold=5,
                    col_embedding="registry_name_embedding",
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
                    "global_custom_score": global_metrics["custom_score"],
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
                    "sub_noise_custom_score": sub_noise_metrics["custom_score"],
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
                    "sub_final_custom_score": sub_final_metrics["custom_score"],
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
                if sub_final_metrics["precision"] > best_precision:
                    best_precision = sub_final_metrics["precision"]
                    logging.warning(
                        f"New best precision found: {sub_final_metrics['precision']:.3f} "
                        f"for eps1={eps1:.3f}, eps2={eps2:.3f}, eps3={eps3:.3f}"
                    )

                # Track best config by sub_final_f1
                if sub_final_metrics["custom_score"] > best_custom_score:
                    # if sub_final_metrics["f1"] > best_f1:
                    logging.warning(
                        f"!! --- New best score found: {sub_final_metrics['custom_score']:.3f} /n"
                        f"F1: {sub_final_metrics['f1']:.3f} "
                        f"Precision: {sub_final_metrics['precision']:.3f} "
                        f"Rcall: {sub_final_metrics['recall']:.3f} "
                        f"for eps1={eps1:.3f}, eps2={eps2:.3f}, eps3={eps3:.3f}"
                    )
                    best_custom_score = sub_final_metrics["custom_score"]
                    best_config = {
                        "eps1": round(eps1, 4),
                        "eps2": round(eps2, 4),
                        "eps3": round(eps3, 4),
                        "custom_score": sub_final_metrics["custom_score"],
                        "f1": sub_final_metrics["f1"],
                        "precision": sub_final_metrics["precision"],
                        "recall": sub_final_metrics["recall"],
                        # Add more metrics if desired
                    }

    # ------------- CREATE DATAFRAME -------------
    results_df = pd.DataFrame(results_list).where(pd.notnull, np.nan)

    # ------------- SAVE TO DISK -------------
    # use grid_search_results_xlsx as output file
    output_dir = Path(grid_search_results_xlsx).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_excel(grid_search_results_xlsx, index=False)
    logging.warning(f"Saved grid search results to {grid_search_results_xlsx}")

    # Save best config to JSON
    if best_config is not None:
        with open(best_config_json, "w") as f:
            json.dump(best_config, f, indent=2)
        logging.warning(f"Saved best config to {best_config_json}")

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
