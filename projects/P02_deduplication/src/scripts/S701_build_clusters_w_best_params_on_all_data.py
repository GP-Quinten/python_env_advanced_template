#!/usr/bin/env python3
import sys
from pathlib import Path

# Set project root and update system path for local modules
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import time
from dotenv import load_dotenv
import logging
import pandas as pd
import numpy as np
import click

from p02_deduplication.logger import init_logger
from p02_deduplication.utils.utils import run_dbscan

best_eps1 = 0.348
best_eps2 = 0.400
best_eps3 = 0.275
BEST_MIN_SAMPLES = 2  # All min_samples are fixed at 2

# Load environment variables from .env file
load_dotenv()


@click.command()
@click.option(
    "--data_source_names_embeddings_all",
    "-i",
    type=str,
    required=True,
    help="Input parquet file path: sampled data source names with embeddings.",
)
@click.option(
    "--clusters_parquet",
    type=str,
    help="Output path where the parquet file with clusters will be saved.",
)
def main(data_source_names_embeddings_all, clusters_parquet):
    # Track Time of Execution
    start_time = time.time()
    # Initialize logger using the custom logger initialization function
    init_logger(level="WARNING")

    # Load table
    df = pd.read_parquet(data_source_names_embeddings_all)

    # Apply initial DBSCAN clustering on the entire dataset
    embeddings = np.vstack(df["embedding"].values)
    # log "compute clustering"
    logging.warning("Computing initial clustering...")
    labels, comp_time = run_dbscan(embeddings, best_eps1, BEST_MIN_SAMPLES)
    df["cluster"] = labels
    df["Subcluster"] = None

    # Logging number of distinct clusters and percentage of noise (cluster 0)
    total_count = len(df)
    distinct_clusters = df["cluster"].nunique()
    noise_count = (df["cluster"] == 0).sum()
    noise_percentage = (noise_count / total_count) * 100
    logging.warning(
        f"Initial clustering: {distinct_clusters} distinct clusters found. Noise (cluster 0) percentage: {noise_percentage:.2f}%"
    )

    # # Remove cluster number 6 (clinical trials) and log the removal
    # # show two examples of data source names in cluster 6
    # cluster6_examples = df[df["cluster"] == 6].head(2)
    # logging.warning(
    #     f"Examples of data source names in cluster 6 (clinical trials):\n{cluster6_examples['data_source_name']}"
    # )
    # cluster6_count = (df["cluster"] == 6).sum()
    # if cluster6_count > 0:
    #     logging.warning(
    #         f"Removing cluster 6: {cluster6_count} data source names (clinical trials), representing {(cluster6_count/total_count)*100:.2f}% of total."
    #     )
    # df = df[df["cluster"] != 6].reset_index(drop=True)

    # # Log updated noise percentage after removal
    # new_total = len(df)
    # new_noise_count = (df["cluster"] == 0).sum()
    # new_noise_percentage = (new_noise_count / new_total) * 100
    # logging.warning(
    #     f"After removal: Noise (cluster 0) percentage: {new_noise_percentage:.2f}%"
    # )

    # Sub-clustering on noise cluster (cluster 0)
    df_noise = df[df["cluster"] == 0]
    if not df_noise.empty:
        embeddings_noise = np.vstack(df_noise["embedding"].values)
        # log "compute sub-clustering"
        logging.warning("Computing sub-clustering on noise cluster (cluster 0)...")
        sub_labels, sub_time = run_dbscan(
            embeddings_noise, eps=best_eps2, min_samples=BEST_MIN_SAMPLES
        )
        df.loc[df["cluster"] == 0, "Subcluster"] = sub_labels
        distinct_subclusters = len(set(sub_labels) - {0})
        logging.warning(
            f"Sub-clustering on noise cluster (cluster 0): {distinct_subclusters} new distinct subclusters found (excluding noise)."
        )
    else:
        logging.warning("No noise cluster (cluster 0) found for sub-clustering.")

    # Sub-clustering on clusters with size >= 10 (excluding noise cluster)
    clusters = df["cluster"].unique()
    clusters_to_subcluster = [
        c for c in clusters if c != 0 and (df["cluster"] == c).sum() >= 10
    ]
    logging.warning(
        f"Found {len(clusters_to_subcluster)} clusters with size >= 10 for sub-clustering."
    )
    total_clusters_to_process = len(clusters_to_subcluster)

    for idx, cluster_label in enumerate(clusters_to_subcluster, start=1):
        logging.warning(
            f"Processing cluster {idx}/{total_clusters_to_process} (cluster {cluster_label})"
        )
        df_cluster = df[df["cluster"] == cluster_label]
        embeddings_cluster = np.vstack(df_cluster["embedding"].values)
        sub_labels, comp_time = run_dbscan(
            embeddings_cluster, eps=best_eps3, min_samples=BEST_MIN_SAMPLES
        )
        df.loc[df["cluster"] == cluster_label, "Subcluster"] = sub_labels
        distinct_subclusters = len(set(sub_labels) - {0})
        logging.warning(
            f"Cluster {cluster_label}): {distinct_subclusters} distinct subclusters found (excluding noise)."
        )

    # Create Final_Cluster column as a combination of cluster and Subcluster
    df["Final_Cluster"] = df["cluster"].astype(str) + "_" + df["Subcluster"].astype(str)

    # save in parquet format
    logging.warning("Uploading final clusters ...")
    # make sure the output directory exists
    clusters_parquet = Path(clusters_parquet)
    clusters_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(clusters_parquet, index=False)

    # Log total time of execution
    total_time = time.time() - start_time
    logging.warning(f"Total time of execution: {total_time:.2f} seconds.")


if __name__ == "__main__":
    main()
