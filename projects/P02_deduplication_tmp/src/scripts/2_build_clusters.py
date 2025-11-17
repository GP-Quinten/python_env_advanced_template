#!/usr/bin/env python3
import sys
from pathlib import Path

# Set project root and update system path for local modules
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import time
import weaviate
from dotenv import load_dotenv
import logging
from src.logger import init_logger
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import plotly.express as px

from config import config
from src.utils.io_functions import load_file_from_s3, upload_file_to_s3
from src.utils.utils import run_dbscan

best_eps1 = 0.348  # 0.3314610779075241
best_eps2 = 0.400  # 0.380396662361368
best_eps3 = 0.275  # 0.275073319059974
BEST_MIN_SAMPLES = 2  # All min_samples are fixed at 2

# Load environment variables from .env file
load_dotenv()


def main():
    # Track Time of Execution
    start_time = time.time()
    # Initialize logger using the custom logger initialization function
    init_logger(stream=True, file=True, file_path="logs/logs.txt", level="INFO")

    # Print current working directory
    print(Path.cwd())

    # Load table from S3 with columns: 'data_source_name' and 'embedding'
    df = load_file_from_s3(
        bucket_name=config.BUCKET_NAME_DEV,
        folder_path="registry_data_catalog_experiments/task_1_deduplication/publications_embeddings/",
        file_name="data_source_name_MistralEmbed_embeddings_all.parquet",
    )

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

    # Log updated noise percentage after removal
    new_total = len(df)
    new_noise_count = (df["cluster"] == 0).sum()
    new_noise_percentage = (new_noise_count / new_total) * 100
    logging.warning(
        f"After removal: Noise (cluster 0) percentage: {new_noise_percentage:.2f}%"
    )

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

    # Save the final dataframe to S3
    folder_path = (
        "registry_data_catalog_experiments/task_1_deduplication/publications_clusters/"
    )
    file_name = "data_source_names_clusters_v2_all.parquet"
    logging.warning("Uploading final clusters to S3...")
    upload_file_to_s3(
        df,
        config.BUCKET_NAME_DEV,
        folder_path,
        file_name,
    )

    # JOIN With perfect string duplicates to then obtain cluster size distribution
    weaviate_client = weaviate.connect_to_custom(**config.WEAVIATE_PROD_CONF)
    collections = weaviate_client.collections  #
    # load publications
    collection_publications = collections.get("Publication_v2")
    # load data source names
    items = []
    for item in collection_publications.iterator(include_vector=False):
        # Extract subset of properties
        items.append(
            {
                k: v
                for k, v in item.properties.items()
                if k
                in [
                    "object_id",
                    # "title",
                    "data_source_name",
                ]
            }
        )
    # close weaviate connection
    weaviate_client.close()
    df_items = pd.DataFrame(items)

    # logging numnber of items
    logging.warning(f"Number of items: {len(df_items)}")
    # drop missing data source names
    df_items.dropna(axis=0, inplace=True)
    df_items.drop(df_items[df_items["data_source_name"] == ""].index, inplace=True)
    logging.warning(f"Number of items after dropping missing values: {len(df_items)}")
    # drop not specified data source names
    df_items = df_items[
        ~df_items["data_source_name"].str.lower().str.contains("not specified")
    ]
    logging.warning(
        f"Number of items after dropping missing values and not specified: {len(df_items)}"
    )
    # add column "name_lower_case"
    df_items["name_lower_case"] = df_items["data_source_name"].str.lower()
    # show value counts top5
    logging.warning(df_items["name_lower_case"].value_counts().head(5))
    # remove the rows with data_source_name with more than 10 words
    df_items = df_items[
        df_items["data_source_name"].apply(lambda x: len(x.split()) <= 10)
    ]
    logging.warning(
        f"Number of items after dropping missing values, not specified, data_source_name with more than 10 words: {len(df_items)}"
    )
    logging.warning(
        f"Number of items after dropping missing values, not specified, data_source_name with more than 10 words, and perfect duplicates: {len(df_items)}"
    )

    # JOIN with perfect string duplicates (left join df_items with df on data_source_name)
    df_all = df_items.merge(
        df[["data_source_name", "Final_Cluster"]],
        how="left",
        on="data_source_name",
    )

    # Save the final dataframe to S3
    folder_path = (
        "registry_data_catalog_experiments/task_1_deduplication/publications_clusters/"
    )
    file_name = "data_source_names_clusters_v2_all_with_perfect_duplicates.parquet"
    logging.warning("Uploading final clusters with perfect duplicates to S3...")
    upload_file_to_s3(
        df_all,
        config.BUCKET_NAME_DEV,
        folder_path,
        file_name,
    )

    # ---- Show Distribution of Cluster Sizes ---- #
    # extract the noise Final clusters
    noise_clusters = df_all[df_all["Final_Cluster"].str.endswith("_0")]
    logging.warning(
        f"Number of isolated data source names (no certain if they are real registries): {len(noise_clusters)}"
    )
    df_no_noise = df_all[
        ~df_all["Final_Cluster"].str.endswith("_0")
    ]  # remove noise clusters
    # compute the size of each cluster (groub by cluster and count the number of data sources)
    cluster_size = df_no_noise.groupby("Final_Cluster").size().reset_index(name="size")
    # compute the number of clusters for each size
    cluster_size_distribution = (
        cluster_size.groupby("size").size().reset_index(name="count")
    )
    # rank by size descending
    cluster_size_distribution = cluster_size_distribution.sort_values(
        "size", ascending=False
    )
    # count number of clusters with size > 50
    huge_clusters = (cluster_size_distribution["size"] > 50).sum()
    logging.warning(f"Number of clusters with size > 50: {huge_clusters}")
    # filter out the clusters with size > 50
    cluster_size_distribution = cluster_size_distribution[
        cluster_size_distribution["size"] <= 50
    ]
    fig_dist = px.bar(
        cluster_size_distribution,
        x="size",
        y="count",
        text="count",
        title="Cluster size distribution",
    )
    fig_dist.update_traces(
        textposition="outside",
        hovertemplate="Size: %{x}<br>Count: %{y}<extra></extra>",
    )
    # add to the figure the counts of noise_clusters and huge_clusters, as notes
    fig_dist.add_annotation(
        x=0.5,
        y=0.9,
        xref="paper",
        yref="paper",
        text=f"Number of isolated data source names (noise): {len(noise_clusters)}",
        showarrow=False,
    )
    fig_dist.add_annotation(
        x=0.5,
        y=0.85,
        xref="paper",
        yref="paper",
        text=f"Number of clusters with size > 50: {huge_clusters}",
        showarrow=False,
    )
    # save figure
    fig_dist.write_html("data/2_build_clusters/cluster_size_distribution_all.html")

    # Log total time of execution
    total_time = time.time() - start_time
    logging.warning(f"Total time of execution: {total_time:.2f} seconds.")

    logging.warning("Upload completed.")


if __name__ == "__main__":
    main()
