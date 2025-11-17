#!/usr/bin/env python3
import time
import json
import click
import logging
import pandas as pd
import numpy as np
from pathlib import Path

from src.p05_refine_dedup import config
from src.p05_refine_dedup.utils.s3_io_functions import load_parquet_from_s3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--s3_input_embeddings",
    type=str,
    help="S3 path to the registry embeddings parquet file.",
)
@click.option(
    "--s3_ema_embeddings",
    type=str,
    help="S3 Path to EMA registry embeddings parquet file.",
)
@click.option("--clusters_table_xlsx", type=str, help="Path to clusters table (Excel).")
@click.option(
    "--ema_prediction_results_xlsx",
    type=str,
    help="Path to save EMA prediction results (Excel).",
)
@click.option(
    "--ema_performance_report_json",
    type=str,
    help="Path to save EMA performance report (JSON).",
)
def main(
    s3_input_embeddings,
    s3_ema_embeddings,
    clusters_table_xlsx,
    ema_prediction_results_xlsx,
    ema_performance_report_json,
):
    start_time = time.time()

    # 1. Load registry embeddings from S3
    logger.info(f"Loading registry embeddings from {s3_input_embeddings}")
    bucket_name = config.BUCKET_NAME_DEV
    folder_path = s3_input_embeddings.rsplit("/", 1)[0]
    file_name = s3_input_embeddings.rsplit("/", 1)[-1]
    embeddings_df = load_parquet_from_s3(
        bucket_name=bucket_name,
        folder_path=folder_path,
        file_name=file_name,
    )

    # 2. Load cluster assignments
    logger.info(f"Loading clusters table from {clusters_table_xlsx}")
    clusters_df = pd.read_excel(clusters_table_xlsx)

    # 3. Load EMA registry embeddings
    logger.info(f"Loading EMA registry embeddings from {s3_ema_embeddings}")
    s3_ema_embeddings = "registry_data_catalog_experiments/P05_refine_dedup/ema_registry_names_embeddings.parquet"
    ema_folder_path = s3_ema_embeddings.rsplit("/", 1)[0]
    ema_file_name = s3_ema_embeddings.rsplit("/", 1)[-1]
    ema_embeddings_df = load_parquet_from_s3(
        bucket_name=bucket_name,
        folder_path=ema_folder_path,
        file_name=ema_file_name,
    )

    # 4. Preprocessing
    embeddings_df["full_name_embedding"] = embeddings_df["full_name_embedding"].apply(
        np.array
    )
    ema_embeddings_df["full_name_embedding"] = ema_embeddings_df[
        "full_name_embedding"
    ].apply(np.array)
    clusters_df["Final_Cluster"] = clusters_df["Final_Cluster"].astype(str)
    logger.info("Embeddings and clusters preprocessed.")

    # 5. Nearest Neighbor Assignment
    from sklearn.metrics.pairwise import cosine_distances

    emb_matrix = np.vstack(embeddings_df["full_name_embedding"].values)
    embeddings_with_clusters = embeddings_df.merge(
        clusters_df, on=["full_name", "number_of_occurrences"], how="left"
    )
    results = []
    for idx, ema_row in ema_embeddings_df.iterrows():
        ema_emb = ema_row["full_name_embedding"].reshape(1, -1)
        dists = cosine_distances(ema_emb, emb_matrix)[0]
        min_idx = np.argmin(dists)
        closest_row = embeddings_with_clusters.iloc[min_idx]
        assigned_cluster = closest_row["Final_Cluster"]
        cluster_aliases = embeddings_with_clusters[
            embeddings_with_clusters["Final_Cluster"] == assigned_cluster
        ]
        total_aliases = cluster_aliases.shape[0]
        total_occurrences = cluster_aliases["number_of_occurrences"].sum()
        n1_row = cluster_aliases.sort_values(
            "number_of_occurrences", ascending=False
        ).iloc[0]
        results.append(
            {
                "ema_full_name": ema_row["full_name"],
                "ema_object_id": ema_row["object_id"],
                "assigned_cluster": assigned_cluster,
                "distance_to_closest": dists[min_idx],
                "closest": closest_row["full_name"],
                "closest_nb_occ": closest_row["number_of_occurrences"],
                "total_aliases": total_aliases,
                "total_occurrences": total_occurrences,
                "N1_alias": n1_row["full_name"],
                "N1_alias_nb_occ": n1_row["number_of_occurrences"],
            }
        )
    results_df = pd.DataFrame(results)
    results_df.to_excel(ema_prediction_results_xlsx, index=False)
    logger.info(f"Saved EMA prediction results to {ema_prediction_results_xlsx}")

    # 6. Analysis & Metrics
    def is_noise(cluster):
        return str(cluster) == "0" or str(cluster).endswith("_0")

    results_df["is_noise"] = results_df["assigned_cluster"].apply(is_noise)
    n_total = results_df.shape[0]
    n_transformed = (~results_df["is_noise"]).sum()
    ema_transformation_rate = n_transformed / n_total
    transformed_clusters = results_df.loc[
        ~results_df["is_noise"], "assigned_cluster"
    ].unique()
    n_clusters_with_ema = len(transformed_clusters)
    cluster_counts = results_df.loc[
        ~results_df["is_noise"], "assigned_cluster"
    ].value_counts()
    multi_ema_clusters = cluster_counts[cluster_counts > 1].to_dict()
    aliases_stats = (
        results_df.loc[~results_df["is_noise"], "total_aliases"].describe().to_dict()
    )
    report = {
        "ema_transformation_rate": ema_transformation_rate,
        "n_total_ema_registries": n_total,
        "n_transformed_ema_registries": int(n_transformed),
        "n_clusters_with_ema": n_clusters_with_ema,
        "multi_ema_clusters": multi_ema_clusters,
        "aliases_stats": aliases_stats,
    }
    with open(ema_performance_report_json, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved EMA performance report to {ema_performance_report_json}")
    logger.info(
        f"EMA transformation rate: {ema_transformation_rate:.3f} ({n_transformed}/{n_total})"
    )
    logger.info(
        f"Number of clusters with at least one EMA registry: {n_clusters_with_ema}"
    )
    logger.info(f"Clusters with multiple EMA registries: {multi_ema_clusters}")
    logger.info(f"Aliases per transformed EMA registry: {aliases_stats}")
    logger.info(f"Completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
