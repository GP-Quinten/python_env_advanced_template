#!/usr/bin/env python3
import time
import json
import click
import logging
import pandas as pd
import numpy as np

from src.p05_refine_dedup import config
from src.p05_refine_dedup.utils.utils import (
    is_noise,
    apply_predictions,
    compute_metrics,
    global_clustering,
    subcluster_noise,
    subcluster_big_clusters,
    build_final_cluster_col,
)
from src.p05_refine_dedup.utils.s3_io_functions import (
    load_parquet_from_s3,
)  # for loading embeddings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@click.command()
@click.option(
    "--evaluation_dataset_any", type=str, help="Path to eval dataset (any pairs) Excel."
)
@click.option(
    "--evaluation_dataset_famous",
    type=str,
    help="Path to eval dataset (famous pairs) Excel.",
)
@click.option(
    "--best_config", type=str, help="Path to the best configuration JSON file."
)
@click.option(
    "--clusters_table_xlsx",
    type=str,
    help="Path to save clusters table (Excel).",
)
@click.option(
    "--prediction_any_results_xlsx",
    type=str,
    help="Path to save predictions (any pairs) Excel.",
)
@click.option(
    "--prediction_famous_results_xlsx",
    type=str,
    help="Path to save predictions (famous pairs) Excel.",
)
@click.option(
    "--performance_any_report_json",
    type=str,
    help="Path to save performance report (any pairs) JSON.",
)
@click.option(
    "--performance_famous_report_json",
    type=str,
    help="Path to save performance report (famous pairs) JSON.",
)
@click.option(
    "--s3_input_embeddings",
    type=str,
    help="S3 path to the registry embeddings parquet file.",
)
def main(
    evaluation_dataset_any,
    evaluation_dataset_famous,
    best_config,
    clusters_table_xlsx,
    prediction_any_results_xlsx,
    prediction_famous_results_xlsx,
    performance_any_report_json,
    performance_famous_report_json,
    s3_input_embeddings,
):
    start_time = time.time()

    # 1. Load best configuration
    with open(best_config, "r") as f:
        config_params = json.load(f)
    eps1 = config_params.get("eps1")
    eps2 = config_params.get("eps2")
    eps3 = config_params.get("eps3")
    # eps2 and eps3 available if needed later.
    logger.info(f"Loaded best config: eps1={eps1}, eps2={eps2}, eps3={eps3}")

    # 2. Load embeddings from S3 (parquet file)
    logger.info(f"Loading embeddings from {s3_input_embeddings}")
    bucket_name = config.BUCKET_NAME_DEV
    # folder_path = s3_input_embeddings, file_name = last part
    folder_path = s3_input_embeddings.rsplit("/", 1)[0]
    file_name = s3_input_embeddings.rsplit("/", 1)[-1]
    embeddings_df = load_parquet_from_s3(
        bucket_name=bucket_name,
        folder_path=folder_path,
        file_name=file_name,
    )

    # 3. Compute clustering mapping using best config
    # a. Global clustering
    logger.info(f"Performing global clustering with eps1={eps1}")
    df_global, global_time = global_clustering(embeddings_df, eps1=eps1, min_samples=2)

    # b. Subcluster noise points
    logger.info(f"Subclustering noise points with eps2={eps2}")
    df_noise, noise_time = subcluster_noise(df_global, eps2=eps2, min_samples=2)

    # c. Subcluster big clusters
    logger.info(f"Subclustering big clusters with eps3={eps3}")
    df_final, final_time = subcluster_big_clusters(
        df_noise, eps3=eps3, min_samples=2, size_threshold=5
    )

    # d. Build final cluster column
    logger.info("Building final cluster column")
    df_final = build_final_cluster_col(df_final)
    # correct noise clusters
    df_final["corrected_cluster"] = df_final["Final_Cluster"].apply(
        lambda x: None if is_noise(x) else x
    )

    # e. Create cluster_map dictionary
    cluster_map = dict(zip(df_final["full_name"], df_final["corrected_cluster"]))

    # Add this cluster mapping to the embeddings DataFrame and save it as an Excel file
    logger.info(f"Save clusters table to {clusters_table_xlsx}")
    df_final.loc[
        :, ["full_name", "number_of_occurrences", "Final_Cluster", "corrected_cluster"]
    ].to_excel(clusters_table_xlsx, index=False)

    # 4. Process evaluation for "any" dataset
    logger.info(
        f"Processing evaluation dataset (any pairs) from {evaluation_dataset_any}"
    )
    eval_df_any = pd.read_excel(evaluation_dataset_any)
    # Apply predictions based on cluster mapping
    eval_df_any = apply_predictions(
        eval_df_any, cluster_map, col_el_1="full_name", col_el_2="alias"
    )
    # Compute metrics (assuming ground truth is in column "final_label")
    metrics_any = compute_metrics(eval_df_any["final_label"], eval_df_any["prediction"])
    # log the metrics with 2 decimal precision
    metrics_any_to_print = {
        k: round(v, 2) if isinstance(v, float) else v for k, v in metrics_any.items()
    }
    logger.info(f"Metrics for any pairs: {metrics_any_to_print}")
    # Save predictions Excel file with required columns
    eval_df_any[
        [
            "full_name",
            "alias",
            "number_of_occurrences",
            "alias_number_of_occurrences",
            "similarity",
            "uncertain",
            "final_label",
            "prediction",
        ]
    ].to_excel(prediction_any_results_xlsx, index=False)
    # Save performance metrics as json
    with open(performance_any_report_json, "w") as f:
        json.dump(metrics_any, f, indent=4)

    # 5. Process evaluation for "famous" dataset
    logger.info(
        f"Processing evaluation dataset (famous pairs) from {evaluation_dataset_famous}"
    )
    eval_df_famous = pd.read_excel(evaluation_dataset_famous)
    # Apply predictions
    eval_df_famous = apply_predictions(
        eval_df_famous, cluster_map, col_el_1="full_name", col_el_2="alias"
    )
    metrics_famous = compute_metrics(
        eval_df_famous["final_label"], eval_df_famous["prediction"]
    )
    # log the metrics with 2 decimal precision
    metrics_famous_to_print = {
        k: round(v, 2) if isinstance(v, float) else v for k, v in metrics_famous.items()
    }
    logger.info(f"Metrics for famous pairs: {metrics_famous_to_print}")
    # Save predictions Excel file with required columns
    eval_df_famous[
        [
            "full_name",
            "alias",
            "number_of_occurrences",
            "alias_number_of_occurrences",
            "similarity",
            "uncertain",
            "final_label",
            "prediction",
        ]
    ].to_excel(prediction_famous_results_xlsx, index=False)
    # Save performance metrics as json
    with open(performance_famous_report_json, "w") as f:
        json.dump(metrics_famous, f, indent=4)

    total_time = time.time() - start_time
    logger.info(f"Evaluation completed in {total_time:.2f} seconds.")


if __name__ == "__main__":
    main()
