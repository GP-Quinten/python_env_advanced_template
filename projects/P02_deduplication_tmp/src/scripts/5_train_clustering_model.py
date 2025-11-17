#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# from dotenv import load_dotenv
import logging
import time
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import f1_score, precision_score, recall_score
import plotly.express as px
import plotly.graph_objects as go

# Set project root and update system path for local modules
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from config import config
from src.logger import init_logger
from src.utils.io_functions import load_file_from_s3, upload_file_to_s3
from src.utils.utils import run_dbscan


def create_figure(results_df):
    """
    Create a plotly figure with two y-axes:
      - Left y-axis: performance metrics (F1, Precision, Recall)
      - Right y-axis: registry counts (Distinct Registries, Isolated Registries, Distinct Clusters)

    Global clustering traces use dotted lines with crosses and belong to the legend group
    "Results after Global Clustering", while sub-clustering traces use solid lines with markers
    and belong to the legend group "Results after Sub-Clustering".
    """
    fig = go.Figure()

    # Define colors
    performance_colors = {
        "F1": "darkblue",
        "Precision": "darkgreen",
        "Recall": "darkred",
    }
    counts_colors = {
        "Distinct Registries": "lightblue",
        "Isolated Registries": "lightgreen",
        "Distinct Clusters": "lightcoral",
    }

    # --------------------- Global Clustering Traces ---------------------
    # Performance Metrics - Global (Left y-axis, dotted lines with crosses)
    fig.add_trace(
        go.Scatter(
            x=results_df["eps"],
            y=results_df["global_f1"],
            mode="lines+markers",
            line=dict(dash="dot", color=performance_colors["F1"]),
            marker=dict(symbol="cross"),
            name="Global F1 Score",
            legendgroup="Results after Global Clustering",
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=results_df["eps"],
            y=results_df["global_precision"],
            mode="lines+markers",
            line=dict(dash="dot", color=performance_colors["Precision"]),
            marker=dict(symbol="cross"),
            name="Global Precision",
            legendgroup="Results after Global Clustering",
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=results_df["eps"],
            y=results_df["global_recall"],
            mode="lines+markers",
            line=dict(dash="dot", color=performance_colors["Recall"]),
            marker=dict(symbol="cross"),
            name="Global Recall",
            legendgroup="Results after Global Clustering",
            yaxis="y",
        )
    )

    # Count Metrics - Global (Right y-axis, dotted lines with crosses)
    fig.add_trace(
        go.Scatter(
            x=results_df["eps"],
            y=results_df["global_registries"],
            mode="lines+markers",
            line=dict(dash="dot", color=counts_colors["Distinct Registries"]),
            marker=dict(symbol="cross"),
            name="Global Distinct Registries",
            legendgroup="Results after Global Clustering",
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=results_df["eps"],
            y=results_df["global_isolated"],
            mode="lines+markers",
            line=dict(dash="dot", color=counts_colors["Isolated Registries"]),
            marker=dict(symbol="cross"),
            name="Global Isolated Registries",
            legendgroup="Results after Global Clustering",
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=results_df["eps"],
            y=results_df["global_clusters"],
            mode="lines+markers",
            line=dict(dash="dot", color=counts_colors["Distinct Clusters"]),
            marker=dict(symbol="cross"),
            name="Global Distinct Clusters",
            legendgroup="Results after Global Clustering",
            yaxis="y2",
        )
    )

    # --------------------- Sub-Clustering Traces ---------------------
    # Performance Metrics - Sub-Clustering (Left y-axis, solid lines with markers)
    fig.add_trace(
        go.Scatter(
            x=results_df["eps"],
            y=results_df["sub_f1"],
            mode="lines+markers",
            line=dict(dash="solid", color=performance_colors["F1"]),
            name="Sub F1 Score",
            legendgroup="Results after Sub-Clustering",
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=results_df["eps"],
            y=results_df["sub_precision"],
            mode="lines+markers",
            line=dict(dash="solid", color=performance_colors["Precision"]),
            name="Sub Precision",
            legendgroup="Results after Sub-Clustering",
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=results_df["eps"],
            y=results_df["sub_recall"],
            mode="lines+markers",
            line=dict(dash="solid", color=performance_colors["Recall"]),
            name="Sub Recall",
            legendgroup="Results after Sub-Clustering",
            yaxis="y",
        )
    )

    # Count Metrics - Sub-Clustering (Right y-axis, solid lines with markers)
    fig.add_trace(
        go.Scatter(
            x=results_df["eps"],
            y=results_df["sub_registries"],
            mode="lines+markers",
            line=dict(dash="solid", color=counts_colors["Distinct Registries"]),
            name="Sub Distinct Registries",
            legendgroup="Results after Sub-Clustering",
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=results_df["eps"],
            y=results_df["sub_isolated"],
            mode="lines+markers",
            line=dict(dash="solid", color=counts_colors["Isolated Registries"]),
            name="Sub Isolated Registries",
            legendgroup="Results after Sub-Clustering",
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=results_df["eps"],
            y=results_df["sub_clusters"],
            mode="lines+markers",
            line=dict(dash="solid", color=counts_colors["Distinct Clusters"]),
            name="Sub Distinct Clusters",
            legendgroup="Results after Sub-Clustering",
            yaxis="y2",
        )
    )

    # --------------------- Layout ---------------------
    fig.update_layout(
        title="DBSCAN Performance vs. Epsilon",
        xaxis=dict(title="Epsilon"),
        # Left y-axis for performance metrics
        yaxis=dict(title="Performance Metrics (F1, Precision, Recall)", color="black"),
        # Right y-axis for count metrics
        yaxis2=dict(
            title="Registry Counts (Distinct Registries, Isolated, Clusters)",
            overlaying="y",
            side="right",
            color="black",
        ),
        legend=dict(x=1.05, y=1),
    )
    return fig


def main():
    # initialize time
    start_time = time.time()
    # Initialize logger using the custom logger initialization function
    init_logger(
        stream=True, file=True, file_path="logs/training_logs.txt", level="WARNING"
    )

    # Print current working directory
    print(Path.cwd())

    # ---- LOAD DATA ---- #
    # Load table from S3 with columns: 'data_source_name' and 'embedding'
    df = load_file_from_s3(
        bucket_name=config.BUCKET_NAME_DEV,
        folder_path="registry_data_catalog_experiments/task_1_deduplication/publications_embeddings/",
        file_name="data_source_name_MistralEmbed_embeddings_all.parquet",
    )
    df_eval = load_file_from_s3(
        bucket_name=config.BUCKET_NAME_DEV,
        folder_path="registry_data_catalog_experiments/task_1_deduplication/publications_final_pairs/",
        file_name="final_pairs_v3.parquet",
    )

    # ---- PREPROCESS DATA ---- #
    # In df_eval, remove rows where 'explanation' starts with 'Inclusion-uncertainty' and 'diff' is different from 0
    df_eval = df_eval[
        ~(
            df_eval["explanation"].str.startswith("Inclusion-uncertainty")
            & (df_eval["diff"] != 0)
        )
    ]
    # Rename 'final_label' to 'LLM_label'
    df_eval.rename(columns={"final_label": "LLM_label"}, inplace=True)

    # ---- CLUSTERING TRAINING ---- #
    # initialize a list of epsilons from 0.25 to 0.45 with a step of 0.001
    epsilons = np.arange(0.25, 0.45, 0.001)
    # track performance and count metrics for each epsilon in a dataframe
    results = []
    for eps in epsilons:
        # ---- GLOBAL CLUSTERING ---- #
        # Apply initial DBSCAN clustering on the entire dataset
        embeddings = np.vstack(df["embedding"].values)
        min_samples = 2
        logging.warning(f"Computing initial clustering with eps={round(eps, 3)}...")
        labels, comp_time = run_dbscan(embeddings, eps, min_samples)
        df["cluster"] = labels
        # Reset Subcluster column (for later use)
        df["Subcluster"] = None

        # ---- Global Clustering Evaluation ---- #
        # For each pair in df_eval, compare global clustering labels
        for idx, row in df_eval.iterrows():
            ds1 = row["data_source_name_1"]
            ds2 = row["data_source_name_2"]
            try:
                global_cluster_ds1 = df[df["data_source_name"] == ds1][
                    "cluster"
                ].values[0]
                global_cluster_ds2 = df[df["data_source_name"] == ds2][
                    "cluster"
                ].values[0]
            except IndexError:
                global_cluster_ds1 = None
                global_cluster_ds2 = None
            df_eval.at[idx, "global_cluster_match_label"] = (
                1 if global_cluster_ds1 == global_cluster_ds2 else 0
            )

        global_f1 = f1_score(
            df_eval["LLM_label"], df_eval["global_cluster_match_label"]
        )
        global_precision = precision_score(
            df_eval["LLM_label"], df_eval["global_cluster_match_label"]
        )
        global_recall = recall_score(
            df_eval["LLM_label"], df_eval["global_cluster_match_label"]
        )

        # Global count metrics
        global_isolated = (df["cluster"] == 0).sum()
        global_distinct_clusters = len(set(df["cluster"]) - {0})
        global_registries = global_isolated + global_distinct_clusters

        # ---- NOISE SUBCLUSTERING ---- #
        df_noise = df[df["cluster"] == 0]
        if not df_noise.empty:
            embeddings_noise = np.vstack(df_noise["embedding"].values)
            logging.warning("Computing sub-clustering on noise cluster (cluster 0)...")
            sub_labels, sub_time = run_dbscan(embeddings_noise, eps=0.45, min_samples=2)
            df.loc[df["cluster"] == 0, "Subcluster"] = sub_labels
        else:
            logging.warning("No noise cluster (cluster 0) found for sub-clustering.")

        # ---- SUBCLUSTERING on BIG CLUSTERS ---- #
        logging.warning("Subclustering on big clusters...")
        clusters = df["cluster"].unique()
        clusters_to_subcluster = [
            c for c in clusters if c != 0 and (df["cluster"] == c).sum() >= 10
        ]
        for idx_cluster, cluster_label in enumerate(clusters_to_subcluster, start=1):
            df_cluster = df[df["cluster"] == cluster_label]
            embeddings_cluster = np.vstack(df_cluster["embedding"].values)
            sub_labels, comp_time = run_dbscan(
                embeddings_cluster, eps=0.28, min_samples=2
            )
            df.loc[df["cluster"] == cluster_label, "Subcluster"] = sub_labels

        # Create Final_Cluster column as a combination of cluster and Subcluster
        df["Final_Cluster"] = (
            df["cluster"].astype(str) + "_" + df["Subcluster"].astype(str)
        )

        # ---- Sub-Clustering Evaluation ---- #
        # For each pair in df_eval, compare sub-clustering labels (Final_Cluster)
        for idx, row in df_eval.iterrows():
            ds1 = row["data_source_name_1"]
            ds2 = row["data_source_name_2"]
            try:
                sub_cluster_ds1 = df[df["data_source_name"] == ds1][
                    "Final_Cluster"
                ].values[0]
                sub_cluster_ds2 = df[df["data_source_name"] == ds2][
                    "Final_Cluster"
                ].values[0]
            except IndexError:
                sub_cluster_ds1 = None
                sub_cluster_ds2 = None
            df_eval.at[idx, "sub_cluster_match_label"] = (
                1 if sub_cluster_ds1 == sub_cluster_ds2 else 0
            )

        sub_f1 = f1_score(df_eval["LLM_label"], df_eval["sub_cluster_match_label"])
        sub_precision = precision_score(
            df_eval["LLM_label"], df_eval["sub_cluster_match_label"]
        )
        sub_recall = recall_score(
            df_eval["LLM_label"], df_eval["sub_cluster_match_label"]
        )

        # Sub-clustering count metrics:
        # isolated: count rows where Final_Cluster ends with "_0"
        sub_isolated = df["Final_Cluster"].astype(str).str.endswith("_0").sum()
        # distinct clusters: unique Final_Cluster values that do not end with "_0"
        unique_final = df["Final_Cluster"].unique()
        sub_distinct_clusters = len(
            [x for x in unique_final if not str(x).endswith("_0")]
        )
        sub_registries = sub_isolated + sub_distinct_clusters

        # ---- STORE METRICS ---- #
        results.append(
            {
                "eps": eps,
                "global_f1": global_f1,
                "global_precision": global_precision,
                "global_recall": global_recall,
                "global_isolated": global_isolated,
                "global_clusters": global_distinct_clusters,
                "global_registries": global_registries,
                "sub_f1": sub_f1,
                "sub_precision": sub_precision,
                "sub_recall": sub_recall,
                "sub_isolated": sub_isolated,
                "sub_clusters": sub_distinct_clusters,
                "sub_registries": sub_registries,
            }
        )

        logging.warning(
            f"EPS: {round(eps,3)} | Global F1: {round(global_f1,3)}, Sub F1: {round(sub_f1,3)}"
        )

    # ---- SAVE RESULTS ---- #
    results_df = pd.DataFrame(results)
    results_df.to_excel(
        "data/5_clustering_perf/train_clustering_model_results_v6_all.xlsx", index=False
    )
    fig = create_figure(results_df)
    fig.write_html("data/5_clustering_perf/clustering_performance_metrics_v6_all.html")

    # end time
    end_time = time.time()
    logging.warning(f"Total time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
