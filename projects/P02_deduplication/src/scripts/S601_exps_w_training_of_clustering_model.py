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
import plotly.express as px
from sklearn.metrics import f1_score
import plotly.graph_objects as go
import click

# Set project root and update system path for local modules
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from p02_deduplication.config import config
from p02_deduplication.logger import init_logger
from p02_deduplication.utils.utils import run_dbscan


def create_figure(results_df):
    """
    Create a single Plotly figure with F1 Score and N_registries vs. eps1
    for each eps2 value.
    """
    fig = go.Figure()

    # Define colors (one per eps2). Adjust to your preferences or use a color cycle.
    colors_map = {
        0.3: "blue",
        0.35: "orange",
        0.4: "green",
        0.45: "red",
        0.5: "purple",
    }

    # Sort eps2 so curves appear in a stable order in the legend
    unique_eps2_values = sorted(results_df["eps2"].unique())

    # Add two traces (F1, N_registries) for each eps2 value
    for eps2_val in unique_eps2_values:
        subset = results_df[results_df["eps2"] == eps2_val].sort_values("eps1")

        # --- F1-Score trace (solid line, circle marker) ---
        fig.add_trace(
            go.Scatter(
                x=subset["eps1"],
                y=subset["f1_score"],
                mode="lines+markers",
                name=f"F1 (eps2={eps2_val})",
                line=dict(color=colors_map[eps2_val], width=1, dash="solid"),
                marker=dict(color=colors_map[eps2_val], size=4, symbol="circle"),
                yaxis="y",  # Left axis
                customdata=np.stack(
                    (
                        subset["noise_percentage"] * 100,
                        subset["distinct_clusters"],
                        subset["distinct_subclusters"],
                        subset["distinct_registries"],
                    ),
                    axis=-1,
                ),
                hovertemplate=(
                    "eps1: %{x:.2f}<br>"
                    "F1: %{y:.2f}<br>"
                    "Noise: %{customdata[0]:.2f}%<br>"
                    "Clusters: %{customdata[1]}<br>"
                    "Subclusters: %{customdata[2]}<br>"
                    "Registries: %{customdata[3]}"
                    "<extra></extra>"
                ),
            )
        )

        # --- N_registries trace (dotted line, cross marker) ---
        fig.add_trace(
            go.Scatter(
                x=subset["eps1"],
                y=subset["distinct_registries"],
                mode="lines+markers",
                name=f"N_registries (eps2={eps2_val})",
                line=dict(color=colors_map[eps2_val], width=1, dash="dot"),
                marker=dict(color=colors_map[eps2_val], size=4, symbol="x"),
                yaxis="y2",  # Right axis
                customdata=np.stack(
                    (
                        subset["f1_score"],
                        subset["distinct_clusters"],
                        subset["noise_percentage"] * 100,
                        subset["distinct_subclusters"],
                    ),
                    axis=-1,
                ),
                hovertemplate=(
                    "eps1: %{x:.2f}<br>"
                    "F1: %{customdata[0]:.2f}<br>"
                    "Clusters: %{customdata[1]}<br>"
                    "Noise: %{customdata[2]:.2f}%<br>"
                    "Subclusters: %{customdata[3]}<br>"
                    "Registries: %{y}"
                    "<extra></extra>"
                ),
            )
        )

    # Layout: Two Y-axes
    fig.update_layout(
        title="DBSCAN Performance vs. eps1 (varying eps2)",
        xaxis=dict(title="eps1"),
        yaxis=dict(
            title="F1 Score",
        ),
        yaxis2=dict(
            title="N° of distinct registries",
            overlaying="y",
            side="right",
        ),
        legend=dict(x=0.01, y=0.99),
    )

    return fig


@click.command()
@click.option(
    "--training_dataset",
    type=str,
    help="Path to the input parquet file.",
)
@click.option(
    "--evaluation_dataset",
    type=str,
    help="Path to the input parquet file.",
)
@click.option(
    "--output_dir",
    type=str,
    help="Directory to save the output files.",
)
def main(training_dataset, evaluation_dataset, output_dir):
    """
    Main function to run the clustering model training and evaluation.
    """
    # initialize time
    start_time = time.time()

    # Initialize logger
    init_logger(level="WARNING")

    # Print current working directory
    print(Path.cwd())

    # ---- LOAD DATA ---- #
    df = pd.read_parquet(training_dataset)
    df_eval = pd.read_parquet(evaluation_dataset)

    # ---- PREPROCESS DATA ---- #
    # only keep rows where 'to_include' is 'Yes'
    df_eval = df_eval[df_eval["to_include"] == "Yes"].copy()
    df_eval.rename(columns={"final_label": "LLM_label"}, inplace=True)

    # Prepare embeddings
    all_embeddings = np.vstack(df["embedding"].values)
    total_count = len(df)

    # ---- DEFINE RANGES FOR eps1 and eps2 ---- #
    eps1_values = np.arange(0.25, 0.51, 0.01)  # step of 0.01
    eps2_values = [0.30, 0.35, 0.40, 0.45, 0.50]

    results = []

    # Loop over eps2 for noise/big clusters subclustering
    for eps2 in eps2_values:
        # Loop over eps1 for main DBSCAN
        for eps1 in eps1_values:
            logging.warning(
                f"Main clustering with eps1={round(eps1, 2)}, subclustering with eps2={eps2}..."
            )

            # Copy df for each iteration so we don't overwrite
            df_temp = df.copy()

            # Main DBSCAN clustering
            labels, comp_time = run_dbscan(all_embeddings, eps1, min_samples=2)
            df_temp["cluster"] = labels
            df_temp["Subcluster"] = None

            # Count distinct clusters
            distinct_clusters = df_temp["cluster"].nunique()

            # ---- NOISE SUBCLUSTERING (using eps2) ---- #
            df_noise = df_temp[df_temp["cluster"] == 0]
            if not df_noise.empty:
                embeddings_noise = np.vstack(df_noise["embedding"].values)
                logging.warning("Computing sub-clustering on noise cluster...")
                sub_labels, sub_time = run_dbscan(
                    embeddings_noise, eps=eps2, min_samples=2
                )
                df_temp.loc[df_temp["cluster"] == 0, "Subcluster"] = sub_labels
            else:
                logging.warning("No noise cluster found for sub-clustering.")

            # ---- SUBCLUSTERING on BIG CLUSTERS (also using eps2) ---- #
            logging.warning("Subclustering on big clusters...")
            clusters_unique = df_temp["cluster"].unique()
            clusters_to_subcluster = [
                c
                for c in clusters_unique
                if c != 0 and (df_temp["cluster"] == c).sum() >= 10
            ]
            for cluster_label in clusters_to_subcluster:
                df_cluster = df_temp[df_temp["cluster"] == cluster_label]
                embeddings_cluster = np.vstack(df_cluster["embedding"].values)
                sub_labels, _ = run_dbscan(embeddings_cluster, eps=eps2, min_samples=2)
                df_temp.loc[df_temp["cluster"] == cluster_label, "Subcluster"] = (
                    sub_labels
                )

            # Final_Cluster = cluster_Subcluster
            df_temp["Final_Cluster"] = (
                df_temp["cluster"].astype(str) + "_" + df_temp["Subcluster"].astype(str)
            )
            distinct_subclusters = df_temp["Final_Cluster"].nunique()

            # Count total noise in final clusters (ends with '_0')
            noise_count = (df_temp["Final_Cluster"].str.endswith("_0")).sum()
            noise_percentage = noise_count / total_count

            # Number of distinct registries
            N_distinct_registries = noise_count + distinct_subclusters

            # ---- EVALUATION on df_eval ---- #
            for idx, row in df_eval.iterrows():
                ds1 = row["data_source_name_1"]
                ds2 = row["data_source_name_2"]
                cluster_ds1 = df_temp.loc[
                    df_temp["data_source_name"] == ds1, "Final_Cluster"
                ].values[0]
                cluster_ds2 = df_temp.loc[
                    df_temp["data_source_name"] == ds2, "Final_Cluster"
                ].values[0]
                df_eval.at[idx, "cluster_match_label"] = (
                    1 if cluster_ds1 == cluster_ds2 else 0
                )

            F1 = f1_score(df_eval["LLM_label"], df_eval["cluster_match_label"])
            logging.warning(f"eps1={eps1}, eps2={eps2} => F1-score={F1:.2f}")

            results.append(
                {
                    "eps1": eps1,
                    "eps2": eps2,
                    "distinct_clusters": distinct_clusters,
                    "distinct_subclusters": distinct_subclusters,
                    "distinct_registries": N_distinct_registries,
                    "noise_percentage": noise_percentage,
                    "f1_score": F1,
                }
            )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    results_df.to_excel(
        os.path.join(output_dir, "clustering_results.xlsx"),
        index=False,
    )

    # Create and save the figure
    fig = create_figure(results_df)
    fig.write_html(
        os.path.join(output_dir, "clustering_results.html"),
        include_plotlyjs="cdn",
    )

    # End time
    end_time = time.time()
    logging.warning(f"Total time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
