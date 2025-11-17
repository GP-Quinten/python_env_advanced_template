#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import logging
import time
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
from sklearn.metrics import f1_score

# Set project root and update system path for local modules
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from config import config
from src.logger import init_logger
from src.utils.io_functions import load_file_from_s3, upload_file_to_s3
from src.utils.utils import run_dbscan


def evaluate_model(df, df_eval, eps1, eps2, eps3):
    """
    Run the complete clustering procedure using eps1, eps2, eps3 and evaluate performance.
    Returns a dictionary of metrics.
    """
    min_samples = 2
    # Work on a copy of df to avoid side effects between iterations.
    df_local = df.copy(deep=True)
    embeddings = np.vstack(df_local["embedding"].values)

    # --- Initial Clustering (using eps1) ---
    labels, time_initial = run_dbscan(embeddings, eps=eps1, min_samples=min_samples)
    df_local["cluster"] = labels
    df_local["Subcluster"] = None

    total_count = len(df_local)
    distinct_clusters = df_local["cluster"].nunique()
    noise_count = (df_local["cluster"] == 0).sum()

    # --- Noise Subclustering (using eps2) ---
    df_noise = df_local[df_local["cluster"] == 0]
    noise_sub_time = 0
    if not df_noise.empty:
        embeddings_noise = np.vstack(df_noise["embedding"].values)
        sub_labels, noise_sub_time = run_dbscan(
            embeddings_noise, eps=eps2, min_samples=min_samples
        )
        df_local.loc[df_local["cluster"] == 0, "Subcluster"] = sub_labels
    else:
        logging.warning("No noise cluster found for subclustering.")

    # --- Subclustering on BIG CLUSTERS (using eps3) ---
    big_cluster_sub_time = 0
    clusters = df_local["cluster"].unique()
    clusters_to_subcluster = [
        c for c in clusters if c != 0 and (df_local["cluster"] == c).sum() >= 10
    ]
    for cluster_label in clusters_to_subcluster:
        df_cluster = df_local[df_local["cluster"] == cluster_label]
        embeddings_cluster = np.vstack(df_cluster["embedding"].values)
        sub_labels, comp_time = run_dbscan(
            embeddings_cluster, eps=eps3, min_samples=min_samples
        )
        df_local.loc[df_local["cluster"] == cluster_label, "Subcluster"] = sub_labels
        big_cluster_sub_time += comp_time

    # Create Final_Cluster as a combination of cluster and Subcluster
    df_local["Final_Cluster"] = (
        df_local["cluster"].astype(str) + "_" + df_local["Subcluster"].astype(str)
    )
    distinct_subclusters = df_local["Final_Cluster"].nunique()
    noise_count_final = (df_local["Final_Cluster"].str.endswith("_0")).sum()
    N_distinct_registries = noise_count_final + distinct_subclusters

    # --- EVALUATION on evaluation dataset ---
    for idx, row in df_eval.iterrows():
        ds1 = row["data_source_name_1"]
        ds2 = row["data_source_name_2"]
        # It is assumed that each ds is present exactly once in df_local.
        cluster_ds1 = df_local[df_local["data_source_name"] == ds1][
            "Final_Cluster"
        ].values[0]
        cluster_ds2 = df_local[df_local["data_source_name"] == ds2][
            "Final_Cluster"
        ].values[0]
        df_eval.at[idx, "cluster_match_label"] = 1 if cluster_ds1 == cluster_ds2 else 0

    current_f1 = f1_score(df_eval["LLM_label"], df_eval["cluster_match_label"])
    total_time = time_initial + noise_sub_time + big_cluster_sub_time

    metrics = {
        "eps1": eps1,
        "eps2": eps2,
        "eps3": eps3,
        "distinct_clusters": distinct_clusters,
        "distinct_subclusters": distinct_subclusters,
        "distinct_registries": N_distinct_registries,
        "Isolated registries": noise_count_final,
        "noise_percentage": noise_count / total_count,
        "f1_score": current_f1,
        "computation_time": total_time,
    }
    return metrics, df_local


def create_figure(metrics_history):
    """
    Create a plotly figure tracking the performance over iterations.
    X-axis: iteration number.
    Left Y-axis: F1 score
    Right Y-axis: noise percentage
    """
    df_metrics = pd.DataFrame(metrics_history)
    iterations = df_metrics.index + 1

    fig = go.Figure()
    # Trace for F1 Score on left axis
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=df_metrics["f1_score"],
            mode="lines+markers",
            name="F1 Score",
            yaxis="y",
            hovertemplate="Iteration: %{x}<br>F1 Score: %{y:.2f}<extra></extra>",
        )
    )
    # Trace for noise percentage on right axis
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=df_metrics["noise_percentage"],
            mode="lines+markers",
            name="Noise %",
            yaxis="y2",
            hovertemplate="Iteration: %{x}<br>Noise %: %{y:.0%}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Optimization Progress over Iterations",
        xaxis=dict(title="Iteration"),
        yaxis=dict(title="F1 Score", color="blue", range=[0.65, 0.9]),
        yaxis2=dict(
            title="% Of isolated registry names<br>(only 1 occurrence in all Publications)",
            overlaying="y",
            side="right",
            color="red",
            # Use tickformat to display as percentages
            tickformat=".0%",
            range=[0.3, 0.7],
        ),
    )
    return fig


def main():
    # track time computation
    start_time = time.time()
    # Initialize logger
    init_logger(
        stream=True,
        file=True,
        file_path="logs/optimized_training_logs.txt",
        level="WARNING",
    )
    print(Path.cwd())

    # ---- LOAD DATA ---- #
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
    df_eval = df_eval[
        ~(
            df_eval["explanation"].str.startswith("Inclusion-uncertainty")
            & (df_eval["diff"] != 0)
        )
    ]
    df_eval.rename(columns={"final_label": "LLM_label"}, inplace=True)

    # ---- OPTIMIZATION CONFIGURATION ---- #
    # Allowed ranges for epsilons:
    eps1_min, eps1_max = 0.25, 0.40
    eps2_min, eps2_max = 0.28, 0.55
    eps3_min, eps3_max = 0.20, 0.37

    init_eps1 = 0.348  # 0.3314610779075241  # eps1_min  #
    init_eps2 = 0.400  # 0.380396662361368  # eps2_min  #
    init_eps3 = 0.275  # 0.275073319059974  # eps3_min  #

    # Initial parameter values
    params = np.array([init_eps1, init_eps2, init_eps3])
    # Adam hyperparameters
    lr = 0.05
    beta1 = 0.9
    beta2 = 0.99
    adam_epsilon = 1e-8
    m = np.zeros(3)
    v = np.zeros(3)

    max_iter = 50
    max_no_improve = 10
    delta = 0.001  # finite difference step for gradient estimation
    best_f1 = -1
    best_params = params.copy()
    best_metrics = {}
    no_improve_count = 0  # counter for early stopping
    metrics_history = []

    # ---- OPTIMIZATION LOOP ---- #
    for t in range(max_iter):
        logging.warning(
            f"Iteration {t+1}/{max_iter} with params: eps1={params[0]:.3f}, eps2={params[1]:.3f}, eps3={params[2]:.3f}"
        )
        # Evaluate current configuration
        logging.warning("Evaluating with current configuration...")
        metrics, df_w_clusters = evaluate_model(
            df, df_eval.copy(), params[0], params[1], params[2]
        )
        current_f1 = metrics["f1_score"]
        logging.warning(f"F1 score: {current_f1:.3f}")
        metrics_history.append(metrics)

        if current_f1 < best_f1:
            no_improve_count += 1
        else:
            no_improve_count = 0
            best_f1 = current_f1
            best_params = params.copy()
            best_metrics = metrics.copy()
            df_best_config = df_w_clusters.copy()

        # Early stopping if no significant progress in 3 consecutive iterations
        if no_improve_count >= max_no_improve:
            logging.warning(
                "Early stopping: no significant improvement in 3 consecutive iterations."
            )
            break

        # Compute loss as negative F1 (we want to maximize F1)
        loss = -current_f1

        # Estimate gradient via finite differences
        logging.warning("Estimating gradient by disturbing each parameter...")
        grad = np.zeros(3)
        for i in range(3):
            logging.warning(f"Estimating gradient for eps {i+1}...")
            params_perturbed = params.copy()
            params_perturbed[i] += delta
            metrics_perturbed, _ = evaluate_model(
                df,
                df_eval.copy(),
                params_perturbed[0],
                params_perturbed[1],
                params_perturbed[2],
            )
            loss_perturbed = -metrics_perturbed["f1_score"]
            grad[i] = (loss_perturbed - loss) / delta

        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)
        m_hat = m / (1 - beta1 ** (t + 1))
        v_hat = v / (1 - beta2 ** (t + 1))
        params = params - lr * m_hat / (np.sqrt(v_hat) + adam_epsilon)

        # Clip the parameters to their allowed ranges
        params[0] = np.clip(params[0], eps1_min, eps1_max)
        params[1] = np.clip(params[1], eps2_min, eps2_max)
        params[2] = np.clip(params[2], eps3_min, eps3_max)

    # Log best configuration found
    logging.warning("Optimization finished.")
    logging.warning(
        f"Best parameters: eps1={best_params[0]:.3f}, eps2={best_params[1]:.3f}, eps3={best_params[2]:.3f}"
    )
    logging.warning(f"Best F1 score: {best_f1:.3f}")

    # Save best model parameters and metrics
    best_model = {
        "eps1": best_params[0],
        "eps2": best_params[1],
        "eps3": best_params[2],
        "f1_score": best_f1,
        "distinct_clusters": best_metrics.get("distinct_clusters", None),
        "distinct_subclusters": best_metrics.get("distinct_subclusters", None),
        "distinct_registries": best_metrics.get("distinct_registries", None),
        "Isolated registries": best_metrics.get("Isolated registries", None),
        "noise_percentage": best_metrics.get("noise_percentage", None),
        "computation_time": best_metrics.get("computation_time", None),
    }
    logging.warning("Best model details:")
    for key, value in best_model.items():
        logging.warning(f"{key}: {value}")

    best_model_df = pd.DataFrame([best_model])
    best_model_df.to_excel("data/6_find_best_model/best_model_v2_all.xlsx", index=False)

    # Save df_clusters with best configuration to s3
    upload_file_to_s3(
        df_best_config,
        bucket_name=config.BUCKET_NAME_DEV,
        folder_path="registry_data_catalog_experiments/task_1_deduplication/publications_clusters/",
        file_name="data_source_names_clusters_v3.parquet",
    )

    # ---- SAVE METRICS HISTORY & PLOT PERFORMANCE ---- #
    history_df = pd.DataFrame(metrics_history)
    history_df.to_excel(
        "data/6_find_best_model/optimization_history_v3_all.xlsx", index=False
    )

    fig = create_figure(metrics_history)
    fig.write_html("data/6_find_best_model/optimization_progress_v3_all.html")

    # save table
    # track time computation
    end_time = time.time()
    logging.warning(f"Total computation time: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
