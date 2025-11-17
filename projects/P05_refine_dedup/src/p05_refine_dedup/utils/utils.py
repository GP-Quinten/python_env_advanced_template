import time
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd
import numpy as np
import logging
import umap
import os

logger = logging.getLogger(__name__)


def is_noise(cluster):
    return str(cluster) == "0" or str(cluster).endswith("_0")


def run_dbscan(embeddings, eps, min_samples):
    """
    Run DBSCAN clustering on the provided embeddings.
    Increments labels by 1 so that noise (-1) becomes 0.
    Returns the labels and the computation time.
    """
    cpu_count = os.cpu_count()
    logger.debug(f"Number of CPUs available: {cpu_count}. Using n_jobs=32 for DBSCAN.")
    start_time = time.time()
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=cpu_count)
    labels = dbscan.fit_predict(embeddings)
    labels = labels + 1  # convert noise (-1) to 0, clusters start at 1
    comp_time = time.time() - start_time
    logger.info(
        f"DBSCAN completed in {comp_time:.2f} seconds with eps={eps}, min_samples={min_samples}."
    )
    return labels, comp_time


def run_hdbscan(
    embeddings,
    min_cluster_size=2,
    min_samples=2,
    cluster_selection_epsilon=0.0,
    max_cluster_size=30,
    metric="euclidean",
    n_jobs=-1,
    cluster_selection_method="leaf",
    store_centers="medoid",
):
    """
    Run HDBSCAN clustering on the provided embeddings.
    Increments labels by 1 so that noise (-1) becomes 0.
    Returns the labels and the computation time.
    """
    start_time = time.time()
    hdbscan_cluster = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        max_cluster_size=max_cluster_size,
        metric=metric,
        n_jobs=n_jobs,
        cluster_selection_method=cluster_selection_method,
        store_centers=store_centers,
    )
    labels = hdbscan_cluster.fit_predict(embeddings)
    labels = labels + 1  # convert noise (-1) to 0, clusters start at 1
    comp_time = time.time() - start_time
    logger.info(
        f"HDBSCAN completed in {comp_time:.2f} seconds with min_cluster_size={min_cluster_size}, min_samples={min_samples}."
    )
    return labels, comp_time


def umap_reduction(
    embeddings, n_neighbors=10, min_dist=0.1, n_components=2, random_state=None
):
    """
    Performs dimensionality reduction on the given embeddings using UMAP.
    """
    cpu_count = os.cpu_count()
    logger.debug(f"Number of CPUs available: {cpu_count}. Using n_jobs=32 for UMAP.")
    reducer = umap.UMAP(
        metric="cosine",
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
        n_jobs=cpu_count,
    )
    return reducer.fit_transform(embeddings)


def global_clustering(df, eps1, min_samples=2, col_embedding="full_name_embedding"):
    """
    Run DBSCAN once for the entire dataset with eps1.
    Return (df_global, global_time).
    df_global['cluster'] has labels (0 for noise, 1.. for clusters).
    """
    df_local = df.copy(deep=True)
    embeddings = np.vstack(df_local[col_embedding].values)
    labels, comp_time = run_dbscan(embeddings, eps=eps1, min_samples=min_samples)
    df_local["cluster"] = labels
    return df_local, comp_time


def subcluster_noise(
    df_global, eps2, min_samples=2, col_embedding="full_name_embedding"
):
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

    embeddings_noise = np.vstack(df_local.loc[noise_mask, col_embedding])
    sub_labels, comp_time = run_dbscan(
        embeddings_noise, eps=eps2, min_samples=min_samples
    )
    df_local.loc[noise_mask, "Subcluster"] = sub_labels
    return df_local, comp_time


def subcluster_big_clusters(
    df_noise, eps3, min_samples=2, size_threshold=5, col_embedding="full_name_embedding"
):
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
        embeddings_cluster = np.vstack(df_local.loc[mask, col_embedding])
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


def apply_predictions(
    evaluation_df, cluster_map, col_el_1="full_name", col_el_2="alias"
):
    """
    For each row in evaluation_df, determine prediction as 1 if 'full_name'
    and 'alias' share the same cluster label (from cluster_map), or 0 otherwise.
    Returns the updated dataframe.
    Assumes evaluation_df has a column "final_label" for ground-truth.
    """
    predictions = []
    for idx, row in evaluation_df.iterrows():
        fn = row[col_el_1]
        al = row[col_el_2]
        label1 = cluster_map.get(fn, None)
        label2 = cluster_map.get(al, None)
        pred = (
            1 if (label1 is not None and label2 is not None and label1 == label2) else 0
        )
        predictions.append(pred)
    evaluation_df["prediction"] = predictions
    return evaluation_df


def compute_metrics(y_true, y_pred):
    """
    Compute precision, recall, f1 and accuracy.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }
    return metrics
