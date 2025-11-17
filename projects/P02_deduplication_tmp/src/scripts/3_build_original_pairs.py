#!/usr/bin/env python3
import sys
from pathlib import Path

# Set project root and update system path for local modules
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import logging
import time
import numpy as np
import pandas as pd
from src.logger import init_logger
from config import config
from src.utils.io_functions import load_file_from_s3, upload_file_to_s3


# Helper function to compute cosine similarity between two vectors.
def compute_cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def main():
    # Initialize logger (iteration details at INFO, totals at WARNING)
    init_logger(stream=True, file=True, file_path="logs/logs.txt", level="WARNING")

    # Load the clustered table from S3 (assumed file from first script)
    df = load_file_from_s3(
        bucket_name=config.BUCKET_NAME_DEV,
        folder_path="registry_data_catalog_experiments/task_1_deduplication/publications_clusters/",
        file_name="data_source_names_clusters_v1.parquet",
    )

    total_rows = len(df)
    distinct_final_clusters = df["Final_Cluster"].nunique()
    no_cluster_count = df["Final_Cluster"].str.endswith("0").sum()
    percentage_no_cluster = (no_cluster_count / total_rows) * 100
    logging.warning(f"Distinct final clusters: {distinct_final_clusters}")
    logging.warning(
        f"Percentage with no cluster (Final_Cluster ends with '0') -> To remove: {percentage_no_cluster:.2f}%"
    )

    # Remove rows that don't have a final cluster (Final_Cluster ends with "0")
    df_no_noise = df[~df["Final_Cluster"].str.endswith("0")].reset_index(drop=True)
    total_clusters = len(df_no_noise)

    # Initialize an empty DataFrame for pairs
    df_pairs = pd.DataFrame(
        columns=["data_source_name_1", "data_source_name_2", "label", "category"]
    )

    # logging -> start adding positive and negative pairs from Final_Clusters
    logging.warning("Adding positive and negative pairs from Final_Clusters ...")
    # Counters for pairs generated per branch
    pairs_2_positive = 0
    pairs_3to10_positive = 0
    pairs_3to10_negative = 0
    pairs_10to50_positive = 0
    pairs_10to50_negative = 0
    pairs_over50_positive = 0
    pairs_over50_negative = 0

    # Process each cluster (group by Final_Cluster)
    for final_cluster, group in df_no_noise.groupby("Final_Cluster"):
        cluster_size = len(group)
        logging.info(
            f"Processing cluster {final_cluster}/{total_clusters} with size {cluster_size}"
        )

        # Branch 1: Cluster size == 2
        if cluster_size == 2:
            anchor = group.sample(n=1)
            anchor_index = anchor.index[0]
            candidate = group.drop(anchor_index)
            candidate_index = candidate.index[0]

            new_row = {
                "data_source_name_1": group.loc[anchor_index, "data_source_name"],
                "data_source_name_2": group.loc[candidate_index, "data_source_name"],
                "label": 1,
                "category": "P: neighbor in Final_Custer of size 2",  # sub cluster of size 2
            }
            df_pairs = pd.concat([df_pairs, pd.DataFrame([new_row])], ignore_index=True)
            pairs_2_positive += 1

        # Branch 2: Cluster size between 3 and 10 (i.e. 2 < size <= 10)
        elif 2 < cluster_size <= 10:
            anchor = group.sample(n=1)
            anchor_index = anchor.index[0]
            anchor_embedding = group.loc[anchor_index, "embedding"]
            candidate_pool = group.drop(anchor_index)

            similarities = candidate_pool["embedding"].apply(
                lambda x: compute_cosine_similarity(anchor_embedding, x)
            )
            if similarities.empty:
                continue
            closest_index = similarities.idxmax()
            furthest_index = similarities.idxmin()

            new_row_closest = {
                "data_source_name_1": group.loc[anchor_index, "data_source_name"],
                "data_source_name_2": candidate_pool.loc[
                    closest_index, "data_source_name"
                ],
                "label": 1,
                "category": "P: Closest neighbor in Final_Custer of size 3-10",  # sub cluster of size 3-10
            }
            new_row_furthest = {
                "data_source_name_1": group.loc[anchor_index, "data_source_name"],
                "data_source_name_2": candidate_pool.loc[
                    furthest_index, "data_source_name"
                ],
                "label": 1,
                "category": "P: Furthest neighbor in Final_Custer of size 3-10",  # sub cluster of size 3-10
            }
            df_pairs = pd.concat(
                [df_pairs, pd.DataFrame([new_row_closest])], ignore_index=True
            )
            df_pairs = pd.concat(
                [df_pairs, pd.DataFrame([new_row_furthest])], ignore_index=True
            )
            pairs_3to10_positive += 2

        # Branch 3: Cluster size between 11 and 50 (i.e. 10 < size <= 50)
        elif 10 < cluster_size <= 50:
            anchors = group.sample(n=5)
            candidate_pool = group.drop(anchors.index)
            for anchor_index, anchor_row in anchors.iterrows():
                anchor_embedding = anchor_row["embedding"]
                similarities = candidate_pool["embedding"].apply(
                    lambda x: compute_cosine_similarity(anchor_embedding, x)
                )
                if similarities.empty:
                    continue
                closest_index = similarities.idxmax()
                furthest_index = similarities.idxmin()

                new_row_positive = {
                    "data_source_name_1": anchor_row["data_source_name"],
                    "data_source_name_2": candidate_pool.loc[
                        closest_index, "data_source_name"
                    ],
                    "label": 1,
                    "category": "P: Closest neighbor in Final_Custer of size 10-50",  # sub cluster of size 11-50
                }
                new_row_negative = {
                    "data_source_name_1": anchor_row["data_source_name"],
                    "data_source_name_2": candidate_pool.loc[
                        furthest_index, "data_source_name"
                    ],
                    "label": 0,
                    "category": "N: Furthest neighbor in Final_Custer of size 10-50",  # sub cluster of size 11-50
                }
                df_pairs = pd.concat(
                    [df_pairs, pd.DataFrame([new_row_positive])], ignore_index=True
                )
                df_pairs = pd.concat(
                    [df_pairs, pd.DataFrame([new_row_negative])], ignore_index=True
                )
                pairs_10to50_positive += 1
                pairs_10to50_negative += 1

        # Branch 4: Cluster size > 50
        else:
            anchors = group.sample(n=30)
            candidate_pool = group.drop(anchors.index)
            for anchor_index, anchor_row in anchors.iterrows():
                anchor_embedding = anchor_row["embedding"]
                similarities = candidate_pool["embedding"].apply(
                    lambda x: compute_cosine_similarity(anchor_embedding, x)
                )
                if similarities.empty:
                    continue
                sorted_candidates = similarities.sort_values(ascending=False)
                pool_len = len(sorted_candidates)
                if pool_len == 0:
                    continue

                closest_index1 = sorted_candidates.index[0]
                closest_index2 = sorted_candidates.index[9] if pool_len >= 10 else 0
                pos_1_3 = int(pool_len / 3) if pool_len >= 3 else 0
                pos_2_3 = int(2 * pool_len / 3) if pool_len >= 3 else 0
                furthest_index = sorted_candidates.index[-1]

                new_row_positive1 = {
                    "data_source_name_1": anchor_row["data_source_name"],
                    "data_source_name_2": candidate_pool.loc[
                        closest_index1, "data_source_name"
                    ],
                    "label": 1,
                    "category": "P: Closest neighbor in Final_Custer of size >50",  # sub cluster of size >50
                }
                new_row_positive2 = {
                    "data_source_name_1": anchor_row["data_source_name"],
                    "data_source_name_2": candidate_pool.loc[
                        closest_index2, "data_source_name"
                    ],
                    "label": 1,
                    "category": "P: 10th Closest neighbor in Final_Custer of size >50",  # sub cluster of size >50
                }

                new_row_negative1 = {
                    "data_source_name_1": anchor_row["data_source_name"],
                    "data_source_name_2": candidate_pool.loc[
                        sorted_candidates.index[pos_1_3], "data_source_name"
                    ],
                    "label": 0,
                    "category": "N: T1 Neighbor in Final_Custer of size >50",  # sub cluster of size >50
                }
                new_row_negative2 = {
                    "data_source_name_1": anchor_row["data_source_name"],
                    "data_source_name_2": candidate_pool.loc[
                        sorted_candidates.index[pos_2_3], "data_source_name"
                    ],
                    "label": 0,
                    "category": "N: T2 Neighbor in Final_Custer of size >50",  # sub cluster of size >50
                }
                new_row_negative3 = {
                    "data_source_name_1": anchor_row["data_source_name"],
                    "data_source_name_2": candidate_pool.loc[
                        furthest_index, "data_source_name"
                    ],
                    "label": 0,
                    "category": "N: Furthest neighbor in Final_Custer of size >50",  # sub cluster of size >50
                }
                df_pairs = pd.concat(
                    [df_pairs, pd.DataFrame([new_row_positive1])], ignore_index=True
                )
                df_pairs = pd.concat(
                    [df_pairs, pd.DataFrame([new_row_positive2])], ignore_index=True
                )
                df_pairs = pd.concat(
                    [df_pairs, pd.DataFrame([new_row_negative1])], ignore_index=True
                )
                df_pairs = pd.concat(
                    [df_pairs, pd.DataFrame([new_row_negative2])], ignore_index=True
                )
                df_pairs = pd.concat(
                    [df_pairs, pd.DataFrame([new_row_negative3])], ignore_index=True
                )
                pairs_over50_positive += 2
                pairs_over50_negative += 3

    # Log overall totals by branch category (warning level)
    logging.warning(
        f"Pairs from clusters of size 2: positive = {pairs_2_positive}, negative = 0"
    )
    logging.warning(
        f"Pairs from clusters of size 3-10: positive = {pairs_3to10_positive}, negative = {pairs_3to10_negative}"
    )
    logging.warning(
        f"Pairs from clusters of size 11-50: positive = {pairs_10to50_positive}, negative = {pairs_10to50_negative}"
    )
    logging.warning(
        f"Pairs from clusters of size >50: positive = {pairs_over50_positive}, negative = {pairs_over50_negative}"
    )

    total_positive = (
        pairs_2_positive
        + pairs_3to10_positive
        + pairs_10to50_positive
        + pairs_over50_positive
    )
    total_negative = (
        pairs_3to10_negative + pairs_10to50_negative + pairs_over50_negative
    )
    logging.warning(f"Total positive pairs: {total_positive}")
    logging.warning(f"Total negative pairs: {total_negative}")

    # logging -> start adding extra negative pairs: Subcluster x Noise in SubCluster
    logging.warning("Adding extra negative pairs: Subcluster x Noise in SubCluster ...")
    # Extra negative pair generation for final clusters of size > 3
    extra_neg_pairs = 0
    for final_cluster, group in df.groupby("Final_Cluster"):
        if len(group) > 2:
            logging.info(
                f"Processing extra negatives for final_cluster {final_cluster} with size {len(group)}"
            )
            anchor = group.sample(n=1)
            anchor_index = anchor.index[0]
            anchor_name = anchor.loc[anchor_index, "data_source_name"]
            anchor_cluster = anchor.loc[anchor_index, "cluster"]

            candidate_pool = df[
                (df["cluster"] == anchor_cluster)
                & (df["Subcluster"] == 0)
                & (df["Final_Cluster"] != final_cluster)
            ]
            if candidate_pool.empty:
                logging.info(
                    f"No extra negative candidates found for final_cluster {final_cluster}"
                )
                continue

            sample_size = min(4, len(candidate_pool))
            candidates = candidate_pool.sample(n=sample_size)
            for cand_index, cand_row in candidates.iterrows():
                new_pair = {
                    "data_source_name_1": anchor_name,
                    "data_source_name_2": cand_row["data_source_name"],
                    "label": 0,
                    "category": "N: In Cluster of size > 3, random pair of (subcluster Y x noise subcluster)",
                }
                df_pairs = pd.concat(
                    [df_pairs, pd.DataFrame([new_pair])], ignore_index=True
                )
                extra_neg_pairs += 1
    logging.warning(
        f"Total extra negative pairs from final clusters: {extra_neg_pairs}"
    )

    # logging -> start adding extra negative pairs: Different Subcluster x Noise in SubCluster
    logging.warning(
        "Adding extra negative pairs: Different Subcluster x Other SubCluster ..."
    )
    # Additional Negative Pairs: Different Sublusters
    extra_neg_subcluster_pairs = 0
    for main_cluster, cluster_group in df.groupby("cluster"):
        if len(cluster_group) > 10:
            for subcluster_val, subcluster_group in cluster_group.groupby("Subcluster"):
                logging.info(
                    f"Processing main cluster {main_cluster}, subcluster {subcluster_val} with {len(subcluster_group)} items"
                )
                num_anchors = min(2, len(subcluster_group))
                anchors = subcluster_group.sample(n=num_anchors, random_state=42)
                for anchor_index, anchor_row in anchors.iterrows():
                    anchor_embedding = anchor_row["embedding"]
                    candidate_pool = cluster_group[
                        cluster_group["Subcluster"] != subcluster_val
                    ]
                    candidate_pool = candidate_pool[
                        candidate_pool.index != anchor_index
                    ]
                    if candidate_pool.empty:
                        continue
                    sample_size = min(2, len(candidate_pool))
                    candidates = candidate_pool.sample(n=1, random_state=42)
                    for cand_index, cand_row in candidates.iterrows():
                        new_pair = {
                            "data_source_name_1": anchor_row["data_source_name"],
                            "data_source_name_2": cand_row["data_source_name"],
                            "label": 0,
                            "category": "N: In Cluster of size > 10, pair of (Subcluster Y x SubCluster Z)",
                        }
                        df_pairs = pd.concat(
                            [df_pairs, pd.DataFrame([new_pair])],
                            ignore_index=True,
                        )
                        extra_neg_subcluster_pairs += 1
                logging.info(
                    f"Main cluster {main_cluster}, subcluster {subcluster_val}: processed {num_anchors} anchors"
                )
    logging.warning(
        f"Total extra negative pairs from different subclusters: {extra_neg_subcluster_pairs}"
    )

    # logging -> start adding extra negative pairs: Noise Cluster x closest in a random sample of 1% of the data
    logging.warning(
        "Adding extra negative pairs: Noise Cluster x closest in a random sample of 1% of the data ..."
    )
    # Extra Negative Pairs from Cluster 0 (noise cluster)
    extra_neg_cluster0_pairs = 0
    noise_cluster = df[(df["cluster"] == 0) & (df["Subcluster"] == 0)]
    if not noise_cluster.empty:
        num_noise_anchors = min(500, len(noise_cluster))
        noise_anchors = noise_cluster.sample(n=num_noise_anchors, random_state=42)
        for anchor_index, anchor_row in noise_anchors.iterrows():
            random_state = int(time.time())
            anchor_embedding = anchor_row["embedding"]
            candidate_pool = df.sample(frac=0.01, random_state=random_state)
            candidate_pool = candidate_pool[
                ~candidate_pool.index.isin(noise_anchors.index)
            ]
            if candidate_pool.empty:
                continue
            similarities = candidate_pool["embedding"].apply(
                lambda x: compute_cosine_similarity(anchor_embedding, x)
            )
            best_candidate_index = similarities.idxmax()
            new_pair = {
                "data_source_name_1": anchor_row["data_source_name"],
                "data_source_name_2": candidate_pool.loc[
                    best_candidate_index, "data_source_name"
                ],
                "label": 0,
                "category": "N: Noise (Cluster=0 and Subcluster=0) x Closest in 1% random sample of all data",
            }
            df_pairs = pd.concat(
                [df_pairs, pd.DataFrame([new_pair])], ignore_index=True
            )
            extra_neg_cluster0_pairs += 1
        logging.warning(
            f"Total extra negative pairs from noise cluster (cluster 0): {extra_neg_cluster0_pairs}"
        )
    else:
        logging.warning(
            "No data found in noise cluster (cluster 0) for extra negative pairs."
        )

    # logging final totals
    total_negative += (
        extra_neg_pairs + extra_neg_subcluster_pairs + extra_neg_cluster0_pairs
    )
    # count and logg values counts of categories
    logging.warning("Category counts:")
    logging.warning(df_pairs["category"].value_counts())
    logging.warning(f"Total positive pairs: {total_positive}")
    logging.warning(f"Total negative pairs: {total_negative}")
    # logging ratio
    ratio = total_positive / total_negative
    logging.warning(f"Positive/Negative ratio: {ratio:.2f}")

    # Save df_pairs to S3
    folder_path = "registry_data_catalog_experiments/task_1_deduplication/publications_original_pairs/"
    file_name = "original_pairs_v1.parquet"
    logging.warning("Uploading pairs to S3 ...")
    upload_file_to_s3(
        df_pairs,
        config.BUCKET_NAME_DEV,
        folder_path,
        file_name,
    )
    logging.warning("Upload completed.")


if __name__ == "__main__":
    main()
