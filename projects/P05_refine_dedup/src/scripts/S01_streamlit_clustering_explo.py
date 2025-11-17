import sys
from pathlib import Path
import os

working_dir = "/home/gpinon/more_europa/clean_rdc_experiments/projects/P05_refine_dedup"
os.chdir(working_dir)

# print working directory
print(Path.cwd())

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from p05_refine_dedup.utils.utils import run_dbscan, umap_reduction
from p05_refine_dedup.utils.s3_io_functions import load_parquet_from_s3

from p05_refine_dedup import config

# Best parameters (fixed)
best_eps1 = 0.340  # 0.3314610779075241
best_eps2 = 0.435  # 0.380396662361368
best_eps3 = 0.27  # 0.275073319059974
BEST_MIN_SAMPLES = 2  # All min_samples are fixed at 2


# ---------------------- Functions ---------------------- #


def insert_newlines_every_three_words(s):
    words = s.split()[:50]
    chunks = [" ".join(words[i : i + 4]) for i in range(0, len(words), 4)]
    return "<br>".join(chunks)


def generate_color_palette(n):
    return [f"hsl({int(x)}, 70%, 50%)" for x in np.linspace(0, 360, n, endpoint=False)]


def get_cluster_table_info(df, total_count):
    # Include all clusters (including noise as 0)
    clusters = np.sort(df["cluster"].unique())
    rows = []
    for cluster in clusters:
        cluster_df = df[df["cluster"] == cluster]
        size = len(cluster_df)
        perc = (size / total_count) * 100

        # Select the two most popular aliases instead of random ones
        top_aliases = cluster_df.sort_values(
            by="number_of_occurrences", ascending=False
        ).head(2)
        examples = top_aliases["full_name"].tolist()
        occurrences = top_aliases["number_of_occurrences"].tolist()

        # Calculate total occurrences for the cluster
        total_occurrences = cluster_df["number_of_occurrences"].sum()

        if len(examples) == 1:
            examples.append(examples[0])
            occurrences.append(occurrences[0])

        row = {
            "Cluster": "0 (Noise)" if cluster == 0 else cluster,
            "%": f"{perc:.1f}%",
            "nb_of_aliases": size,
            "total_occ": total_occurrences,
            "Alias 1": examples[0],
            "nb_occ_1": occurrences[0],
            "Alias 2": examples[1],
            "nb_occ_2": occurrences[1],
        }
        rows.append(row)
    df_info = pd.DataFrame(rows)
    return df_info


# ---------------------- Main Application ---------------------- #
def main():
    st.set_page_config(layout="wide")
    st.title("Cluster Analysis Tool")

    # Initialize session state variables if not present
    for key in [
        "dbscan_run",
        "cluster_analysis_run",
        "cluster_analysis_reset",
        "df_clustering",
        "dbscan_params",
        "dbscan_time",
        "cluster_table",
        "selected_cluster",
        "umap_result",
        "sub_labels",
        "sub_dbscan_params",
        # New session state variables for global UMAP
        "global_umap_result",
        "global_umap_params",
    ]:
        if key not in st.session_state:
            st.session_state[key] = False if "run" in key or "reset" in key else None

    # ------------------- Load Data ------------------- #
    bucket_name = config.BUCKET_NAME_DEV
    folder_path = "registry_data_catalog_experiments/P05_refine_dedup"
    file_name = "registry_names_embeddings.parquet"
    df = load_parquet_from_s3(
        bucket_name=bucket_name,
        folder_path=folder_path,
        file_name=file_name,
    )
    # rank by 'number_of_occurrences' by descending order
    df["rank"] = (
        df["number_of_occurrences"].rank(method="first", ascending=False).astype(int)
    )
    # # filter on the top 5000
    # df = df[df["rank"] <= 5000].reset_index(drop=True)
    # # select 30% of the data for testing, with a fixed seed for reproducibility
    # df = df.sample(frac=0.3, random_state=42).reset_index(drop=True)
    total_data_sources = len(df)
    embeddings = np.vstack(df["full_name_embedding"].values)

    # ------------------- Sidebar ------------------- #
    st.sidebar.header("DBSCAN Clustering Parameters")
    # Vertical layout for DBSCAN parameters
    eps = st.sidebar.slider(
        "Epsilon",
        min_value=0.28,
        max_value=0.50,
        value=best_eps1,
        step=0.005,
        format="%.3f",
    )
    min_samples = st.sidebar.slider(
        "Min Samples", min_value=1, max_value=5, value=BEST_MIN_SAMPLES
    )

    if st.sidebar.button("Run clustering"):
        labels, comp_time = run_dbscan(embeddings, eps, min_samples)
        df["cluster"] = labels
        st.session_state["df_clustering"] = df.copy()
        st.session_state["dbscan_params"] = {"Epsilon": eps, "Min Samples": min_samples}
        st.session_state["dbscan_time"] = comp_time
        st.session_state["dbscan_run"] = True
        st.session_state["cluster_table"] = get_cluster_table_info(
            df, total_data_sources
        )
        st.session_state["cluster_analysis_reset"] = True
        st.session_state["cluster_analysis_run"] = False
        st.session_state["sub_labels"] = None
        st.session_state["sub_dbscan_params"] = None

        st.sidebar.success("DBSCAN clustering completed.")

    # Add global UMAP parameters section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Global UMAP Parameters")
    global_umap_n_neighbors = st.sidebar.slider(
        "Global n_neighbors",
        min_value=2,
        max_value=20,
        value=10,
        key="global_umap_n_neighbors",
    )
    global_umap_min_dist = st.sidebar.slider(
        "Global min_dist",
        min_value=0.01,
        max_value=0.5,
        value=0.1,
        key="global_umap_min_dist",
    )
    if st.sidebar.button("Run UMAP on all clusters", key="run_global_umap"):
        if st.session_state.get("df_clustering") is not None:
            embeddings = np.vstack(
                st.session_state["df_clustering"]["full_name_embedding"].values
            )
            with st.spinner("Computing UMAP for all data points..."):
                global_umap_result = umap_reduction(
                    embeddings,
                    n_neighbors=global_umap_n_neighbors,
                    min_dist=global_umap_min_dist,
                    n_components=2,
                )
                st.session_state["global_umap_result"] = global_umap_result
                st.session_state["global_umap_params"] = {
                    "n_neighbors": global_umap_n_neighbors,
                    "min_dist": global_umap_min_dist,
                }
            st.sidebar.success("Global UMAP completed.")
        else:
            st.sidebar.warning("Run DBSCAN clustering first.")

    st.sidebar.markdown("---")
    st.sidebar.header("Cluster Analysis")
    st.sidebar.subheader("Cluster Selection")
    col_cluster, col_cluster_clear = st.sidebar.columns([3, 2])
    cluster_input = col_cluster.text_input("Cluster Number", key="cluster_input")
    if col_cluster_clear.button("Clear", key="clear_cluster_input"):
        st.session_state["cluster_input"] = ""
        cluster_input = ""
    if st.sidebar.button("Run analysis", key="run_analysis"):
        try:
            actual_cluster = int(cluster_input)
        except:
            st.sidebar.error("Cluster number input should be a number")
            st.session_state["cluster_analysis_run"] = False
        else:
            if (
                st.session_state["df_clustering"] is None
                or actual_cluster
                not in st.session_state["df_clustering"]["cluster"].unique()
            ):
                st.sidebar.error("Input cluster number doesn't exist")
                st.session_state["cluster_analysis_run"] = False
            else:
                st.session_state["selected_cluster"] = actual_cluster
                cluster_df = st.session_state["df_clustering"][
                    st.session_state["df_clustering"]["cluster"] == actual_cluster
                ]
                # Default UMAP parameters
                n_neighbors = st.session_state.get("umap_n_neighbors", 10)
                min_dist = st.session_state.get("umap_min_dist", 0.1)
                with st.spinner("Running UMAP analysis on selected cluster..."):
                    umap_result = umap_reduction(
                        np.vstack(cluster_df["full_name_embedding"].values),
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        n_components=2,
                    )
                st.session_state["umap_result"] = umap_result
                st.session_state["cluster_analysis_run"] = True
                st.session_state["cluster_analysis_reset"] = False
                st.session_state["sub_labels"] = None
                st.session_state["sub_dbscan_params"] = None
                if actual_cluster == 0:
                    st.sidebar.success("Cluster analysis completed for noise (0).")
                else:
                    st.sidebar.success("Cluster analysis completed.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("UMAP Parameters (Selected Cluster)")
    umap_n_neighbors = st.sidebar.slider(
        "n_neighbors", min_value=2, max_value=20, value=10, key="umap_n_neighbors"
    )
    umap_min_dist = st.sidebar.slider(
        "min_dist", min_value=0.01, max_value=0.5, value=0.1, key="umap_min_dist"
    )
    if st.sidebar.button("Run UMAP on selected cluster", key="run_umap"):
        if st.session_state["selected_cluster"] is None:
            st.sidebar.error("Please run cluster analysis first")
        else:
            actual_cluster = st.session_state["selected_cluster"]
            cluster_df = st.session_state["df_clustering"][
                st.session_state["df_clustering"]["cluster"] == actual_cluster
            ]
            with st.spinner("Running UMAP analysis with new parameters..."):
                umap_result = umap_reduction(
                    np.vstack(cluster_df["full_name_embedding"].values),
                    n_neighbors=umap_n_neighbors,
                    min_dist=umap_min_dist,
                    n_components=2,
                )
            st.session_state["umap_result"] = umap_result
            st.sidebar.success("UMAP re-analysis completed.")

    # --- New Sub-clustering Section in Sidebar ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Sub-clustering")
    sub_eps = st.sidebar.slider(
        "Sub-clustering Epsilon",
        min_value=0.22,
        max_value=0.48,
        value=0.275,
        step=0.005,
        format="%.3f",
    )
    sub_min_samples = st.sidebar.slider(
        "Sub-clustering Min Samples", min_value=1, max_value=10, value=2
    )
    if st.sidebar.button("Run sub clustering", key="run_sub_clustering"):
        if st.session_state["selected_cluster"] is None:
            st.sidebar.error("Please run cluster analysis first")
        else:
            actual_cluster = st.session_state["selected_cluster"]
            cluster_df = st.session_state["df_clustering"][
                st.session_state["df_clustering"]["cluster"] == actual_cluster
            ]
            cluster_embeddings = np.vstack(cluster_df["full_name_embedding"].values)
            sub_labels, sub_comp_time = run_dbscan(
                cluster_embeddings, sub_eps, sub_min_samples
            )
            cluster_df["SubCluster"] = sub_labels
            st.session_state["sub_labels"] = sub_labels
            st.session_state["df_sub_clustering"] = cluster_df.copy()
            st.session_state["sub_dbscan_params"] = {
                "Epsilon": sub_eps,
                "Min Samples": sub_min_samples,
            }
            st.session_state["sub_dbscan_time"] = sub_comp_time
            st.sidebar.success("Sub-clustering completed.")

    # ------------------- Main Page ------------------- #
    st.header("DBSCAN Clustering Results")
    if st.session_state["dbscan_run"]:
        df_clust = st.session_state["df_clustering"]
        noise_count = len(df_clust[df_clust["cluster"] == 0])
        noise_percentage = (noise_count / total_data_sources) * 100
        num_clusters = len(df_clust[df_clust["cluster"] != 0]["cluster"].unique())
        st.markdown(f"**Total data source names:** {total_data_sources}")
        st.markdown(
            f"**DBSCAN Parameters:** Epsilon = {st.session_state['dbscan_params']['Epsilon']:.2f}, Min Samples = {st.session_state['dbscan_params']['Min Samples']}"
        )
        st.markdown(
            f"**Computation Time:** {st.session_state['dbscan_time']:.2f} seconds"
        )
        st.markdown(f"**Number of clusters obtained:** {num_clusters}")
        st.markdown(
            f"**Number of noise data sources:** {noise_count} ({noise_percentage:.1f}%)"
        )
    else:
        st.info("Run DBSCAN clustering from the sidebar to see results.")

    st.header("Cluster Selection")
    if st.session_state["dbscan_run"]:
        # Create two columns: left for filters, right for the distribution graph
        col_filters, col_distribution = st.columns([1, 2])
        with col_distribution:
            st.markdown("### Distribution of cluster sizes")
            cluster_table = st.session_state["cluster_table"]
            cluster_table["Size_bin"] = cluster_table["nb_of_aliases"].apply(
                lambda x: x if x <= 50 else ">50"
            )
            distribution = (
                cluster_table.groupby("Size_bin").size().reset_index(name="Count")
            )
            fig_dist = px.bar(distribution, x="Size_bin", y="Count", text="Count")
            fig_dist.update_traces(
                textposition="outside",
                hovertemplate="Size: %{x}<br>Count: %{y}<extra></extra>",
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        with col_filters:
            st.markdown("### Cluster Table (default: first 5 rows)")
            num_gt50 = int((cluster_table["nb_of_aliases"] > 50).sum())
            st.markdown(f"* Number of clusters with size > 50: {num_gt50}")

            clusters_gt50 = cluster_table[cluster_table["nb_of_aliases"] > 50][
                ["Cluster", "nb_of_aliases"]
            ]
            if not clusters_gt50.empty:
                clusters_gt50 = clusters_gt50.sort_values(
                    by="nb_of_aliases", ascending=False
                )
                st.markdown("**Clusters with size > 50:**")
                st.table(clusters_gt50)

            with st.expander("Filters"):
                col_f1, col_f1_clear = st.columns([3, 1])
                filter_cluster = col_f1.text_input(
                    "Filter by Cluster Number (exact, 1-based)", key="filter_cluster"
                )
                if col_f1_clear.button("Clear", key="clear_filter_cluster"):
                    st.session_state["filter_cluster"] = ""
                    filter_cluster = ""

                # Size filter
                col_f2, col_f2b, col_f2_clear = st.columns([3, 2, 1])
                size_operator = col_f2.selectbox(
                    "Nb of aliases Operator",
                    options=[">", "<", "="],
                    key="size_operator",
                )
                size_value = col_f2b.number_input(
                    "Value", min_value=0, value=0, step=1, key="size_value"
                )
                if col_f2_clear.button("Clear", key="clear_size_filter"):
                    st.session_state["size_value"] = 0

                # Percentage filter
                col_f3, col_f3b, col_f3_clear = st.columns([3, 2, 1])
                perc_operator = col_f3.selectbox(
                    "% Operator", options=[">", "<", "="], key="perc_operator"
                )
                perc_value = col_f3b.number_input(
                    "Value",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=0.01,
                    key="perc_value",
                )
                if col_f3_clear.button("Clear", key="clear_perc_filter"):
                    st.session_state["perc_value"] = 0.0

                # Total occurrences filter
                col_f4, col_f4b, col_f4_clear = st.columns([3, 2, 1])
                occ_operator = col_f4.selectbox(
                    "Total occurrences Operator",
                    options=[">", "<", "="],
                    key="occ_operator",
                )
                occ_value = col_f4b.number_input(
                    "Value",
                    min_value=0,
                    value=0,
                    step=1,
                    key="occ_value",
                )
                if col_f4_clear.button("Clear", key="clear_occ_filter"):
                    st.session_state["occ_value"] = 0

                # Text filter
                col_f5, col_f5_clear = st.columns([3, 1])
                text_filter = col_f5.text_input(
                    "Text filter on alias name", key="text_filter"
                )
                if col_f5_clear.button("Clear", key="clear_text_filter"):
                    st.session_state["text_filter"] = ""
                    text_filter = ""
            filtered_table = cluster_table.copy()
            if filter_cluster:
                try:
                    filter_val = int(filter_cluster)
                    if filter_val == 0:
                        filtered_table = filtered_table[
                            filtered_table["Cluster"] == "0 (Noise)"
                        ]
                    else:
                        filtered_table = filtered_table[
                            filtered_table["Cluster"] == filter_val
                        ]
                except:
                    st.error("Cluster number filter should be a number")
            if size_operator and size_value:
                if size_operator == ">":
                    filtered_table = filtered_table[
                        filtered_table["nb_of_aliases"] > size_value
                    ]
                elif size_operator == "<":
                    filtered_table = filtered_table[
                        filtered_table["nb_of_aliases"] < size_value
                    ]
                elif size_operator == "=":
                    filtered_table = filtered_table[
                        filtered_table["nb_of_aliases"] == size_value
                    ]
            if (
                perc_operator
                and perc_value is not None
                and perc_value != 0.0
                and "%" in filtered_table.columns
            ):
                filtered_table["Percentage_num"] = (
                    filtered_table["%"].str.replace("%", "").astype(float)
                )
                if perc_operator == ">":
                    filtered_table = filtered_table[
                        filtered_table["Percentage_num"] > perc_value
                    ]
                elif perc_operator == "<":
                    filtered_table = filtered_table[
                        filtered_table["Percentage_num"] < perc_value
                    ]
                elif perc_operator == "=":
                    filtered_table = filtered_table[
                        filtered_table["Percentage_num"] == perc_value
                    ]
                filtered_table.drop(columns=["Percentage_num"], inplace=True)
            if occ_operator and occ_value:
                if occ_operator == ">":
                    filtered_table = filtered_table[
                        filtered_table["total_occ"] > occ_value
                    ]
                elif occ_operator == "<":
                    filtered_table = filtered_table[
                        filtered_table["total_occ"] < occ_value
                    ]
                elif occ_operator == "=":
                    filtered_table = filtered_table[
                        filtered_table["total_occ"] == occ_value
                    ]
            if text_filter:
                filtered_table = filtered_table[
                    filtered_table["Alias 1"].str.contains(
                        text_filter, case=False, na=False
                    )
                    | filtered_table["Alias 2"].str.contains(
                        text_filter, case=False, na=False
                    )
                ]
            sort_option = st.selectbox(
                "Sort by",
                options=[
                    "None",
                    "nb_of_aliases ascending",
                    "nb_of_aliases descending",
                    "% ascending",
                    "% descending",
                    "total_occ ascending",
                    "total_occ descending",
                ],
            )
            if sort_option == "nb_of_aliases ascending":
                filtered_table = filtered_table.sort_values(
                    by="nb_of_aliases", ascending=True
                )
            elif sort_option == "nb_of_aliases descending":
                filtered_table = filtered_table.sort_values(
                    by="nb_of_aliases", ascending=False
                )
            elif sort_option == "% ascending":
                filtered_table["Percentage_num"] = (
                    filtered_table["%"].str.replace("%", "").astype(float)
                )
                filtered_table = filtered_table.sort_values(
                    by="Percentage_num", ascending=True
                )
                filtered_table.drop(columns=["Percentage_num"], inplace=True)
            elif sort_option == "% descending":
                filtered_table["Percentage_num"] = (
                    filtered_table["%"].str.replace("%", "").astype(float)
                )
                filtered_table = filtered_table.sort_values(
                    by="Percentage_num", ascending=False
                )
                filtered_table.drop(columns=["Percentage_num"], inplace=True)
            elif sort_option == "total_occ ascending":
                filtered_table = filtered_table.sort_values(
                    by="total_occ", ascending=True
                )
            elif sort_option == "total_occ descending":
                filtered_table = filtered_table.sort_values(
                    by="total_occ", ascending=False
                )
        st.dataframe(filtered_table.head(5), use_container_width=True)

        # Add Global UMAP visualization here
        st.subheader("UMAP Visualization of All Clusters")
        if st.session_state["global_umap_result"] is not None:
            umap_params = st.session_state["global_umap_params"]
            st.markdown(
                f"**UMAP Parameters:** n_neighbors = {umap_params['n_neighbors']}, min_dist = {umap_params['min_dist']}"
            )

            # Create UMAP dataframe with cluster information
            df_clust = st.session_state["df_clustering"]
            global_umap = st.session_state["global_umap_result"]

            umap_df = pd.DataFrame(
                {
                    "UMAP1": global_umap[:, 0],
                    "UMAP2": global_umap[:, 1],
                    "Cluster": df_clust["cluster"].astype(str),
                    "DataSource": df_clust["full_name"],
                    "Name": df_clust["full_name"].apply(
                        insert_newlines_every_three_words
                    ),
                    "nb_occ": df_clust["number_of_occurrences"],
                }
            )

            # Set up color mapping
            unique_clusters = sorted(df_clust["cluster"].unique())
            color_map = {}
            color_map["0"] = "#333333"  # Noise color
            color_list = px.colors.qualitative.Light24
            i = 0
            for cluster in unique_clusters:
                if cluster == 0:
                    continue
                color_map[str(cluster)] = color_list[i % len(color_list)]
                i += 1

            # Add filter for clusters
            cluster_filter = st.text_input(
                "Filter cluster(s) (comma-separated; leave empty for all)",
                key="global_cluster_filter",
            )

            if cluster_filter.strip() == "":
                selected_clusters = sorted(umap_df["Cluster"].unique())
            else:
                selected_clusters = [
                    s.strip() for s in cluster_filter.split(",") if s.strip() != ""
                ]

            filtered_umap_df = umap_df[umap_df["Cluster"].isin(selected_clusters)]

            # Create UMAP scatter plot
            fig = px.scatter(
                filtered_umap_df,
                x="UMAP1",
                y="UMAP2",
                custom_data=["Cluster", "Name", "nb_occ"],
                color="Cluster",
                width=1200,
                height=600,
                color_discrete_map=color_map,
            )
            fig.update_traces(
                marker=dict(size=12, opacity=0.7),
                hovertemplate="<b>Cluster: %{customdata[0]}</b><br><br>%{customdata[1]}<br>nb_occ=%{customdata[2]}<extra></extra>",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run DBSCAN clustering to see UMAP visualization of all clusters.")

    else:
        st.info("Run DBSCAN clustering to display the cluster selection table.")

    # Cluster Analysis Section
    if st.session_state["dbscan_run"] and st.session_state["cluster_analysis_run"]:
        actual_cluster = st.session_state["selected_cluster"]
        df_clust = st.session_state["df_clustering"]
        if actual_cluster == 0:
            st.header("Cluster Analysis: 0 (Noise)")
        else:
            st.header(f"Cluster Analysis: Cluster {actual_cluster}")
        cluster_df = df_clust[df_clust["cluster"] == actual_cluster]
        cluster_size = len(cluster_df)
        cluster_percentage = (cluster_size / total_data_sources) * 100
        total_occurrences = cluster_df["number_of_occurrences"].sum()
        st.markdown(
            f"**Cluster Size:** {cluster_size} (**{cluster_percentage:.1f}%** of total) | **Total Occurrences:** {total_occurrences}"
        )
    else:
        st.header("Cluster Analysis")

    col_umap, col_table = st.columns([2, 1])
    if st.session_state["dbscan_run"]:
        if st.session_state["cluster_analysis_reset"]:
            with col_umap:
                st.subheader("2D Umap Representation")
                st.markdown("**UMAP Parameters:** XXX")
                st.plotly_chart(px.scatter(), height=600, use_container_width=True)
            with col_table:
                st.markdown("### Data Sources in this Cluster")
                st.markdown("Scrollable Table: (No data)")
        elif st.session_state["cluster_analysis_run"]:
            with col_umap:
                if st.session_state["sub_labels"] is not None:
                    st.subheader("2D Umap Representation + DBScan sub Clustering")
                    umap_n_neighbors_current = st.session_state.get(
                        "umap_n_neighbors", 10
                    )
                    umap_min_dist_current = st.session_state.get("umap_min_dist", 0.1)
                    sub_params = st.session_state["sub_dbscan_params"]
                    cluster_df = cluster_df.copy()
                    cluster_df["SubCluster"] = st.session_state["sub_labels"]
                    sub_cluster_table = (
                        cluster_df.groupby("SubCluster")
                        .size()
                        .reset_index(name="nb_aliases")
                    )
                    sub_cluster_table["Size_bin"] = sub_cluster_table[
                        "nb_aliases"
                    ].apply(lambda x: x if x <= 20 else ">20")
                    sub_distribution = (
                        sub_cluster_table.groupby("Size_bin")
                        .size()
                        .reset_index(name="Count")
                    )
                    sub_dist = (
                        cluster_df.groupby("SubCluster")
                        .size()
                        .reset_index(name="Count")
                    )
                    fig_sub = px.bar(
                        sub_distribution, x="Size_bin", y="Count", text="Count"
                    )
                    fig_sub.update_traces(
                        textposition="outside",
                        hovertemplate="Size: %{x}<br>Count: %{y}<extra></extra>",
                    )
                    # update fig_sub size
                    fig_sub.update_layout(height=500)
                    sub_dbscan_time = st.session_state.get("sub_dbscan_time", 0)
                    sub_labels_arr = np.array(st.session_state["sub_labels"])
                    num_sub_clusters = len(
                        np.unique(sub_labels_arr[sub_labels_arr != 0])
                    )
                    num_sub_noise = np.sum(sub_labels_arr == 0)
                    num_sub_noise_percentage = (
                        (num_sub_noise / cluster_size) * 100 if cluster_size > 0 else 0
                    )
                    num_large = np.sum(sub_dist["Count"] > 20)
                    col_info, col_chart = st.columns([1, 1])
                    with col_info:
                        st.markdown(
                            f"**UMAP Parameters:** n_neighbors = {umap_n_neighbors_current}, min_dist = {umap_min_dist_current}"
                        )
                        st.markdown(
                            f"**Sub-clustering Parameters:** Epsilon = {sub_params['Epsilon']:.3f}, Min Samples = {sub_params['Min Samples']}"
                        )
                        st.markdown(
                            f"**Sub Clustering info:**\n"
                            f"- Total data sources in cluster: {cluster_size}\n"
                            f"- Computation time: {sub_dbscan_time:.2f} seconds\n"
                            f"- Number of sub clusters obtained: {num_sub_clusters}\n"
                            f"- Number of noise data sources: {num_sub_noise} ({num_sub_noise_percentage:.1f}%)\n"
                            f"- Number of sub clusters with size > 20: {num_large}"
                        )
                        # Display top subclusters with size > 20 (Top 12 if available)
                        sub_clusters_gt20 = sub_cluster_table[
                            sub_cluster_table["nb_aliases"] > 20
                        ][["SubCluster", "nb_aliases"]]

                        # Calculate total occurrences for each subcluster
                        sub_clusters_with_occurrences = (
                            cluster_df.groupby("SubCluster")
                            .agg(
                                nb_aliases=("SubCluster", "size"),
                                total_occurrences=("number_of_occurrences", "sum"),
                            )
                            .reset_index()
                        )

                        # Merge with the existing sub_clusters_gt20
                        if not sub_clusters_gt20.empty:
                            sub_clusters_gt20 = sub_clusters_gt20.merge(
                                sub_clusters_with_occurrences[
                                    ["SubCluster", "total_occurrences"]
                                ],
                                on="SubCluster",
                            )
                            sub_clusters_gt20 = sub_clusters_gt20.rename(
                                columns={"Size": "nb_aliases"}
                            )
                            sub_clusters_gt20 = sub_clusters_gt20.sort_values(
                                by="nb_aliases", ascending=False
                            )
                            st.markdown("**Sub-Clusters with size > 20:**")
                            st.dataframe(sub_clusters_gt20, height=200)

                    with col_chart:
                        st.markdown("##### Distribution of sub cluster sizes (<= 20)")
                        st.plotly_chart(fig_sub, use_container_width=True)
                    if st.session_state["umap_result"] is not None:
                        umap_res = st.session_state["umap_result"]
                        umap_df = pd.DataFrame(
                            {
                                "UMAP1": umap_res[:, 0],
                                "UMAP2": umap_res[:, 1],
                                "DataSource": cluster_df["full_name"].tolist(),
                                "nb_occ": cluster_df["number_of_occurrences"].tolist(),
                            }
                        )

                        # For subclustering view
                        if st.session_state["sub_labels"] is not None:
                            sub_labels = (
                                st.session_state["sub_labels"].astype(str)
                                if hasattr(st.session_state["sub_labels"], "astype")
                                else [
                                    str(label)
                                    for label in st.session_state["sub_labels"]
                                ]
                            )
                            umap_df["SubCluster"] = sub_labels
                            umap_df["Name"] = umap_df.apply(
                                lambda row: insert_newlines_every_three_words(
                                    row["DataSource"]
                                ),
                                axis=1,
                            )
                            filter_col, chart_col = st.columns([1, 4])
                            with filter_col:
                                subcluster_input = st.text_input(
                                    "Filter subcluster(s) (comma-separated; leave empty for all)",
                                    key="subcluster_filter_text",
                                )
                            if subcluster_input.strip() == "":
                                selected_subclusters = sorted(
                                    umap_df["SubCluster"].unique(),
                                    key=lambda x: (
                                        int(x) if x.lstrip("-").isdigit() else x
                                    ),
                                )
                            else:
                                selected_subclusters = [
                                    s.strip()
                                    for s in subcluster_input.split(",")
                                    if s.strip() != ""
                                ]
                            filtered_umap_df = umap_df[
                                umap_df["SubCluster"].isin(selected_subclusters)
                            ]
                            unique_subclusters = umap_df["SubCluster"].unique()
                            color_map = {}
                            color_map["0"] = "#333333"  # Noise color for subclusters
                            color_list = px.colors.qualitative.Light24
                            i = 0
                            for sub in sorted(unique_subclusters):
                                if sub == "0":
                                    continue
                                color_map[sub] = color_list[i % len(color_list)]
                                i += 1
                            fig = px.scatter(
                                filtered_umap_df,
                                x="UMAP1",
                                y="UMAP2",
                                custom_data=["SubCluster", "Name", "nb_occ"],
                                color="SubCluster",
                                width=1200,
                                height=600,
                                color_discrete_map=color_map,
                            )
                            fig.update_traces(
                                marker=dict(size=12, opacity=0.8),
                                hovertemplate="<b>SubCluster: %{customdata[0]}</b><br><br>%{customdata[1]}<br>nb_occ=%{customdata[2]}<extra></extra>",
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        # For regular cluster view (no subclustering)
                        else:
                            umap_df["Name"] = umap_df.apply(
                                lambda row: insert_newlines_every_three_words(
                                    row["DataSource"]
                                ),
                                axis=1,
                            )
                            fig = px.scatter(
                                umap_df,
                                x="UMAP1",
                                y="UMAP2",
                                custom_data=["Name", "nb_occ"],
                            )
                            fig.update_traces(
                                marker=dict(color="red", size=12, opacity=0.8),
                                hovertemplate="<br>%{customdata[0]}<br>nb_occ=%{customdata[1]}<extra></extra>",
                            )
                            fig.update_layout(height=600)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("UMAP analysis not run yet for this cluster.")
                else:
                    st.subheader("2D Umap Representation")
                    default_neighbors = st.session_state.get(
                        "umap_n_neighbors_default", 10
                    )
                    default_min_dist = st.session_state.get(
                        "umap_min_dist_default", 0.1
                    )
                    st.markdown(
                        f"**n_neighbors:** {default_neighbors}, **min_dist:** {default_min_dist}"
                    )
                    if st.session_state["umap_result"] is not None:
                        umap_res = st.session_state["umap_result"]
                        umap_df = pd.DataFrame(
                            {
                                "UMAP1": umap_res[:, 0],
                                "UMAP2": umap_res[:, 1],
                                "DataSource": cluster_df["full_name"].tolist(),
                                "nb_occ": cluster_df["number_of_occurrences"].tolist(),
                            }
                        )
                        umap_df["Name"] = umap_df.apply(
                            lambda row: insert_newlines_every_three_words(
                                row["DataSource"]
                            ),
                            axis=1,
                        )
                        fig = px.scatter(
                            umap_df,
                            x="UMAP1",
                            y="UMAP2",
                            custom_data=["Name", "nb_occ"],
                        )
                        fig.update_traces(
                            marker=dict(color="red", size=12, opacity=0.8),
                            hovertemplate="<br>%{customdata[0]}<br>nb_occ=%{customdata[1]}<extra></extra>",
                        )
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("UMAP analysis not run yet for this cluster.")
            with col_table:
                st.markdown("### Data Sources in this Cluster")
                # --- Top 10 Subclusters Table ---
                # Only display if sub-clustering has been run (i.e. the "SubCluster" column exists)
                if "SubCluster" in cluster_df.columns:
                    # Group by SubCluster and compute Size and Example (first occurrence)
                    top_subclusters = (
                        cluster_df.groupby("SubCluster")
                        .agg(
                            nb_aliases=("SubCluster", "size"),
                            Example=("full_name", lambda x: x.iloc[0]),
                            total_occurrences=("number_of_occurrences", "sum"),
                        )
                        .reset_index()
                    )
                    # Compute percentage for each subcluster relative to the current cluster size
                    top_subclusters["Percentage"] = (
                        top_subclusters["nb_aliases"] / len(cluster_df) * 100
                    ).round(1).astype(str) + "%"
                    # Convert noise label 0 to "0 (Noise)"
                    top_subclusters["SubCluster"] = top_subclusters["SubCluster"].apply(
                        lambda x: "0 (Noise)" if x == 0 else x
                    )
                    # Sort by size (descending) and select the top 10 subclusters
                    top_subclusters = top_subclusters.sort_values(
                        by="nb_aliases", ascending=False
                    ).head(10)
                    # Reorder columns as required: SubCluster, Percentage, nb_aliases, total_occurrences, Example
                    top_subclusters = top_subclusters[
                        [
                            "SubCluster",
                            "Percentage",
                            "nb_aliases",
                            "total_occurrences",
                            "Example",
                        ]
                    ]
                    st.markdown("#### Top 10 Subclusters in Cluster")
                    # Display the table with a fixed height to show the first 5 rows (scrollable up to 10)
                    st.dataframe(top_subclusters, height=250)

                # --- Data Sources Table ---
                subcluster_filter = st.text_input(
                    "Filter by SubCluster (exact)", key="subcluster_filter"
                )
                data_filter = st.text_input(
                    "Filter data sources (contains)", key="cluster_data_filter"
                )
                sample_df = cluster_df.copy()
                if st.session_state["sub_labels"] is not None:
                    sample_df["SubCluster"] = st.session_state["sub_labels"]
                if subcluster_filter:
                    try:
                        subcluster_val = int(subcluster_filter)
                        sample_df = sample_df[sample_df["SubCluster"] == subcluster_val]
                    except Exception as e:
                        st.error("SubCluster filter must be a number")
                if data_filter:
                    sample_df = sample_df[
                        sample_df["full_name"].str.contains(
                            data_filter, case=False, na=False
                        )
                    ]

                # Sort by number_of_occurrences in descending order
                sample_df = sample_df.sort_values(
                    by="number_of_occurrences", ascending=False
                )

                if len(sample_df) > 30:
                    # select top 30 rows
                    sample_df = sample_df.head(30)
                if "SubCluster" in sample_df.columns:
                    st.dataframe(
                        sample_df[
                            ["SubCluster", "full_name", "number_of_occurrences"]
                        ].rename(columns={"number_of_occurrences": "nb_occ"}),
                        height=600,
                        use_container_width=True,
                    )
                else:
                    st.dataframe(
                        sample_df[["full_name", "number_of_occurrences"]].rename(
                            columns={"number_of_occurrences": "nb_occ"}
                        ),
                        height=600,
                        use_container_width=True,
                    )
        else:
            st.info("Run DBSCAN clustering to enable cluster analysis.")


if __name__ == "__main__":
    main()
