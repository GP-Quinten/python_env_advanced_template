import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# print working directory
print(Path.cwd())

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.cluster import DBSCAN
from config import config
from src.utils.io_functions import load_file_from_s3


import umap
from sklearn.cluster import DBSCAN

# from sklearn.manifold import TSNE

folder_path = (
    "registry_data_catalog_experiments/task_1_deduplication/publications_embeddings/"
)
file_name = "data_source_name_MistralEmbed_embeddings.parquet"


def umap_reduction(
    embeddings, n_neighbors=10, min_dist=0.1, n_components=2, random_state=42
):
    """
    Performs dimensionality reduction on the given embeddings using UMAP.

    Parameters:
    - embeddings (array-like): The input data to reduce, typically high-dimensional.
    - n_neighbors (int): The number of neighboring points used in the local manifold approximation.
      Higher values can result in broader views of the manifold, while lower values preserve more
      of the local structure. Typical values range from 5 to 50.
    - min_dist (float): The minimum distance between points in the low-dimensional space.
      Smaller values preserve finer topological features, with values typically between 0.001 and 0.5.
    - n_components (int): The number of dimensions of the space where the data will be embedded.
      Typically 2 for visualization or higher for more complex reductions.
    - random_state (int): The seed used by the random number generator for reproducibility.

    Returns:
    - reduced_embeddings (array-like): The embeddings reduced to `n_components` dimensions.
    """
    reducer = umap.UMAP(
        metric="cosine",
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=42,
        n_jobs=1,
    )
    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings


def clustering(reduced_embeddings, eps=0.15, min_samples=3):
    """
    Perform DBSCAN clustering on reduced embeddings.

    Parameters
    ----------
    reduced_embeddings : array-like
        The low-dimensional data to cluster.
    eps : float, optional
        The maximum distance between two samples for them to be considered neighbors.
    min_samples : int, optional
        The number of samples required in a neighborhood to form a cluster.

    Returns
    -------
    labels : array-like of shape (n_samples,)
        Cluster labels for each point. Noisy samples are labeled -1.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1)
    labels = dbscan.fit_predict(reduced_embeddings)
    return labels


# Set page configuration for Streamlit
st.set_page_config(layout="wide")
np.random.seed(42)


def plot_embeddings_interactive(df, fix_size=False):
    """
    Create an interactive scatter plot of embeddings using Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'X', 'Y', 'cluster', 'data_source_name', 'color', and 'n_products'.
    fix_size : bool, optional
        If True, sets a fixed marker size.

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        The Plotly figure for display.
    """
    # Pass cluster and data_source_name as custom data for the hovertemplate
    fig = px.scatter(
        df,
        x="X",
        y="Y",
        custom_data=["cluster", "Name"],
        color="color",
        size="n_products",
        width=1200,
        height=800,
    )
    if fix_size:
        fig.update_traces(
            marker=dict(size=12, opacity=0.8), selector=dict(mode="markers")
        )
    # Hover template: first line is cluster number, second line is data source name
    fig.update_traces(
        hovertemplate="<b>Cluster: %{customdata[0]}</b><br><br>%{customdata[1]}<extra></extra>",
    )
    fig.update_layout(
        title="Text Embeddings Visualized with UMAP",
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
    )
    return fig


def insert_newlines_every_three_words(s):
    """
    Insert line breaks approximately every four words (up to 50 words).

    Parameters
    ----------
    s : str
        The string to format.

    Returns
    -------
    str
        The formatted string with <br> tags.
    """
    words = s.split()[:50]
    chunks = [" ".join(words[i : i + 4]) for i in range(0, len(words), 4)]
    return "<br>".join(chunks)


def main():
    """
    Main function for running the Streamlit app.

    - Loads a Parquet file containing embedded data.
    - Allows users to set UMAP and DBSCAN parameters.
    - Performs dimensionality reduction (via UMAP) and clustering (via DBSCAN).
    - Displays an interactive Plotly chart of points colored by their DBSCAN cluster label.
    """
    st.title("Embedding Visualization Tool")

    # Path to the Parquet file containing embeddings
    df = load_file_from_s3(
        bucket_name=config.BUCKET_NAME_DEV,
        folder_path=folder_path,
        file_name=file_name,
    )  # .head(500)

    # filter out all rows where data_source_name.lower() contains 'not specified'
    df = df[~df["data_source_name"].str.lower().str.contains("not specified")]

    st.write(df.head())

    # Extract embeddings from the 'description_embedding' column
    embeddings = np.vstack(df["embedding"].values)

    # Sidebar controls for UMAP parameters
    st.sidebar.header("UMAP Parameters")
    # Generate 50 values between 2 and 200 with exponential spacing
    neighbors_options = np.geomspace(2, 200, num=50, endpoint=True).astype(int)
    n_neighbors = st.sidebar.select_slider(
        "Number of Neighbors (Exponential)", options=neighbors_options
    )
    # st.write("Number of Neighbors:", n_neighbors)

    # Define options for each range:
    # 0.001 to 0.009 with step 0.001
    options1 = np.linspace(0.001, 0.009, num=9)
    # 0.01 to 0.09 with step 0.01
    options2 = np.linspace(0.01, 0.09, num=9)
    # 0.1 to 1.0 with step 0.1
    options3 = np.linspace(0.1, 1.0, num=10)

    # Combine the options into one array:
    min_dist_options = np.concatenate([options1, options2, options3])

    # Define a formatting function:
    def format_min_dist(val):
        if val < 0.01:
            return f"{val:.3f}"
        elif val < 0.1:
            return f"{val:.2f}"
        else:
            return f"{val:.1f}"

    # Create the select slider with custom formatting:
    min_dist = st.sidebar.select_slider(
        "Minimum Distance (Non-linear)",
        options=min_dist_options,
        format_func=format_min_dist,
    )

    # st.write("Minimum Distance:", min_dist)

    # Sidebar controls for DBSCAN parameters
    st.sidebar.header("DBSCAN Parameters")
    eps = st.sidebar.slider(
        "EPS", min_value=0.001, max_value=1.0, value=0.15, step=0.001
    )
    min_samples = st.sidebar.slider(
        "Minimum samples", min_value=1, max_value=10, value=3
    )

    if st.button("Run Analysis"):

        with st.spinner("Running UMAP..."):
            reduced_embeddings = umap_reduction(
                embeddings, n_neighbors=n_neighbors, min_dist=min_dist
            )

        with st.spinner("Running DBSCAN..."):
            clusters = clustering(reduced_embeddings, eps=eps, min_samples=min_samples)

        # Add the UMAP coordinates and cluster labels to the DataFrame
        df["X"] = reduced_embeddings[:, 0]
        df["Y"] = reduced_embeddings[:, 1]
        df["cluster"] = clusters

        # Generate hover text for each point
        df["Name"] = df.apply(
            lambda row: insert_newlines_every_three_words(row["data_source_name"]),
            axis=1,
        )

        # Plot the clusters only
        st.plotly_chart(
            plot_embeddings_interactive(
                df.assign(color=df["cluster"].astype(str)).assign(n_products=1),
                fix_size=True,
            )
        )


if __name__ == "__main__":
    main()
