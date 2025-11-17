import numpy as np
import pandas as pd



def build_df_results(searches: dict[str, object], annotations: dict[str, object]) -> pd.DataFrame:
    """Build a DataFrame with results from searches and annotations.

    Args:
        searches (dict[str, object]): searches data.
        annotations (dict[str, object]): annotations data.

    Returns:
        pd.DataFrame: DataFrame with results.
    """

    df_searches = pd.DataFrame(searches)
    df_annotations = pd.DataFrame(annotations)

    return (
        df_searches
        .merge(df_annotations, on=["query_id", "registry_id"], how="outer")
        .sort_values(by=["query_id", "search_rank"])
    )

def clean_df_results(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df
        .assign(
            search_rank=lambda df: df["search_rank"].fillna(np.inf), 
            annotation_label=lambda df: df["annotation_label"] == "YES", 
        )
    )

def compute_precision_k(df: pd.DataFrame) -> pd.DataFrame:
    """Compute precision@K metric column.
    
    Args:
        df (pd.DataFrame): dataframe to compute precision@K on.
        
    Returns:
        pd.DataFrame: dataframe with precision@K column.
        """
    df["precision_k"] = (
        df.groupby("query_id")["annotation_label"].transform("cumsum")
        /
        df["search_rank"]
    ).fillna(0)
    return df

def compute_recall_k(df: pd.DataFrame) -> pd.DataFrame:
    """Compute recall@K metric column.

    Args:
        df (pd.DataFrame): dataframe to compute recall@K on.

    Returns:
        pd.DataFrame: dataframe with recall@K column.
    """
    df["recall_k"] = (
        df.groupby("query_id")["annotation_label"].transform("cumsum")
        /
        df.groupby("query_id")["annotation_label"].transform("sum")
    ).fillna(0)
    return df

def compute_f1_k(df: pd.DataFrame) -> pd.DataFrame:
    """Compute F1@K metric column.

    Args:
        df (pd.DataFrame): dataframe to compute F1@K on.

    Returns:
        pd.DataFrame: dataframe with F1@K column.
    """
    df["f1_k"] = (
        2 * df["precision_k"] * df["recall_k"]
        /
        (df["precision_k"] + df["recall_k"])
    ).fillna(0)
    return df

def compute_partial_mean_average_precision(df: pd.DataFrame) -> pd.Series:
    """Compute mean average precision for each query.

    Args:
        df (pd.DataFrame): dataframe to compute mean average precision on.

    Returns:
        pd.Series: series with mean average precision for each query.
    """
    return (
        df

        .loc[lambda df: df["annotation_label"]]
        
        .groupby("query_id")
        ["precision_k"]
        .mean()

        .rename("mean_average_precision")
    )

def compute_global_mean_average_precision(df: pd.DataFrame) -> float:
    """Compute global mean average precision.

    Args:
        df (pd.DataFrame): dataframe to compute global mean average precision on.

    Returns:
        float: global mean average precision.
    """
    return (
        df

        .pipe(compute_partial_mean_average_precision)
        .mean()
    )
