import numpy as np
import pandas as pd



# setting

def generate_settings_grid(n_ranks: int, n_relevances: int, relevances_values: list[float]) -> list[dict]:
    """Generate a grid of settings for thresholding.

    Args:
        n_ranks (int): number of rank thresholds to try.
        n_relevances (int): number of relevance thresolds to try.
        relevances_values (list[float]): list of relevance values.

    Returns:
        list[dict]: grid of settings.
    """

    rank_thresholds = np.arange(1, n_ranks+1)
    relevance_thresholds = np.quantile(relevances_values, np.linspace(0, 1, n_relevances+1))
    
    return [
        *[{"rank": rank} for rank in rank_thresholds], 
        *[{"relevance": relevance} for relevance in relevance_thresholds], 
        *[
            {"rank": rank, "relevance": relevance}
            for rank in rank_thresholds
            for relevance in relevance_thresholds
        ], 
    ]

# Thresholding

def apply_thresholding(df: pd.DataFrame, setting: dict) -> pd.DataFrame:
    """Filter rows of the dataframe based on the thresholding setting.
    
    Args:
        df (pd.DataFrame): dataframe to filter.
        setting (dict): setting for thresholding.
        
    Returns:
        pd.DataFrame: filtered dataframe.
    """
    
    if "rank" in setting:
        df = df[df["search_rank"] <= setting["rank"]]
    if "relevance" in setting:
        df = df[df["search_relevance"] >= setting["relevance"]]
    return df

def compute_partial_metrics(df: pd.DataFrame) -> pd.Series:
    """Compute metrics per query for the dataframe.

    Args:
        df (pd.DataFrame): dataframe to compute metrics on.

    Returns:
        pd.Series: series with metrics per query.
    """

    return (
        df

        .groupby("query_id")
        .agg(
            n_registries=("registry_id", "nunique"), 
            precision=("precision_k", "last"), 
            recall=("recall_k", "last"), 
            f1=("f1_k", "last"), 
        )
    )
