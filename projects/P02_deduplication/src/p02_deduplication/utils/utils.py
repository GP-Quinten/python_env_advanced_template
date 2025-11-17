import time
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import logging


def run_dbscan(embeddings, eps, min_samples):
    """
    Run DBSCAN clustering on the provided embeddings.
    Increments labels by 1 so that noise (-1) becomes 0.
    Returns the labels and the computation time.
    """
    start_time = time.time()
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = dbscan.fit_predict(embeddings)
    labels = labels + 1  # convert noise (-1) to 0, clusters start at 1
    comp_time = time.time() - start_time
    return labels, comp_time
