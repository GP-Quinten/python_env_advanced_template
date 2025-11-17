import logging
import pandas as pd
import os
import boto3


def load_file_from_s3(
    bucket_name: str,
    folder_path: str,
    file_name: str,
) -> None:
    file_path = os.path.join("s3://", bucket_name, folder_path, file_name)
    logging.warning("Data will be loaded from {}".format(file_path))
    df = pd.read_parquet(file_path)
    return df


def upload_file_to_s3(
    df: pd.DataFrame,
    bucket_name: str,
    folder_path: str,
    file_name: str,
) -> None:
    file_path = os.path.join("s3://", bucket_name, folder_path, file_name)
    logging.warning("Data will be saved to {}".format(file_path))
    df.to_parquet(file_path)
    return None
