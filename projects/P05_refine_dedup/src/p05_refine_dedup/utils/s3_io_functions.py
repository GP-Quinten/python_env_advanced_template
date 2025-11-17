import json
import os
import boto3
from botocore.exceptions import ClientError
import logging
import pandas as pd


def load_jsonl_from_s3(bucket_name: str, folder_path: str, file_name: str) -> list:
    file_path = os.path.join("s3://", bucket_name, folder_path, file_name)
    logging.warning("Data will be loaded from {}".format(file_path))
    s3 = boto3.client("s3")
    try:
        response = s3.get_object(
            Bucket=bucket_name, Key=os.path.join(folder_path, file_name)
        )
        data = response["Body"].read().decode("utf-8")
        json_lines = [json.loads(line) for line in data.splitlines()]
        return json_lines
    except ClientError as e:
        logging.error(f"Error loading JSONL from S3: {e}")
        return []  # Return an empty list on error


def upload_jsonl_to_s3(
    data: list, bucket_name: str, folder_path: str, file_name: str
) -> None:
    file_path = os.path.join("s3://", bucket_name, folder_path, file_name)
    logging.warning("Data will be saved to {}".format(file_path))
    s3 = boto3.client("s3")
    try:
        json_lines = "\n".join(json.dumps(item) for item in data)
        s3.put_object(
            Bucket=bucket_name,
            Key=os.path.join(folder_path, file_name),
            Body=json_lines.encode("utf-8"),
        )
    except ClientError as e:
        logging.error(f"Error uploading JSONL to S3: {e}")


def load_parquet_from_s3(
    bucket_name: str,
    folder_path: str,
    file_name: str,
) -> None:
    file_path = os.path.join("s3://", bucket_name, folder_path, file_name)
    logging.warning("Data will be loaded from {}".format(file_path))
    df = pd.read_parquet(file_path)
    return df


def upload_parquet_to_s3(
    df, bucket_name: str, folder_path: str, file_name: str
) -> None:
    file_path = os.path.join("s3://", bucket_name, folder_path, file_name)
    logging.warning("Data will be saved to {}".format(file_path))
    s3 = boto3.client("s3")
    try:
        buffer = df.to_parquet(index=False)
        s3.put_object(
            Bucket=bucket_name,
            Key=os.path.join(folder_path, file_name),
            Body=buffer,
        )
    except ClientError as e:
        logging.error(f"Error uploading Parquet to S3: {e}")
