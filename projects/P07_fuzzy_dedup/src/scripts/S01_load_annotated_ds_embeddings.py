#!/usr/bin/env python3
# File: src/scripts/S10a_preload_embeddings.py

import click
from pathlib import Path
import pandas as pd
from P07_fuzzy_dedup import config
from P07_fuzzy_dedup.utils.s3_io_functions import load_parquet_from_s3


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--s3_input_embeddings", type=str, required=True,
              help="S3 path to embeddings parquet file.")
@click.option("--output_path", type=click.Path(dir_okay=False, path_type=Path),
              required=True, help="Where to save processed embeddings parquet.")
def main(s3_input_embeddings: str, output_path: Path):
    bucket_name = config.BUCKET_NAME_DEV
    folder_path = s3_input_embeddings.rsplit("/", 1)[0]
    file_name = s3_input_embeddings.rsplit("/", 1)[-1]

    df = load_parquet_from_s3(bucket_name, folder_path, file_name)
    # show column names
    # click.echo(f"Columns in loaded dataframe: {df.columns.tolist()}")
    # if present, rename full_name and ful_name_embeding to registry_name and registry_name_embedding
    if "full_name" in df.columns and "full_name_embedding" in df.columns:
        df = df.rename(columns={
            "full_name": "registry_name",
            "full_name_embedding": "registry_name_embedding"
        })
    df=df.loc[
        :, ["object_id", "registry_name_embedding"]
    ]
    df["object_id"] = df["object_id"].astype(str)

    # ensure embeddings are lists
    df["registry_name_embedding"] = df["registry_name_embedding"].apply(
        lambda x: x.tolist() if hasattr(x, "tolist") else x
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    click.echo(f"[SUCCESS] Saved processed embeddings to {output_path}")


if __name__ == "__main__":
    main()
