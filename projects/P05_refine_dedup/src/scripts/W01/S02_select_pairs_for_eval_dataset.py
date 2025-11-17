#!/usr/bin/env python3
import os
import time
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import click
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

from src.p05_refine_dedup import config
from src.p05_refine_dedup.utils.s3_io_functions import load_parquet_from_s3

# Logging setup
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Load environment variables from .env file
load_dotenv()


@click.command()
@click.option(
    "--s3_input_embeddings_parquet",
    type=str,
    required=True,
    help="S3 path to the embeddings parquet file.",
)
@click.option(
    "--output_pairs_xlsx",
    type=str,
    required=True,
    help="Output Excel file for selected pairs.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for shuffling.",
)
def select_pairs_for_eval_dataset(s3_input_embeddings_parquet, output_pairs_xlsx, seed):
    start_time = time.time()
    logger.warning("Loading embeddings from S3...")

    bucket_name = config.BUCKET_NAME_DEV
    folder_path = s3_input_embeddings_parquet.rsplit("/", 1)[0]
    file_name = s3_input_embeddings_parquet.rsplit("/", 1)[-1]
    df = load_parquet_from_s3(
        bucket_name=bucket_name,
        folder_path=folder_path,
        file_name=file_name,
    )

    logger.warning(f"Loaded {len(df)} embeddings.")

    # Sort and rank by number_of_occurrences
    df = df.sort_values(by="number_of_occurrences", ascending=False)
    df["rank"] = (
        df["number_of_occurrences"].rank(method="first", ascending=False).astype(int)
    )

    # Select top 5000
    max_rank = 5000
    df_singles = df[df["rank"] <= max_rank].reset_index(drop=True)
    df_singles = df_singles.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Prepare for pairing
    used_pairs = set()
    df_singles["alias"] = None
    df_singles["alias_number_of_occurrences"] = None
    df_singles["alias_object_id"] = None
    df_singles["similarity"] = None

    def find_closest_alias_not_in_pairs(row, df_candidates, used_pairs):
        object_id = row["object_id"]
        full_name_embedding = row["full_name_embedding"]
        similarities = cosine_similarity(
            [full_name_embedding], df_candidates["full_name_embedding"].tolist()
        )[0]
        sorted_indices = similarities.argsort()[::-1]
        for idx in sorted_indices:
            candidate_id = df_candidates.iloc[idx]["object_id"]
            if candidate_id == object_id:
                continue
            pair_key = tuple(sorted([object_id, candidate_id]))
            if pair_key not in used_pairs:
                used_pairs.add(pair_key)
                return (
                    df_candidates.iloc[idx]["full_name"],
                    df_candidates.iloc[idx]["number_of_occurrences"],
                    candidate_id,
                    similarities[idx],
                )
        return None, None, None, None

    logger.warning("Pairing registry names...")
    for idx, row in tqdm(
        df_singles.iterrows(), total=len(df_singles), desc="Creating pairs"
    ):
        alias, alias_occurrences, alias_id, similarity = (
            find_closest_alias_not_in_pairs(row, df_singles, used_pairs)
        )
        df_singles.at[idx, "alias"] = alias
        df_singles.at[idx, "alias_number_of_occurrences"] = alias_occurrences
        df_singles.at[idx, "alias_object_id"] = alias_id
        df_singles.at[idx, "similarity"] = similarity

    # Save results
    columns_to_save = [
        "object_id",
        "alias_object_id",
        "full_name",
        "alias",
        "number_of_occurrences",
        "alias_number_of_occurrences",
        "similarity",
    ]
    output_dir = Path(output_pairs_xlsx).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df_singles[columns_to_save].to_excel(output_pairs_xlsx, index=False)
    logger.warning(f"Saved {len(df_singles)} pairs to {output_pairs_xlsx}")

    elapsed = (time.time() - start_time) / 60
    logger.warning(f"Total time taken: {elapsed:.2f} minutes.")


if __name__ == "__main__":
    select_pairs_for_eval_dataset()
