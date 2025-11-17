import json
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import asyncio
import click
import hashlib
from P07_fuzzy_dedup.utils import weaviate_helpers
import weaviate.classes.query as wq
import time
from llm_backends import MistralEmbeddingBackend
import numpy as np

load_dotenv()

@click.command()
@click.option("--registry_json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--annotated_with_hash_json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--output_dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--collection_name", type=str, default="RegistryCollection")
@click.option("--auto_limit", type=int, default=3)
@click.option("--concurrency", type=int, default=3)
def cli(registry_json: Path, annotated_with_hash_json: Path, output_dir: Path, collection_name: str, auto_limit: int, concurrency: int):
    asyncio.run(main(registry_json, annotated_with_hash_json, output_dir, collection_name, auto_limit, concurrency))

class RateLimiter:
    def __init__(self, rate: int, per: float = 1.0):
        self.rate = rate
        self.per = per
        self._tokens = rate
        self._lock = asyncio.Lock()
        self._last_refill = time.monotonic()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            if elapsed >= self.per:
                self._tokens = self.rate
                self._last_refill = now
            while self._tokens <= 0:
                sleep_for = self.per - (now - self._last_refill)
                await asyncio.sleep(max(sleep_for, 0))
                now = time.monotonic()
                self._tokens = self.rate
                self._last_refill = now
            self._tokens -= 1

async def query_one(client, collection_name, row, auto_limit, rate_limiter):
    await rate_limiter.acquire()
    collection = client.collections.get(collection_name)
    query_text = f"{row['registry_name']} {row['acronym']}".strip()
    response = await collection.query.near_text(
        query=query_text,
        limit=300,
        return_metadata=wq.MetadataQuery(distance=True)
    )
    return row, response


def _make_pair_hash(left_id: str, right_id: str) -> str:
    ids = sorted([str(left_id), str(right_id)])
    return hashlib.md5(f"{ids[0]}::{ids[1]}".encode("utf-8")).hexdigest()


def unique_pairs_count(n):
    return n * (n - 1) // 2  # integer division


# New function: embed all unique registries using MistralEmbeddingBackend
def embed_unique_registries(registry_df: pd.DataFrame, mistral_api_key: str, model_config: dict = None, batch_size: int = 400) -> dict:
    """Embed every registry row and return a mapping from object_id (as str) to embedding (list of floats).

    Uses llm_backends.MistralEmbeddingBackend and supplies items with a 'custom_id' set to the object id.
    The function now batches requests to avoid "Too many tokens overall" errors from the Mistral API.
    """
    # respect a provided model_config, otherwise use a sensible default
    model_config = {"model": "mistral-embed"}

    backend = MistralEmbeddingBackend(api_key=mistral_api_key)

    items = []
    object_ids = []
    for _, row in registry_df.iterrows():
        prompt_text = f"{row.get('registry_name', '')} {row.get('acronym', '')}".strip()
        obj_id = str(row['object_id'])
        items.append({"prompt": prompt_text, "custom_id": obj_id})
        object_ids.append(obj_id)

    embeddings_raw = []

    # simple batching to avoid sending too many tokens in one request
    total_batches = (len(items) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(items), batch_size), desc="Embedding batches", total=total_batches):
        chunk = items[i:i + batch_size]
        try:
            chunk_embs = list(backend.infer_many(chunk, model_config=model_config))
        except Exception:
            # if the chunk still causes token errors, split it and retry in smaller sub-chunks
            if len(chunk) <= 1:
                raise
            mid = max(1, len(chunk) // 2)
            chunk_embs = []
            for j in range(0, len(chunk), mid):
                sub = chunk[j:j + mid]
                chunk_embs.extend(list(backend.infer_many(sub, model_config=model_config)))

        embeddings_raw.extend(chunk_embs)

    embeddings_map = {}
    for obj_id, item in zip(object_ids, embeddings_raw):
        emb = item.get("embedding")
        if isinstance(emb, np.ndarray):
            emb = emb.tolist()
        embeddings_map[obj_id] = emb

    return embeddings_map


# New async function: query by vector using precomputed embeddings
async def query_vector_one(client, collection_name, row, auto_limit, sem, embeddings_map):
    # use a semaphore to limit concurrent Weaviate vector queries
    async with sem:
        collection = client.collections.get(collection_name)
        emb = embeddings_map.get(str(row['object_id']))

        response = await collection.query.near_vector(
            near_vector=emb,
            limit=50,
            return_metadata=wq.MetadataQuery(distance=True)
        )
        return row, response


async def main(registry_json, annotated_with_hash_json, output_dir, collection_name, auto_limit, concurrency):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    registry_df = pd.read_json(registry_json)
    annotated_df = pd.read_json(annotated_with_hash_json)

    # deduplicate annotated_df by pair_hash_id, keeping the first possible occurrence
    # sort by label descending so that if there are conflicting labels, we keep the positive one
    annotated_df = annotated_df.sort_values("label", ascending=False).drop_duplicates("pair_hash_id", keep="first")
    annotated_df = annotated_df.set_index("pair_hash_id")

    # Connect to Weaviate (async)
    WEAVIATE__HTTP_HOST = os.getenv("WEAVIATE__HTTP_HOST")
    WEAVIATE__HTTP_PORT = os.getenv("WEAVIATE__HTTP_PORT")
    WEAVIATE__GRPC_HOST = os.getenv("WEAVIATE__GRPC_HOST")
    WEAVIATE__GRPC_PORT = os.getenv("WEAVIATE__GRPC_PORT")
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    client = await weaviate_helpers.get_async_client(
        http_host=WEAVIATE__HTTP_HOST,
        http_port=WEAVIATE__HTTP_PORT,
        grpc_host=WEAVIATE__GRPC_HOST,
        grpc_port=WEAVIATE__GRPC_PORT,
        mistral_api_key=MISTRAL_API_KEY
    )

    rate_limiter = RateLimiter(rate=6, per=1.0)
    pairs = []
    # use a semaphore to throttle concurrent vector queries
    sem = asyncio.Semaphore(concurrency)

    # Precompute embeddings for all registries
    model_config = {"model": "mistral-embed", "output_dimension": 128}
    embeddings_map = embed_unique_registries(registry_df, MISTRAL_API_KEY, model_config=model_config)

    async with client:
        # create tasks so we can monitor completion
        tasks = [
            asyncio.create_task(query_vector_one(client, collection_name, row, auto_limit, sem, embeddings_map))
            for _, row in registry_df.iterrows()
        ]

        pbar = tqdm(total=len(tasks), desc="Processing results")
        for coro in asyncio.as_completed(tasks):
            row, response = await coro
            for obj in response.objects:
                if str(obj.properties['object_id']) == str(row['object_id']):
                    continue  # skip self-match
                pair_hash_id = _make_pair_hash(row['object_id'], obj.properties['object_id'])
                pairs.append({
                    "pair_hash_id": pair_hash_id,
                    "left_name": row['registry_name'],
                    "left_acronym": row['acronym'],
                    "right_name": obj.properties.get('registry_name', ''),
                    "right_acronym": obj.properties.get('acronym', ''),
                    "left_object_id": str(row['object_id']),
                    "right_object_id": obj.properties['object_id'],
                    "distance": float(obj.metadata.distance),
                    "label": annotated_df.loc[pair_hash_id]['label'] if pair_hash_id in annotated_df.index else -1
                })

            pbar.update(1)
        pbar.close()

    pair_df = pd.DataFrame(pairs)
    # deduplicate pair by pair_hash_id, keeping the one with the smallest distance
    pair_df = pair_df.sort_values("distance").drop_duplicates("pair_hash_id", keep="first").reset_index(drop=True)

    # Ensure label column is int and fill missing values
    pair_df['label'] = pair_df['label'].fillna(-1).astype(int)

    output_json_path = output_dir / "semantic_candidate_pairs.json"
    pair_df.to_json(output_json_path, orient="records", indent=4)
    print(f"Saved semantic candidate pairs to {output_json_path}")

    # Calculate recall of positive pairs
    annotated_positive = annotated_df[annotated_df['label'] == 1]
    found_positive = pair_df[pair_df['label'] == 1]
    recall = len(found_positive) / len(annotated_positive) if len(annotated_positive) > 0 else None
    print(f"Recall of positive pairs: {recall}")

    # Save metadata with summary statistics only (at the end)
    num_pairs = len(pair_df)
    num_positives = int((pair_df['label'] == 1).sum())
    total_possible_pairs = unique_pairs_count(len(registry_df))
    percent_sampled = num_pairs / total_possible_pairs if total_possible_pairs > 0 else None
    metadata = {
        "num_pairs": num_pairs,
        "num_positives": num_positives,
        "recall": recall,
        "total_possible_pairs": total_possible_pairs,
        "percent_sampled": percent_sampled
    }
    metadata_json_path = output_dir / "metadata.json"
    with open(metadata_json_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to {metadata_json_path}")

if __name__ == "__main__":
    cli()