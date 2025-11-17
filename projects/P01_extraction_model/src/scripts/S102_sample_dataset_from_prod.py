import weaviate
import json
import random
import os
import io
import click
from tqdm import tqdm
from datetime import datetime
from weaviate.classes.query import Filter
from pprint import pprint

from dotenv import load_dotenv

# set random seed for reproducibility
random.seed(42)

# Load environment variables
load_dotenv()

# Custom function to handle datetime serialization
def serialize(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def sample_from_collection(publication_collection, n_samples):
    """
    Sample n_samples from iterator
    """
    total_count = publication_collection.aggregate.over_all(total_count=True).total_count
    print(f"Total objects in collection: {total_count}")

    iterator = publication_collection.iterator()

    print("Collecting UUIDs")
    uuids = []

    pbar = tqdm(total=total_count)
    for i, item in enumerate(iterator):
        uuids.append(item.uuid)
        pbar.update(1)

    pbar.close()

    print(f"Collected {len(uuids)} UUIDs")
    sampled = random.sample(uuids, n_samples)

    print(f"Sampled {len(sampled)} UUIDs")

    return sampled

@click.command()
@click.option('--n_samples', help='Number of objects to fetch initially', type=int)
@click.option('--collection_name', help='Collection name to fetch data from')
@click.option('--output_jsonl', help='Output JSONL file')
@click.option('--seed', default=42, help='Random seed for reproducibility')
def fetch_weaviate_data(n_samples, collection_name, output_jsonl, seed):
    # Set random seed for reproducibility
    random.seed(seed)

    # Configuration dictionary
    weaviate_conf = {
        "http_host": os.getenv("WEAVIATE__HTTP_HOST"),
        "http_port": os.getenv("WEAVIATE__HTTP_PORT"),
        "http_secure": True,
        "grpc_host": os.getenv("WEAVIATE__GRPC_HOST"),
        "grpc_port": os.getenv("WEAVIATE__GRPC_PORT"),
        "grpc_secure": False,
        "skip_init_checks": True,
    }

    # Connect to Weaviate
    weaviate_client = weaviate.connect_to_custom(**weaviate_conf)

    # Access the specified collection
    print(f"Fetching data from collection: {collection_name}")
    publication_collection = weaviate_client.collections.get(collection_name)
    
    # Sample UUIDs
    random_uuid = sample_from_collection(publication_collection, n_samples)

    # Fetch detailed objects by sampled UUIDs
    print(f"Fetching {len(random_uuid)} objects")
    response = publication_collection.query.fetch_objects(
        filters=Filter.by_id().contains_any(random_uuid)
    )
    
    print(f"Fetched {len(response.objects)} objects")

    # Close connection
    weaviate_client.close()

    # make sure output directory exists
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    # Save to JSONL file
    with io.open(output_jsonl, "w", encoding="utf-8") as f:
        for obj in response.objects:
            f.write(json.dumps(obj.properties, default=serialize) + "\n")

    print(f"Saved {len(response.objects)} objects to {output_jsonl}")

if __name__ == "__main__":
    fetch_weaviate_data()
