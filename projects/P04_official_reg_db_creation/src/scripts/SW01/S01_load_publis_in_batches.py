import logging
import time
import json
from pathlib import Path
import weaviate
import click
import random

from src.p04_official_reg_db_creation import config

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--collection", help="Collection name to fetch data from")
@click.option(
    "--output_jsonl_template",
    type=str,
    help="Output file name template where the batch JSONL files will be saved.",
)
def load_publis_in_batches(
    collection: str,
    output_jsonl_template: str,
):
    # track time of execution
    start_time = time.time()

    random.seed(42)
    batch_size = 2000

    weaviate_client = weaviate.connect_to_custom(**config.WEAVIATE_PROD_CONF)
    collections = weaviate_client.collections  #
    # load publications
    collection_publications = collections.get(collection)
    # load data source names
    items = []
    for item in collection_publications.iterator(include_vector=False):
        # Extract subset of properties
        items.append(
            {
                k: v
                for k, v in item.properties.items()
                if k
                in [
                    "object_id",
                    "title",
                    "abstract",
                    "data_source_name",
                ]
            }
        )
    # close weaviate connection
    weaviate_client.close()

    shuffled_items = items.copy()
    random.shuffle(shuffled_items)

    # ensure output directory exists (extract path of output_jsonl_template)
    output_dir = Path(output_jsonl_template).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_number = 1
    for i in range(0, len(shuffled_items), batch_size):
        batch_items = shuffled_items[i : i + batch_size]
        output_file = output_dir / f"{batch_number}.jsonl"
        with output_file.open("w") as f:
            for item in batch_items:
                f.write(json.dumps(item) + "\n")
        batch_number += 1
    # logging number of batches
    logging.warning(f"Number of batches created: {batch_number - 1}")
    logging.warning(f"Loading time: {(time.time() - start_time):.2f} seconds")


if __name__ == "__main__":
    load_publis_in_batches()
