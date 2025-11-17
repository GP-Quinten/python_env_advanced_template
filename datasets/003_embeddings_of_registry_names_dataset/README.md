# Registry Names Embeddings Dataset

This folder contains **Mistral embeddings** for registry names extracted from Titale+abstract PubMed or Semantic scholar publications.
These embeddings were primarily used for deduplication and clusterig task stream 1.

## Directory Structure

```
datasets/
├── 003_embeddings_of_registry_names_dataset/
│   ├── Mistral_embeddings.parquet
│   ├── Mistral_embeddings_sample.parquet
│   ├── Mistral_embeddings.parquet.dvc
│   ├── Mistral_embeddings_sample.parquet.dvc
│   └── .gitignore
```

## Overview

* **Source:** Embeddings are generated for registry names annotated in the *Weaviate Publication Sample 300* dataset and related downstream annotation sets.
* **Embedding Model:** Vectors were generated using the Mistral Large language model.
* **File Format:** All embeddings are stored in Apache Parquet format.

## File Descriptions

### `Mistral_embeddings.parquet`

* Full embeddings for all registry names in the source dataset.
* Each row corresponds to a unique registry name reference.

### `Mistral_embeddings_sample.parquet`

* A 30k (out of 90k) randomly sampled subset of the full embeddings dataset for first empirical clustering and to find some positive pairs.

### `.dvc` files

* Track large data files via DVC for versioning, reproducibility, and team collaboration.

### `.gitignore`

* Ensures local data files are not tracked by git but only by DVC.

## Example Schema

Each row of the embedding files typically contains:

| Field Name  | Description                              |
| ----------- | ---------------------------------------- |
| `object_id` | Unique identifier of the registry entry. |
| `embedding` | Vector (array of floats) representation. |
| ...         | (Optional) Other metadata fields.        |

## Usage Notes

* Use embeddings for clustering, deduplication, similarity search, or as features in downstream ML models.
* For fast development or interactive exploration, start with the sample embeddings file.
* All data files are versioned with DVC for reproducibility.
