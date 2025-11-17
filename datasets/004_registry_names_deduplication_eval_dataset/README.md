# Registry Names Deduplication Evaluation Dataset

This folder contains resources for evaluating the deduplication and disambiguation of registry names extracted from publications (from our initial weaviate DB).
It contains examples that were given to the LLM as a judge to complete its task of assessing 'same' or 'different' to registry names.
Then it contains the evaluation dataset (8000 pairs, 50% are positive, 50% are negative) that will be used for the training and the final performance assessment of our deduplication algorithm.

## Directory Structure

```
datasets/
├── 004_registry_names_deduplication_eval_dataset/
│   ├── evaluation_dataset.xlsx
│   ├── examples_for_prompt_LLM_as_a_judge_clean.xlsx
│   ├── evaluation_dataset.parquet
│   ├── evaluation_dataset.xlsx.dvc
│   ├── examples_for_prompt_LLM_as_a_judge_clean.xlsx.dvc
│   ├── evaluation_dataset.parquet.dvc
│   └── .gitignore
```

## Overview

* **Purpose:** Benchmarking registry name deduplication (entity matching, clustering, disambiguation) in biomedical text mining workflows.
* **Data Sources:** Derived from the annotated registry references in the *Weaviate Publication Sample 300* and related datasets. Datasets include both model-generated candidates and manually curated/validated cases.
* **Typical Use Cases:**

  * LLM or human evaluation of string similarity, clustering, or matching algorithms.
  * Prompting large language models to act as judges for entity deduplication.
  * Comparison of automated methods versus manual curation.

## File Descriptions

### `evaluation_dataset.xlsx`

Main evaluation set. Contains registry name candidate pairs (or groups), gold labels for true matches, and supporting metadata. Used for manual, LLM, or programmatic evaluation of deduplication.

### `examples_for_prompt_LLM_as_a_judge_clean.xlsx`

Clean examples curated for constructing prompts or gold standards for LLM-based evaluation of registry name similarity and matching. Useful for prompt engineering and establishing evaluation standards.

### `evaluation_dataset.parquet`

Parquet-formatted version of the main evaluation set for efficient programmatic access.

### `.dvc` files

DVC versioning files for all large dataset files, ensuring traceability and reproducibility.

### `.gitignore`

Ensures large data files are tracked only by DVC and not by git.

## Usage Notes

* Use these datasets for evaluating or developing registry deduplication algorithms, benchmarking LLM-based entity resolution, or for human annotation workflows.
* The Excel and Parquet formats are intended for both manual annotation and automated analysis.
* All data files are versioned using DVC.
