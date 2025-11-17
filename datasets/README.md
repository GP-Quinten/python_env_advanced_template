# Datasets Directory

## Purpose

This `datasets/` folder contains the main datasets used and shared across multiple projects at Quinten Health. It also includes resources for manual annotation and evaluation. All datasets are version-controlled and managed using DVC (Data Version Control) to ensure reproducibility and traceability.

## High-Level Structure

The directory is organized by dataset types, annotation stages, and evaluation resources. The typical structure is as follows:

```
datasets/
├── 001_publications_dataset/
├── 002_registry_names_dataset/
├── 003_embeddings_of_registry_names_dataset/
├── 004_registry_names_deduplication_eval_dataset/
├── 005_evaluate_extraction_process_datasets/
├── 006_bis_registry_names_datasets/
└── ...
```

Below you will find a brief description and a lower-level file structure for each subfolder.

---

## 1. 001_publications_dataset/

**Description:**

Publications (mostly from PubMed or other biomedical sources) processed for downstream use in registry name extraction, outcome analysis, and related tasks.

**Contents:**

- `publications_dataset.jsonl`: Main publications dataset.
- `prod_publication_test_dataset.jsonl`: Test subset.
- [`README.md`](001_publications_dataset/README.md): Documentation.

---

## 2. 002_registry_names_dataset/

**Description:**

Manual and LLM-assisted annotation datasets for registry name extraction from biomedical publications.

**Contents:**

- `registry_names_dataset.json`: Final dataset of extracted registry names.
- `manual_annotation/`: Human annotations, error corrections, and annotation evaluation.
- `llm_annotation/`: Outputs from LLM registry name extraction runs.
- [`README.md`](002_registry_names_dataset/README.md): Documentation.

---

## 3. 003_embeddings_of_registry_names_dataset/

**Description:**

Precomputed embeddings (e.g., Mistral) for registry names, used for similarity search, deduplication, and clustering.

**Contents:**

- `Mistral_embeddings.parquet`: Embeddings for full registry names dataset.
- `Mistral_embeddings_sample.parquet`: Sample embeddings for fast evaluation.
- [`README.md`](003_embeddings_of_registry_names_dataset/README.md): Documentation.

---

## 4. 004_registry_names_deduplication_eval_dataset/

**Description:**

Evaluation datasets for deduplication and disambiguation of registry names, including ground truth for LLM judgment and performance analysis.

**Contents:**

- `evaluation_dataset.xlsx`: Main evaluation Excel file.
- `examples_for_prompt_LLM_as_a_judge_clean.xlsx`: Prompt and annotation examples for LLM as judge.
- [`README.md`](004_registry_names_deduplication_eval_dataset/README.md): Documentation.

---

## 5. 005_evaluate_extraction_process_datasets/

**Description:**

Manually annotated and final evaluation datasets for extraction processes, split by extraction target (registry name, registry related, medical condition, etc.).

**Contents:**

- `registry_name/`: Datasets related to registry name extraction, manual annotation, and evaluation.
- `registry_related/`: Datasets for extracting registry-related information, manual annotation, and evaluation.
- `medical_condition/`: Datasets for medical condition extraction, manual annotation, and evaluation.
- `outcome_measure/`: Datasets for outcome measure extraction, manual annotation, and evaluation.
- Each subfolder typically contains files like `manual_annotation/`, `final_eval_dataset.json`.
- [`README.md`](005_evaluate_extraction_process_datasets/README.md): Documentation.

---

## 6. 006_bis_registry_names_datasets/

**Description:**

Shared and harmonized registry datasets used for various registry projects and name deduplication efforts. Contains source, harmonized, and public registry datasets, including tool-specific splits and medical condition/geo annotation.

**Contents:**

- `ema_reg_data/`: EMA registry datasets (source, public, split, with DVC tracking and JSON/JSONL files).
- `tool_reg_data/`: Tool-driven registry data updates (e.g., with geo area or medical condition), organized by update stage.
- Each registry data subfolder may contain: `dedup_100_famous_european_registries*.json`, public datasets, sample splits.
- [`README.md`](006_bis_registry_names_datasets/README.md): Documentation.

---

## Notes

- All datasets are tracked by DVC. The `.dvc` files and `.gitignore` are essential for proper data management and reproducibility.
- Subfolders and files may change as projects evolve. See each subfolder’s own README (if present) for more details.
