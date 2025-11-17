# P06 Search Engine Project

## Short Description
This project develops and benchmarks a **search engine for registries** using:
- **Weaviate** for vector indexing and hybrid search
- **Snakemake** to orchestrate the data pipeline
- **MLflow** for experiment tracking
- **Python notebooks and scripts** for reporting and evaluation

The pipeline loads registries, processes annotations, runs benchmarking, and produces a final performance report with plots.

---

## Project Structure

```
P06_search_engine/
│
├── Snakefile                # Main pipeline definition
├── Makefile                 # Convenience commands (Weaviate, MLflow)
├── pyproject.toml           # Poetry project configuration
├── src/
│   ├── p06_search_engine/   # Core Python package
│   │   ├── config.py        # Paths & experiment settings
│   │   ├── indexing.py      # Weaviate indexing logic
│   │   ├── searching.py     # Search methods (vector/keyword/hybrid)
│   │   ├── assessing.py     # Metrics computation
│   │   ├── preparing.py     # Registry preprocessing
│   │   └── thresholding.py  # Threshold tuning
│   └── scripts/             # Pipeline steps:
│       ├── s01_load_registries.py
│       ├── s02_prepare_registries.py
│       ├── s03_index_registries.py
│       ├── s04_process_first_annotations.py
│       ├── s05_process_second_annotations.py
│       ├── s06_benchmark_search.py
│       ├── s07_fetch_ema_results.py
│       ├── s08_reporting.py
│       └── utils.py
│
├── notebooks/               # Prototyping & analysis notebooks
├── data/                    # Outputs from pipeline steps
└── datasets/                # Input datasets (registries, annotations)
```

---

## Scripts Overview

- `s01_load_registries.py`: Load raw registries from S3 and EMA JSON.
- `s02_prepare_registries.py`: Clean and filter registries.
- `s03_index_registries.py`: Index registries in Weaviate.
- `s04_process_first_annotations.py`: Normalize first round of queries and annotations.
- `s05_process_second_annotations.py`: Normalize second round of queries and annotations.
- `s06_benchmark_search.py`: Run searches across parameter grid and evaluate.
- `s07_fetch_ema_results.py`: Scrape EMA catalogue for third round of queries.
- `s08_reporting.py`: Aggregate MLflow results and generate plots/reports.
- `utils.py`: Shared utility functions for pipeline steps.

---

## Makefile Commands

Use the provided `Makefile` to run or stop local services:

```bash
# Start Weaviate (vector database)
make run-weaviate

# Stop Weaviate
make stop-weaviate

# Start MLflow UI
make run-mlflow

# Stop MLflow UI
make stop-mlflow
```

---

## Pipeline Overview

The Snakemake pipeline automates data preparation, benchmarking, and reporting.

### Steps

- **R02 – Build Registries Dataset**
  - `R02a_load_registries`: Load raw registries from S3 + EMA JSON
  - `R02b_prepare_registries`: Clean/filter registries

- **R04 – Process First Annotations**
  - Normalize queries + annotations (first round)

- **R05 – Process Second Annotations**
  - Normalize queries + annotations (second round)

- **R01 – Benchmark Search Engines**
  - Index registries in Weaviate
  - Run searches across parameter grid
  - Evaluate with annotations
  - Log metrics in MLflow

- **R03 – Fetch EMA Results**
  - Scrape EMA catalogue for third round of queries

- **R06 – Reporting**
  - Aggregate MLflow results
  - Generate plots (MAP, precision@k, recall, thresholds)
  - Save human-readable `report.txt` and plots in `data/R06_reporting/`

---

## Outputs

- Final performance report and plots are saved in `data/R06_reporting/`.
- MLflow tracks experiment metrics and parameter grids.

---
