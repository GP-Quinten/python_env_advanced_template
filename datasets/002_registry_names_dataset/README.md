# Annotated Dataset: Registry Names from Weaviate Publication Sample 300

This dataset presents **annotated registry name references** based on a subset of the original *Weaviate Publication Sample 300*. It is aimed at supporting entity recognition and registry reference extraction tasks within biomedical and clinical research texts.

## Directory Structure

```
datasets/
├── 002_registry_names_dataset/
│   ├── registry_names_dataset.json
│   ├── manual_annotation/
│   │   ├── 00_incorrect_registry_name.json
│   │   └── 01_annotated_ghinwa_incorrect_registry_name.json
│   └── llm_annotation/
│       └── annotated_mistral_large_full.jsonl
```

## Overview

This annotated dataset identifies whether a scientific publication refers to a specific **registry** and, if so, extracts its name.

* **Source:** The original pool of 300 publications was randomly sampled from Quinten Health's Weaviate production database.
* **Annotation Model:** The initial annotations were generated automatically using the **Mistral Large** language model.
* **Manual Review:** A **subset of 76 samples** were manually reviewed and corrected by a domain expert (annotator: *Ghinwa*). These corrections are stored in the `manual_annotation/` folder.

## File Descriptions

### `registry_names_dataset.json`

This file contains 300 entries with fields annotated using the **Mistral Large** model.

| Field Name           | Description                                                                                    |
| -------------------- | ---------------------------------------------------------------------------------------------- |
| `object_id`          | Unique identifier from the Weaviate database.                                                  |
| `pmid`               | PubMed ID of the publication.                                                                  |
| `title`              | Title of the scientific article.                                                               |
| `abstract`           | Abstract text of the article.                                                                  |
| `registry_name`      | Extracted name of the registry mentioned in the text (or "NONE" if no registry is referenced). |
| `annotated_by`       | Annotation source: either `mistral-large` or `ghinwa`.                                         |
| `annotation_comment` | Additional notes from manual annotation (when applicable).                                     |

### `manual_annotation/01_annotated_ghinwa_incorrect_registry_name.json`

This file includes the **76 manually reviewed samples**, specifically focusing on cases where the Mistral-generated annotation was inaccurate or ambiguous.

### `manual_annotation/00_incorrect_registry_name.json`

List of cases identified as incorrect or ambiguous by the automated model—serving as the input for manual review.

### `llm_annotation/annotated_mistral_large_full.jsonl`

Structured output of registry-related field extractions produced by the Mistral Large model, for benchmarking or further development.

## Sample Entries

### Example 1: Correctly Identified Registry (Model)

```json
{
  "object_id": "26628d5a-af29-592f-aeee-04149681e275",
  "pmid": 19827149.0,
  "title": "Limited prognostic value of the 2004 International Union Against Cancer staging classification for adrenocortical carcinoma",
  "abstract": "In their article, Fassnacht et al reviewed the outcome relative to staging of a subset of 416 patients from the German Adrenocortical Carcinoma Registry...",
  "registry_name": "German Adrenocortical Carcinoma Registry",
  "annotated_by": "mistral-large",
  "annotation_comment": null
}
```

### Example 2: Manually Labeled as Not Registry-Based

```json
{
  "object_id": "2a729787-ec2c-55d0-8d54-0787e0de6311",
  "pmid": 10885554.0,
  "title": "Changing surgical principles for gastro-oesophageal reflux disease...",
  "abstract": "Morbidity and mortality after surgical treatment for gastro-oesophageal reflux disease (GORD)...",
  "registry_name": "NONE",
  "annotated_by": "ghinwa",
  "annotation_comment": "This is not a study based on registry or RWD. It is an opinion paper"
}
```

## Usage Notes

* Use the Mistral-annotated dataset for broad-scale NLP applications, entity recognition, and registry mention extraction in biomedical literature.
* For tasks requiring high precision or gold-standard benchmarking, use the manually reviewed subset.
* Fields with `registry_name = "NONE"` represent documents where no registry-based evidence could be confirmed.
* All files in this dataset are tracked via DVC for version control and reproducibility.
