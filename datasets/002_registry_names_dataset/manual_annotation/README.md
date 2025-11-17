# README: LLM Registry Name Extraction Evaluation

## Objective

This task is designed to **evaluate the accuracy of two large language models (LLMs)** in extracting registry names from scientific publications. The accompanying Excel file contains structured data and is intended to facilitate manual annotation and comparative performance analysis.

## Dataset Structure

Each record in the dataset contains the following columns:

| Column Name             | Description |
|-------------------------|-------------|
| `object_id`             | Unique identifier for the publication record. |
| `title`                 | Title of the scientific paper. |
| `abstract`              | Abstract of the paper containing clinical or research details. |
| `a_registry_name`       | Registry name predicted by Method A (LLM A). |
| `b_registry_name`       | Registry name predicted by Method B (LLM B). |
| `correct_registry_name` | Column to be annotated with the verified registry name. |

## Annotation Guidelines

Annotators are asked to review each record and determine the **correct registry name** according to the following rules:

1. **If `a_registry_name` is correct**, copy it into `correct_registry_name`.
2. **If `b_registry_name` is correct**, copy it into `correct_registry_name`.
3. **If both predictions are incorrect**, manually type the correct registry name.
4. **If the paper is unrelated to any registry**, set `correct_registry_name` to `"NONE"`.
5. **If a registry is mentioned but not named**, set `correct_registry_name` to `"Not Specified"`.

These rules ensure consistency across annotations and create a high-quality ground truth for model evaluation.

## Outcome

The final annotated dataset will serve as a **benchmark** to:

- Quantify and compare the performance of two LLMs on a practical NER task.
- Identify common failure modes in registry name extraction.
- Improve registry reference extraction pipelines used in biomedical NLP.
