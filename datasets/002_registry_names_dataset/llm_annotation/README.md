# LLM Annotation - Mistral Large (Full)

This file documents the dataset located at:

```
datasets/002_registry_names_dataset/llm_annotation/annotated_mistral_large_full.jsonl
```

## Description

This dataset consists of structured annotations generated using the **Mistral Large** language model. The annotations aim to extract key clinical and methodological features from scientific publication metadata, particularly focused on registry-based studies.

Each entry corresponds to a single publication and includes detailed entity-level fields commonly found in registry-based observational research.

## File Format

- Format: **JSON Lines (`.jsonl`)**
- Each line represents a JSON object with the following structure:
  - `object_id`: A unique identifier linking back to the original publication record.
  - `llm_annotation`: A dictionary of extracted attributes and values.

## Extracted Fields

The following fields are included under the `llm_annotation` object:

| Field Name              | Description |
|------------------------|-------------|
| `Registry related`     | Indicates whether the study is registry-related (`yes` / `no`). |
| `Registry name`        | The name of the registry, or "Not specified" if not explicitly mentioned. |
| `Population description` | Free-text summary of the study population. |
| `Intervention`         | Description of the clinical or therapeutic intervention under study. |
| `Comparator`           | Description of the comparator group (if any). |
| `Outcome measure`      | Key outcome metrics or endpoints evaluated. Multiple outcomes separated by `|`. |
| `Medical condition`    | Primary condition or disease area under study. |
| `Population sex`       | Sex/gender breakdown (e.g., `Male`, `Female`, or both). |
| `Population age group` | Age range or classification (e.g., `Adult`, `Elderly`). |
| `Design model`         | Study design (e.g., `Cohort studies`, `Case-control`). |
| `Population size`      | Number of participants (integer). |
| `Geographical area`    | Geographic location of the study, or `Not specified`. |
| `Population follow-up` | Duration or characteristics of follow-up, if mentioned. |

## Example Record

```json
{
  "object_id": "098beac1-681c-5153-9259-7754896c3e15",
  "llm_annotation": {
    "Registry related": "yes",
    "Registry name": "Not specified",
    "Population description": "Patients starting with evolocumab or alirocumab in a lipid clinic",
    "Intervention": "PCSK9 monoclonal antibodies (evolocumab or alirocumab)",
    "Comparator": "Male vs Female",
    "Outcome measure": "LDL-c reduction | Side effects | PCSK9 mAbs discontinuation",
    "Medical condition": "Hypercholesterolemia",
    "Population sex": "Male | Female",
    "Population age group": "Adult",
    "Design model": "Cohort studies",
    "Population size": 436,
    "Geographical area": "Not specified",
    "Population follow-up": "Not specified"
  }
}
```

## Usage Notes

- This dataset enables downstream tasks like document classification, information extraction evaluation, and registry linkage verification.
- Fields such as `Registry name`, `Geographical area`, and `Follow-up` may be marked as `Not specified` if the LLM could not confidently extract that information.
- Use in combination with manual annotations for benchmarking or fine-tuning extraction accuracy.
