# Weaviate Publication Sample 300

This dataset contains **300 randomly selected publication records** extracted from the More EUROPA production Weaviate database. It serves as a representative reference set for downstream annotation (such as registry name extraction) and modeling efforts in clinical and biomedical informatics.

## Dataset Overview

Each record represents metadata and content from a scientific publication—primarily clinical research and registry-based observational studies. The sample is designed to support annotation, benchmarking, and modeling tasks across multiple Quinten Health projects.

## Fields Description

Each record in the dataset includes the following fields:

| Field Name                 | Description                                                               |
| -------------------------- | ------------------------------------------------------------------------- |
| `object_id`                | Unique identifier in the Weaviate database.                               |
| `title`                    | Full title of the publication.                                            |
| `abstract`                 | Abstract text summarizing the publication's goals, methods, and findings. |
| `authors`                  | List of author names.                                                     |
| `pmid`                     | PubMed ID, a unique reference number.                                     |
| `publishing_date`          | Publication date (ISO 8601 format).                                       |
| `journal`                  | (Optional) Journal name.                                                  |
| `keywords`                 | Research keywords or phrases.                                             |
| `chemicals`                | Chemical substances mentioned.                                            |
| `mesh_terms`               | Medical Subject Headings assigned.                                        |
| `medical_condition`        | (Optional) Medical conditions discussed.                                  |
| `intervention`             | (Optional) Clinical intervention described.                               |
| `comparator`               | (Optional) Comparator in a study.                                         |
| `outcome_measure`          | (Optional) Outcomes measured.                                             |
| `category`                 | Publication classification (e.g., registry-based, clinical trial).        |
| `design_model`             | (Optional) Study design model.                                            |
| `design_model_description` | (Optional) Detailed description of the study design.                      |
| `population_description`   | (Optional) Description of the study population.                           |
| `population_size`          | (Optional) Number of participants.                                        |
| `population_sex`           | (Optional) Sex/gender breakdown.                                          |
| `population_age_group`     | (Optional) Age group or classification.                                   |
| `population_follow_up`     | (Optional) Duration or type of follow-up.                                 |
| `registry_related`         | Boolean flag—true if study is linked to a registry.                       |
| `data_source_name`         | (Optional) Name of the original data source.                              |
| `retrieved_from`           | Source of retrieval (e.g., PubMed).                                       |
| `summary`                  | (Optional) Short summary, if available.                                   |
| `pdf_link`                 | (Optional) Direct link to PDF.                                            |
| `redirection_link`         | External article link (e.g., PubMed URL).                                 |
| `geographical_area`        | (Optional) Relevant region/country.                                       |

## Sample Record

```json
{
  "object_id": "26a84273-b5f8-5900-b80e-675358053b5c",
  "title": "Patient-reported outcomes after lumbar epidural steroid injection for degenerative spine disease in depressed versus non-depressed patients.",
  "abstract": "Medical interventional modalities such as lumbar epidural steroid injections (LESIs)...",
  "authors": ["Elliott J Kim", "Silky Chotai", "David P Stonko", ...],
  "pmid": "27777051",
  "publishing_date": "2017-04-01T00:00:00+00:00",
  "keywords": ["Depression", "Epidural Steroid Injections", "Lumbar Spine", ...],
  "chemicals": ["Steroids"],
  "mesh_terms": ["Aged", "Anesthesia, Epidural", "Depression", ...],
  "registry_related": true,
  "category": "Publication refering to a registry",
  "redirection_link": "https://pubmed.ncbi.nlm.nih.gov/27777051",
  "retrieved_from": ["PubMed"]
}
```

## Usage Notes

* Fields such as `intervention`, `comparator`, `medical_condition`, and all population-related descriptors may be `null` or missing if not extracted or not available for a record.
* The dataset is suitable for training and evaluating NLP models (especially for biomedical text), conducting meta-analyses, developing clinical knowledge graphs, and information retrieval pipelines.
* All files in this dataset are tracked via DVC for version control and reproducibility.
