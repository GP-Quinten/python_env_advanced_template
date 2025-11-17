# README: Evaluate Extraction Process Datasets

This folder contains datasets used to **evaluate the accuracy and robustness of the registry-related entity extraction process**. These datasets are structured across multiple semantic categories (e.g., registry name, registry relatedness, medical condition, outcome measure, geographical area) and support both model evaluation and manual annotation workflows.

All files are versioned using **DVC** and tracked appropriately via `.gitignore`.

---

## Directory Structure

```
datasets/
├── 005_evaluate_extraction_process_datasets/
│   ├── registry_name/
│   │   ├── final_eval_dataset.json
│   │   └── manual_annotation/
│   │       ├── 01_manually_corrected_registry_name.xlsx
│   │       └── 00_incorrect_registry_name.json
│   ├── registry_related/
│   │   ├── final_eval_dataset.json
│   │   ├── final_test_dataset.json
│   │   └── manual_annotation/
│   │       ├── 01_manually_corrected_registry_related.xlsx
│   │       ├── test_set_annotated_registry_related.xlsx
│   │       └── 00_incorrect_registry_related.json
│   ├── medical_condition/
│   │   ├── medical_condition_eval_dataset.json
│   │   ├── final_eval_dataset.json
│   │   └── manual_annotation/
│   │       ├── 01_manually_corrected_medical_condition.xlsx
│   │       └── 00_incorrect_medical_condition.json
│   ├── outcome_measure/
│   │   ├── outcome_measure_eval_dataset.json
│   │   ├── final_eval_dataset.json
│   │   └── manual_annotation/
│   │       ├── 01_manually_corrected_outcome_measure.xlsx
│   │       └── 00_incorrect_outcome_measure.json
│   ├── geographical_area/
│   │   ├── geographical_area_eval_dataset.json
│   │   ├── final_eval_dataset.json
│   │   └── manual_annotation/
│   │       ├── 01_manually_corrected_geographical_area.xlsx
│   │       └── 00_incorrect_geographical_area.json
```

---

## Folder and File Descriptions

### 1. `registry_name/`

* **`final_eval_dataset.json`**: Registry name extraction evaluation dataset.
* **`manual_annotation/`**:

  * `01_manually_corrected_registry_name.xlsx`: Manually validated corrections.
  * `00_incorrect_registry_name.json`: Automatically detected incorrect predictions for review.

### 2. `registry_related/`

* **`final_eval_dataset.json`**: 1st version of the evaluation dataset for registry-related entity extraction (300 samples).
* **`final_test_dataset.json`**: 2nd version of the evaluation dataset for registry-related entity extraction (100 new samples).
* **`manual_annotation/`**:

  * `01_manually_corrected_registry_related.xlsx`: Human-validated corrections for version 1.
  * `test_set_annotated_registry_related.xlsx`: Manually annotated corrections for the test set (version 2 of eval dataset).
  * `00_incorrect_registry_related.json`: Samples (first evalt dataset version) identified as likely incorrect, to manually review and correct.

### 3. `medical_condition/`

* **`final_eval_dataset.json`**: Evaluation dataset for medical condition extraction.
* **`manual_annotation/`**:

  * `01_manually_corrected_medical_condition.xlsx`: Manual annotation corrections.
  * `00_incorrect_medical_condition.json`: Misclassified samples for correction.

### 4. `outcome_measure/`

* **`final_eval_dataset.json`**: Final evaluation dataset for outcome measure extraction.
* **`manual_annotation/`**:

  * `01_manually_corrected_outcome_measure.xlsx`: Corrected outcome measure references.
  * `00_incorrect_outcome_measure.json`: Incorrect entries flagged during automated extraction.

### 5. `geographical_area/`

* **`final_eval_dataset.json`**: Evaluation dataset for geographical area extraction.
* **`manual_annotation/`**:

  * `01_manually_corrected_geographical_area.xlsx`: Human-reviewed corrections.
  * `00_incorrect_geographical_area.json`: Initial error candidates from automatic model.

---

## Notes

* Each semantic category follows a common annotation structure: an automatically generated candidate set, error spotting via heuristics, and expert correction.
* Excel files are formatted for ease of manual annotation and auditability.
* JSON files are used for programmatic evaluation and benchmarking.
* All major datasets are DVC-tracked for reproducibility and team collaboration.

---

## Usage Recommendations

* Use `final_eval_dataset.json` files for quantitative benchmarking.
* Use `manual_annotation` Excel files for understanding annotation logic, edge cases, and qualitative error analysis.
* Incorporate `00_incorrect_*.json` files into iterative model fine-tuning and error diagnosis workflows.

---
