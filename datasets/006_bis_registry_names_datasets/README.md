# Harmonized Registry Names Datasets

This folder provides datasets for 2 sets of registry names and their associated publications
1. The 237 EMA registry names
└── directly extracted from the Weaviate Database and from the EMA Data Catalogue.
2. 100 famous European registries. 
└── Those were extracted with the new prompt of registry_name, listed at least 10 times, and had at least one european country mentioned in their original goegraphic area field (aggregation of all geo areas of their publications)
└── Then their are successive updates of metadata fields alogn the way. First geographic area, then medical condition.

All large files are tracked using DVC for version control; ignore patterns are specified in each subfolder’s `.gitignore`.

## Directory Structure

```
datasets/
└── 006_bis_registry_names_datasets/
    ├── ema_reg_data/
    │   ├── ema_registries_dataset.json
    │   ├── ema_registries_dataset.json.dvc
    │   ├── .gitignore
    │   ├── ema_registries_dataset_publi_data.dvc
    │   └── ema_registries_dataset_publi_data/
    └── tool_reg_data/
        ├── 0_official_reg/
        │   ├── dedup_100_famous_european_registries.json
        │   ├── dedup_100_famous_european_registries.json.dvc
        │   ├── .gitignore
        │   ├── famous_european_registries_sample_publi_data.dvc
        │   └── famous_european_registries_sample_publi_data/
        ├── current_state_of_data/
        │   ├── dedup_100_famous_european_registries.json.dvc
        │   ├── famous_european_registries_sample_publi_data.dvc
        │   └── .gitignore
        ├── 1_update_geo_area/
        │   ├── dedup_100_famous_european_registries_with_geo_area.json.dvc
        │   └── .gitignore
        └── 2_update_medical_condition/
            ├── dedup_100_famous_european_registries_with_medical_condition.json.dvc
            ├── famous_european_reg_publi_data_with_medical_condition.dvc
            └── .gitignore
```

## Subfolder Details

### `ema_reg_data/`

* **`ema_registries_dataset.json`**: EMA registry metadata (name and original list of geo areas). DVC-tracked.
* **`ema_registries_dataset_publi_data/`**: Publication ata associated to the EMA registries, split into chunks of manageable JSON files.

### `tool_reg_data/current_state_of_data/`

* Contains current latest version of the 100 famous European registries and their publication data.

### `tool_reg_data/0_official_reg/`

* 100 famous European registries with their original metadata.

### `tool_reg_data/1_update_geo_area/`

* 100 famous European registries with updated geographic areas.

### `tool_reg_data/2_update_medical_condition/`

* 100 famous European registries with updated medical conditions.

## Usage Notes

* Use the **EMA data** as the canonical source for regulatory registry metadata.
* Use **tool\_reg\_data** stages to progressively enrich and refine the official registry list with geographic and clinical context.
* DVC-tracked files require `dvc pull` to fetch content locally.
* JSON splits (`1.json`, etc.) are provided to avoid loading extremely large JSON arrays in one go; they can be concatenated programmatically if needed.
