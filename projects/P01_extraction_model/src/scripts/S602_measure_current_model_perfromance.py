import os
import json
import click
import pandas as pd

from scripts.S601_measure_model_perfromance import calculate_metrics

@click.command()
@click.option('--all_in_one_dataset_json', type=click.Path(exists=True), required=True,
              help='Path to the all-in-one dataset JSON file.')
@click.option('--registry_names_dataset_json', type=click.Path(exists=True), required=True,
              help='Path to the registry names dataset JSON file.')
@click.option('--output_dir', type=click.Path(), required=True,
              help='Directory to save the performance metrics.')
def main(all_in_one_dataset_json, registry_names_dataset_json, output_dir):
    # Load datasets
    raw_dataset_dt = pd.read_json(all_in_one_dataset_json).set_index("object_id")
    registry_names_dataset = pd.read_json(registry_names_dataset_json).set_index("object_id")

    registry_names_dataset.loc[:, "pred_registry_name"] = raw_dataset_dt["data_source_name"]

    registry_names_dataset.loc[:, "is_correct"] = registry_names_dataset.registry_name == registry_names_dataset.pred_registry_name
    is_correct_by_llm = all_in_one_dataset_dt.llm_response == 'same'

    registry_names_dataset.loc[is_correct_by_llm, "is_correct"] = True

    metrcis = calculate_metrics(registry_names_dataset)

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    print(metrcis)
    # Save performance metrics
    metrics_output = os.path.join(output_dir, "performance_metrics.json")
    with open(metrics_output, 'w') as f:
        json.dump(metrcis, f, indent=4)

    print(f"Performance metrics saved to {metrics_output}")

if __name__ == '__main__':
    main()