import os
import json
import click
import pandas as pd


def calculate_metrics(registry_names_dataset):
    cln_registry_name = registry_names_dataset.registry_name.str.lower().str.replace(r'[^a-z]', '').str.strip()
    cln_pred_registry_name = registry_names_dataset.pred_registry_name.str.lower().str.replace(r'[^a-z]', '').str.strip()
    is_correct_partially = []
    for i, (r, p) in enumerate(zip(cln_registry_name, cln_pred_registry_name)):
        if r in p or p in r:
            is_correct_partially.append(True)
        else:
            is_correct_partially.append(False)

    registry_names_dataset.loc[is_correct_partially, "is_correct"] = True

    metrcis = {}
    none_names = ["Not specified", "NONE"]

    # dataset
    metrcis["dataset"] = {
        "total": int(len(registry_names_dataset)),
        "n_none": int((registry_names_dataset.registry_name == "NONE").sum()),
        "n_not_specified": int((registry_names_dataset.registry_name == "Not specified").sum()),
        "n_specified": int((registry_names_dataset.registry_name.isin(none_names) == False).sum())
    }

    # general
    metrcis["general"] = {
        "total": int(len(registry_names_dataset)),
        "correct": int(registry_names_dataset.is_correct.sum()),
        "precision_perc": registry_names_dataset.is_correct.mean() * 100
    }

    # None precision
    metrcis["NONE"] = {
        "total": int((registry_names_dataset.registry_name == "NONE").sum()),
    }
    y_true = registry_names_dataset.registry_name == "NONE"
    y_pred = registry_names_dataset.pred_registry_name == "NONE"

    TP = ((y_true == True) & (y_pred == True)).sum()
    FP = ((y_true == False) & (y_pred == True)).sum()
    FN = ((y_true == True) & (y_pred == False)).sum()
    TN = ((y_true == False) & (y_pred == False)).sum()
    metrcis["NONE"] = {
        "recall": float(TP / (TP + FN)),
        "precision": float(TP / (TP + FP)),
        "TP": int(TP),
        "FP": int(FP),
        "FN": int(FN),
        "TN": int(TN)
    }

    # only related
    none_mask = registry_names_dataset.registry_name.isin(none_names)

    metrcis["only_related"] = {
        "total": int((~none_mask).sum()),
        "correct": int(registry_names_dataset[~none_mask].is_correct.sum()),
        "precision_perc": registry_names_dataset[~none_mask].is_correct.mean() * 100
    }

    # only not related
    metrcis["only_not_related"] = {
        "total": int(none_mask.sum()),
        "correct": int(registry_names_dataset[none_mask].is_correct.sum()),
        "precision_perc": registry_names_dataset[none_mask].is_correct.mean() * 100
    }

    return metrcis

@click.command()
@click.option('--all_in_one_dataset_json', type=click.Path(exists=True), required=True,
              help='Path to the all-in-one dataset JSON file.')
@click.option('--registry_names_dataset_json', type=click.Path(exists=True), required=True,
              help='Path to the registry names dataset JSON file.')
@click.option('--output_dir', type=click.Path(), required=True,
              help='Directory to save the performance metrics.')
def main(all_in_one_dataset_json, registry_names_dataset_json, output_dir):
    # Load datasets
    all_in_one_dataset_dt = pd.read_json(all_in_one_dataset_json).set_index("object_id")
    registry_names_dataset = pd.read_json(registry_names_dataset_json).set_index("object_id")

    registry_names_dataset.loc[:, "pred_registry_name"] = all_in_one_dataset_dt.b_registry_name

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