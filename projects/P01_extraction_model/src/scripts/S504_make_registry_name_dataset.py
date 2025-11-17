import os
import json
import shutil
import click
import pandas as pd

@click.command()
@click.option('--all_in_one_dataset_json', type=click.Path(exists=True), required=True,
              help='Path to the main dataset JSONL file.')
@click.option('--corrected_registry_names_json', type=click.Path(exists=True), required=True,
                help='Path to the corrected registry names JSONL file.')
@click.option('--output_dir', type=click.Path(), required=True,
              help='Path to output Excel file with incorrect registry names.')
def main(all_in_one_dataset_json, corrected_registry_names_json, output_dir):
    all_in_one_df = pd.read_json(all_in_one_dataset_json).set_index("object_id")
    corrected_registry_names_json = pd.read_json(corrected_registry_names_json).set_index("object_id")

    all_in_one_df = all_in_one_df.join(corrected_registry_names_json, how="left")

    all_in_one_df.loc[:, "registry_name"] = all_in_one_df.a_registry_name
    all_in_one_df.loc[all_in_one_df.correct_registry_name.notnull(), "registry_name"] = all_in_one_df.correct_registry_name
    all_in_one_df.loc[:, "annotated_by"] = "mistral-large"
    all_in_one_df.loc[all_in_one_df.correct_registry_name.notnull(), "annotated_by"] = "ghinwa"
    all_in_one_df.loc[:, "annotation_comment"] = all_in_one_df.comment

    registry_name_dataset = all_in_one_df[[
        "pmid",
        "title",
        "abstract",
        "registry_name",
        "annotated_by",
        "annotation_comment",
    ]]

    # create output directory
    os.makedirs(output_dir, exist_ok=True)

    # save the dataset as json
    output_json = os.path.join(output_dir, "registry_names_dataset.json")
    registry_name_dataset.reset_index().to_json(output_json, orient="records", indent=4)


if __name__ == '__main__':
    main()
