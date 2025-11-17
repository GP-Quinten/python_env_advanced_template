import os
import json
import shutil
import click
import pandas as pd

@click.command()
@click.option('--all_in_one_dataset_json', type=click.Path(exists=True), required=True,
              help='Path to the main dataset JSONL file.')
@click.option('--output_dir', type=click.Path(), required=True,
              help='Path to output Excel file with incorrect registry names.')
def main(all_in_one_dataset_json, output_dir):
    # Load the main dataset
    print(f"Loading main dataset from {all_in_one_dataset_json}")
    dataset_dt = pd.read_json(all_in_one_dataset_json).set_index("object_id")

    # Identify records with mismatched registry names and no LLM judge decision,
    # excluding cases where names are set to defaults ("Not specified" / "NONE")
    mask = (dataset_dt.simple_matching == False) & (dataset_dt.llm_response.isnull())
    mask2 = (dataset_dt.a_registry_name == "Not specified") & (dataset_dt.b_registry_name == "NONE")
    diff_records = dataset_dt[mask & (~mask2)][["a_registry_name", "b_registry_name"]]
    dataset_dt.loc[:, "annotation_category"] = "Not specified"
    print("Preview of records with differing registry names:")
    print(diff_records.head())

    # # Prepare DataFrame for incorrect registry names with an empty correction column
    incorrect_dt = dataset_dt[mask & (~mask2)][["title", "abstract", "a_registry_name", "b_registry_name"]].assign(correct_registry_name="").assign(comment="")
    print("Incorrect registry names shape:", incorrect_dt.shape)
    
    # create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # save excel file
    output_excel = f"{output_dir}/incorrect_registry_name.xlsx"
    incorrect_dt.to_excel(output_excel)
    print(f"Saved incorrect registry names to {output_excel}")

    output_json = f"{output_dir}/incorrect_registry_name.json"
    incorrect_dt.reset_index().to_json(output_json, orient="records", indent=4)
    print(f"Saved incorrect registry names to {output_json}")


    # save incorrect registry names as json
    incorrect_registry_names_folder = f"{output_dir}/incorrect_registry_names"
    os.makedirs(incorrect_registry_names_folder, exist_ok=True)

    for index, row in incorrect_dt.iterrows():
        with open(f"{incorrect_registry_names_folder}/{index}.txt", "w") as f:
            json.dump(row.to_dict(), f, indent=4)

    # zip this folder
    shutil.make_archive(incorrect_registry_names_folder, 'zip', incorrect_registry_names_folder)


if __name__ == '__main__':
    main()
