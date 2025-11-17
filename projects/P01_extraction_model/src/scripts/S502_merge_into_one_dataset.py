import os
import click
import json
import pandas as pd

@click.command()
@click.option('--dataset_jsonl', type=click.Path(exists=True), required=True,
              help='Path to the main dataset JSONL file.')
@click.option('--model_a_jsonl', type=click.Path(exists=True), required=True,
              help='Path to model A LLM annotation JSONL file.')
@click.option('--model_b_jsonl', type=click.Path(exists=True), required=True,
              help='Path to model B LLM annotation JSONL file.')
@click.option('--llm_judge_jsonl', type=click.Path(exists=True), required=True,
              help='Path to LLM judge comparison JSONL file.')
@click.option('--output_dir', type=click.Path(), required=True,
              help='Path to output Excel file with incorrect registry names.')
def main(dataset_jsonl, model_a_jsonl, model_b_jsonl, llm_judge_jsonl, output_dir):
    """
    Processes dataset and LLM results to identify records with mismatched registry names
    and exports them to an Excel file.
    """
    # Load the main dataset
    print(f"Loading main dataset from {dataset_jsonl}")
    dataset_dt = pd.read_json(dataset_jsonl, lines=True).set_index("object_id")

    # Load model A results
    print(f"Loading Model A results from {model_a_jsonl}")
    model_a_results_dt = pd.read_json(model_a_jsonl, lines=True).set_index("object_id")
    model_a_results_dt.loc[:, "a_registry_name"] = model_a_results_dt.llm_annotation.apply(
        lambda x: x.get("Registry name", "NONE")
    )

    # Load model B results
    print(f"Loading Model B results from {model_b_jsonl}")
    model_b_results_dt = pd.read_json(model_b_jsonl, lines=True).set_index("object_id")
    model_b_results_dt.loc[:, "b_registry_name"] = model_b_results_dt.llm_annotation.apply(
        lambda x: x.get("Registry name", "NONE")
    )

    # Merge registry names into the main dataset
    dataset_dt.loc[:, "a_registry_name"] = model_a_results_dt.a_registry_name.str.strip()
    dataset_dt.loc[:, "b_registry_name"] = model_b_results_dt.b_registry_name.str.strip()

    # Display dataset info
    print("Dataset shape:", dataset_dt.shape)
    print("Dataset preview:")
    print(dataset_dt[["title", "abstract", "a_registry_name", "b_registry_name"]].head())

    # Compute a simple matching indicator for registry names
    dataset_dt.loc[:, "simple_matching"] = model_a_results_dt.a_registry_name == model_b_results_dt.b_registry_name
    print("Simple matching counts (normalized):")
    print(dataset_dt.simple_matching.value_counts(normalize=True))
    
    # Show a preview of matched records
    matched_dt = dataset_dt[dataset_dt.simple_matching]
    print("Matched dataset shape:", matched_dt.shape)
    print("Registry names preview:")
    print(dataset_dt[["a_registry_name", "b_registry_name"]].head())

    # Load LLM as a Judge results
    print(f"Loading LLM as a Judge results from {llm_judge_jsonl}")
    llm_as_a_judge_dt = pd.read_json(llm_judge_jsonl, lines=True).set_index("object_id")
    llm_as_a_judge_dt.loc[:, "llm_response"] = llm_as_a_judge_dt.llm_annotation.apply(
        lambda x: x.get("final_decision")
    )
    llm_as_a_judge_dt.loc[:, "llm_justification"] = llm_as_a_judge_dt.llm_annotation.apply(
        lambda x: x.get("explanation")
    )
    print("LLM as a Judge preview:")
    print(llm_as_a_judge_dt.head())

    # Merge judge results into the main dataset
    dataset_dt.loc[:, "llm_response"] = llm_as_a_judge_dt.llm_response
    dataset_dt.loc[:, "llm_justification"] = llm_as_a_judge_dt.llm_justification

    # create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # save dataset_dt as jsonl
    # Index(['outcome_measure', 'comparator', 'registry_related',
    #    'geographical_area', 'intervention', 'publishing_date', 'abstract',
    #    'data_source_name', 'design_model_description', 'population_follow_up',
    #    'redirection_link', 'population_sex', 'journal', 'design_model',
    #    'summary', 'keywords', 'chemicals', 'pmid', 'population_age_group',
    #    'authors', 'retrieved_from', 'category', 'pdf_link',
    #    'population_description', 'medical_condition', 'population_size',
    #    'mesh_terms', 'title', 'a_registry_name', 'b_registry_name',
    #    'simple_matching', 'llm_response', 'llm_justification'],
    #   dtype='object')
    cols_to_save = [
        "object_id",
        "pmid",
        "title",
        "abstract",
        "a_registry_name",
        "b_registry_name",
        'simple_matching',
        'llm_response',
        'llm_justification',
    ]
    all_in_one_dataset_json = f"{output_dir}/all_in_one_dataset.json"
    # indent 4
    dataset_dt.reset_index()[cols_to_save].to_json(all_in_one_dataset_json, orient="records", indent=4)
    print(f"Saved all in one dataset to {all_in_one_dataset_json}")

    # result categories
    # 


    # Identify records with mismatched registry names and no LLM judge decision,
    # excluding cases where names are set to defaults ("Not specified" / "NONE")
    # mask = (dataset_dt.simple_matching == False) & (dataset_dt.llm_response.isnull())
    # mask2 = (dataset_dt.a_registry_name == "Not specified") & (dataset_dt.b_registry_name == "NONE")
    # diff_records = dataset_dt[mask & (~mask2)][["a_registry_name", "b_registry_name"]]
    # dataset_dt.loc[:, "annotation_category"] = "Not specified"
    # print("Preview of records with differing registry names:")
    # print(diff_records.head())

    # # Prepare DataFrame for incorrect registry names with an empty correction column
    # incorrect_dt = dataset_dt[mask & (~mask2)][["title", "abstract", "a_registry_name", "b_registry_name"]].assign(correct_registry_name="")
    # print(f"Saving incorrect registry names to Excel: {output_excel}")
    # incorrect_dt.to_excel(output_excel)
    # print("Excel file saved.")

if __name__ == '__main__':
    main()
