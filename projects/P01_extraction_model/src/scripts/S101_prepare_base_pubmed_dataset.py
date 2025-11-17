import os
import click
import pandas as pd

@click.command()
@click.option('--raw_annotated_ds_excel', type=str, required=True, help='Path to input Excel file')
@click.option('--base_pubmed_dataset_jsonl', type=str, required=True, help='Path to output JSONL file')
def preprocess(raw_annotated_ds_excel, base_pubmed_dataset_jsonl):
    """Preprocess the dataset and save the extracted data in JSONL format."""
    # makedir
    os.makedirs(os.path.dirname(base_pubmed_dataset_jsonl), exist_ok=True)

    
    # Load the data from Excel
    df = pd.read_excel(raw_annotated_ds_excel, sheet_name='evaluation_dataset', engine='openpyxl')

    # Column renaming mapping
    mapping = {
        'Population': 'population',
        'Registry name': 'registry_name',
        'Outcome/Endpoint': 'outcome_endpoint',
        'Study type justification': 'study_type_justification',
        'Study type': 'study_type',
        'Follow-up': 'follow_up',
        'Size': 'size',
        'Age group': 'age_group',
        'Condition': 'condition',
        'Is registry': 'is_registry',
        'Intervention': 'intervention',
        'Sex': 'sex',
        'Country': 'country',
        'Comparator': 'comparator'
    }

    # Rename columns
    df = df.rename(columns=mapping)

    # Extract the required columns
    df_base = df[["pmid", "title", "abstract"]]

    # Save to JSONL with UTF-8 encoding
    df_base.to_json(base_pubmed_dataset_jsonl, orient='records', lines=True, force_ascii=False)

    click.echo(f"Processed data saved to {base_pubmed_dataset_jsonl}")

if __name__ == '__main__':
    preprocess()
