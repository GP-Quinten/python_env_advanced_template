import os
import json

import click


@click.command()
@click.option('--input_jsonl', '-i', type=str, required=True, help='Path to the input JSONL file.')
@click.option('--output_dir', '-o', type=str, required=True, help='Directory to store the cleaned dataset.')
def clean_dataset(input_jsonl, output_dir):
    """
    Clean the dataset in JSONL format.
    
    This script reads the input JSONL file, applies a dummy cleaning operation by adding a 'cleaned' flag,
    and writes the cleaned records to 'dataset_clean.jsonl' in the specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "dataset_clean.jsonl")
    
    with open(input_jsonl, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            record = json.loads(line)
            # Dummy cleaning: mark record as cleaned
            record["cleaned"] = True
            outfile.write(json.dumps(record) + "\n")
    
    click.echo(f"Cleaned dataset written to {output_file}")

if __name__ == '__main__':
    clean_dataset()
