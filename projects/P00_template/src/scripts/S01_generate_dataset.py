import os
import json

import click


@click.command()
@click.option('--n_samples', '-n', type=int, required=True, help='Number of samples to generate.')
@click.option('--output_dir', '-o', type=str, required=True, help='Path to the output directory where the JSONL file will be saved.')
def generate_dataset(n_samples, output_dir):
    """
    Generate a dummy dataset in JSONL format.
    
    The generated JSONL file will be saved as "dataset.jsonl" in the specified output directory.
    Each line in the output file is a JSON object with an 'id' and a 'value'.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    output_jsonl = os.path.join(output_dir, "dataset.jsonl")
    with open(output_jsonl, 'w') as f:
        for i in range(1, n_samples + 1):
            record = {"id": i, "value": f"value_{i}"}
            f.write(json.dumps(record) + "\n")
    
    click.echo(f"Dataset with {n_samples} samples generated at {output_jsonl}")


if __name__ == '__main__':
    generate_dataset()
