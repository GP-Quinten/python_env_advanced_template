import click
import json
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util

def load_jsonl(file_path):
    """Load a JSONL file and return a dictionary mapping object_id to record."""
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                obj_id = record.get("object_id")
                if obj_id:
                    data[obj_id] = record
    return data

def clean_registry_name(reg_name):
    reg_name = reg_name.replace("Registry", "").replace("registry", "").strip()
    return reg_name

@click.command()
@click.option('--model_a_dataset_jsonl', type=str, required=True, help="Path to JSONL file for model A dataset.")
@click.option('--model_b_dataset_jsonl', type=str, required=True, help="Path to JSONL file for model B dataset.")
@click.option('--output_jsonl', type=str, required=True, help="Path to output JSONL file for comparison results.")
def compare_registry_names(model_a_dataset_jsonl, model_b_dataset_jsonl, output_jsonl):
    """
    Compare the 'Registry name' field from two JSONL datasets using both a bi-encoder and a cross encoder.

    The script performs batch inference to compute:
      - A cosine similarity score (bi_encoder_score) using 'all-mpnet-base-v2'.
      - A cross encoder score (cross_encoder_score) using 'cross-encoder/stsb-roberta-large'.

    Both models utilize CUDA if available.

    Each output record contains:
      - object_id
      - model_a_registry_name
      - model_b_registry_name
      - bi_encoder_score
      - cross_encoder_score
    """
    # Set device to cuda if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    click.echo(f"Using device: {device}")

    # Load datasets
    data_a = load_jsonl(model_a_dataset_jsonl)
    data_b = load_jsonl(model_b_dataset_jsonl)
    common_ids = sorted(set(data_a.keys()).intersection(data_b.keys()))
    click.echo(f"Found {len(common_ids)} common object_ids.")

    # Prepare records for batch processing
    common_records = []
    for obj_id in common_ids:
        record_a = data_a[obj_id]
        record_b = data_b[obj_id]
        reg_name_a = record_a.get("llm_annotation", {}).get("Registry name", "")
        reg_name_b = record_b.get("llm_annotation", {}).get("Registry name", "")
        if reg_name_a and reg_name_b:
            reg_name_a = clean_registry_name(reg_name_a)
            reg_name_b = clean_registry_name(reg_name_b)
            common_records.append((obj_id, reg_name_a, reg_name_b))
        else:
            click.echo(f"Skipping object_id {obj_id} due to missing registry name.")

    if not common_records:
        click.echo("No common records with valid registry names found. Exiting.")
        return

    # Unzip records into separate lists
    object_ids, model_a_names, model_b_names = zip(*common_records)

    # ---------------------------
    # Bi-encoder batch inference using all-mpnet-base-v2
    # ---------------------------
    bi_model = SentenceTransformer('all-mpnet-base-v2', device=device)
    embeddings_a = bi_model.encode(model_a_names, convert_to_tensor=True, batch_size=32)
    embeddings_b = bi_model.encode(model_b_names, convert_to_tensor=True, batch_size=32)
    cos_sim_matrix = util.cos_sim(embeddings_a, embeddings_b)
    bi_scores = cos_sim_matrix.diag().tolist()

    # ---------------------------
    # Cross encoder batch inference using cross-encoder/stsb-roberta-large
    # ---------------------------
    cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-large', device=device)
    pairs = [[a, b] for a, b in zip(model_a_names, model_b_names)]
    cross_scores = cross_encoder.predict(pairs, batch_size=32)

    # Combine results, converting scores to native Python floats
    results = []
    for obj_id, a_name, b_name, bi_score, cross_score in zip(object_ids, model_a_names, model_b_names, bi_scores, cross_scores):
        results.append({
            "object_id": obj_id,
            "model_a_registry_name": a_name,
            "model_b_registry_name": b_name,
            "bi_encoder_score": float(bi_score),
            "cross_encoder_score": float(cross_score)
        })

    results = sorted(results, key=lambda x: x["cross_encoder_score"], reverse=True)

    # Write results to the output JSONL file
    with open(output_jsonl, 'w', encoding='utf-8') as out_f:
        for result in results:
            out_f.write(json.dumps(result) + "\n")
    click.echo(f"Comparison results saved to {output_jsonl}")

if __name__ == '__main__':
    compare_registry_names()
