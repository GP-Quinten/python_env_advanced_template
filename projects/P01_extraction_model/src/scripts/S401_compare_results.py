import os
import json
import click
from dotenv import load_dotenv
from more_europa import helpers

# Load environment variables
load_dotenv()

@click.command()
@click.option('--model_a_dataset_jsonl', type=str, required=True, help="Path to input JSONL file with base PubMed dataset")
@click.option('--model_b_dataset_jsonl', type=str, required=True, help="Path to input JSONL file with base PubMed dataset")
@click.option('--prompt_txt', type=str, required=True, help="Path to the annotation prompt text file")
@click.option('--model_config', type=str, required=True, help="Path to the model configuration JSON file")
@click.option('--output_jsonl', type=str, required=True, help="Path to output JSONL file with LLM annotations")
def annotate_with_llm(model_a_dataset_jsonl, model_b_dataset_jsonl, prompt_txt, model_config, output_jsonl):
    """Annotate the base PubMed dataset using an LLM model."""
    
    # Load model configuration
    with open(model_config, "r", encoding="utf-8") as f:
        model_config_data = json.load(f)
    
    print(f"Loaded model config: {model_config_data}")

    # Load the annotation prompt
    with open(prompt_txt, "r", encoding="utf-8") as f:
        annotation_prompt = f.read().strip()

    print(f"Loaded annotation prompt: {annotation_prompt}")

    # Load base PubMed dataset
    model_a_records = []
    with open(model_a_dataset_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            model_a_records.append(json.loads(line))

    print(f"Loaded {len(model_a_records)} records from {model_a_dataset_jsonl}")

    model_b_records = []
    with open(model_b_dataset_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            model_b_records.append(json.loads(line))

    print(f"Loaded {len(model_b_records)} records from {model_b_dataset_jsonl}")

    # Prepare prompts for LLM inference
    prompts = []
    for record_a, record_b in zip(model_a_records, model_b_records):
        record_a_json = json.dumps(record_a.get("llm_annotation"))
        record_b_json = json.dumps(record_b.get("llm_annotation"))
        
        full_prompt = annotation_prompt.replace("{{content_a}}", record_a_json)
        full_prompt = full_prompt.replace("{{content_b}}", record_b_json)

        prompts.append(full_prompt)

    # # Perform batch inference using the LLM
    llm_responses = helpers.mistral.inference_by_one(prompts=prompts, model_config=model_config_data)

    results = []
    # # Attach annotations to records
    for record, annotation in zip(model_a_records, llm_responses):
        results.append({
            "object_id": record.get("object_id"),
            "llm_response": annotation
        })

    # makedir
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    # Save annotated dataset to JSONL
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Saved annotated dataset to {output_jsonl}")

if __name__ == "__main__":
    annotate_with_llm()
