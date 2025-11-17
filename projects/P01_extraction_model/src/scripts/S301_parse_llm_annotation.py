import os
import json
import click
from dotenv import load_dotenv
from more_europa import helpers

# Load environment variables
load_dotenv()

@click.command()
@click.option('--llm_annotated_jsonl', type=str, required=True, help="Path to input JSONL file with base PubMed dataset annotated by LLM")
@click.option('--output_jsonl', type=str, required=True, help="Path to output JSONL file with LLM annotations")
def annotate_with_llm(llm_annotated_jsonl, output_jsonl):
    # Load base PubMed dataset
    records = []
    with open(llm_annotated_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    results = []
    for record in records:
        llm_response = record.get("llm_response")
        if "response" in llm_response:
            llm_annotation = llm_response.get("response").get("body").get("choices")[0].get("message").get("content")
        else:
            llm_annotation = llm_response.get("choices")[0].get("message").get("content")
        try:
            llm_annotation = llm_annotation.replace("```json\n", "").replace("```", "")
            llm_annotation = llm_annotation.strip()
            if llm_annotation[-1] == ",":
                llm_annotation = llm_annotation[:-1]
            
            if "}" not in llm_annotation:
                llm_annotation += "}"

            llm_annotation = json.loads(llm_annotation)
        except json.JSONDecodeError as e:
            print(llm_annotation)
            print(f"Error decoding JSON for record {record.get('object_id')}")
            raise e
        results.append({
            "object_id": record.get("object_id"),
            "llm_annotation": llm_annotation
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
