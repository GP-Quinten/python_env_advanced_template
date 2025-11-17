rule W00_R00_sample_publication_dataset:
    input:
        script="src/scripts/S102_sample_dataset_from_prod.py"
    output:
        jsonl="data/W00_R00_sample_publication_dataset/prod_publication_dataset.jsonl"
    params:
        n_samples=300,
        collection_name="Publication_v2"
    shell:
        """
        python {input.script} \
          --n_samples {params.n_samples} \
          --collection_name {params.collection_name} \
          --output_jsonl {output.jsonl}
        """
