rule W01_R00_extraction_with_mistral_large:
    input:
        script="src/scripts/S201_annotate_with_llm.py",
        base_pubmed_dataset_jsonl="data/W00_R00_sample_publication_dataset/prod_publication_dataset.jsonl",
        prompt_txt="etc/prompts/prompt_publications_v1.1.txt",
        model_config="etc/configs/large_mistral_config.json"
    output:
        jsonl="data/W01_R00_extraction_with_mistral_large/llm_inference.jsonl"
    shell:
        """
        python {input.script} \
          --base_pubmed_dataset_jsonl {input.base_pubmed_dataset_jsonl} \
          --prompt_txt {input.prompt_txt} \
          --model_config {input.model_config} \
          --output_jsonl {output.jsonl}
        """

rule W01_R01_parse_llm_annotation_large:
    input:
        script="src/scripts/S301_parse_llm_annotation.py",
        llm_annotated_jsonl="data/W01_R00_extraction_with_mistral_large/llm_inference.jsonl"
    output:
        jsonl="data/W01_R01_parse_llm_annotation_large/llm_parsed.jsonl"
    shell:
        """
        python {input.script} \
          --llm_annotated_jsonl {input.llm_annotated_jsonl} \
          --output_jsonl {output.jsonl}
        """

rule W01_R10_extraction_with_mistral_small:
    input:
        script="src/scripts/S201_annotate_with_llm.py",
        base_pubmed_dataset_jsonl="data/W00_R00_sample_publication_dataset/prod_publication_dataset.jsonl",
        prompt_txt="etc/prompts/prompt_publications_v1.1.txt",
        model_config="etc/configs/small_mistral_config.json"
    output:
        jsonl="data/W01_R10_extraction_with_mistral_small/llm_inference.jsonl"
    shell:
        """
        python {input.script} \
          --base_pubmed_dataset_jsonl {input.base_pubmed_dataset_jsonl} \
          --prompt_txt {input.prompt_txt} \
          --model_config {input.model_config} \
          --output_jsonl {output.jsonl}
        """

rule W01_R11_parse_llm_annotation_small:
    input:
        script="src/scripts/S301_parse_llm_annotation.py",
        llm_annotated_jsonl="data/W01_R10_extraction_with_mistral_small/llm_inference.jsonl"
    output:
        jsonl="data/W01_R11_parse_llm_annotation_small/llm_parsed.jsonl"
    shell:
        """
        python {input.script} \
          --llm_annotated_jsonl {input.llm_annotated_jsonl} \
          --output_jsonl {output.jsonl}
        """

rule W01_R20_compare_large_vs_small:
    input:
        script="src/scripts/S401_compare_results.py",
        model_a_dataset_jsonl="data/W01_R01_parse_llm_annotation_large/llm_parsed.jsonl",
        model_b_dataset_jsonl="data/W01_R11_parse_llm_annotation_small/llm_parsed.jsonl",
        prompt_txt="etc/prompts/compare_two_results_v1.1.txt",
        model_config="etc/configs/large_mistral_config.json"
    output:
        jsonl="data/W01_R20_compare_large_vs_small/llm_comparison_large_vs_small.jsonl"
    shell:
        """
        python {input.script} \
          --model_a_dataset_jsonl {input.model_a_dataset_jsonl} \
          --model_b_dataset_jsonl {input.model_b_dataset_jsonl} \
          --prompt_txt {input.prompt_txt} \
          --model_config {input.model_config} \
          --output_jsonl {output.jsonl}
        """

rule W01_R20_compare_large_vs_small_registry_name:
    input:
        script="src/scripts/S401_compare_results_openai.py",
        model_a_dataset_jsonl="data/W01_R01_parse_llm_annotation_large/llm_parsed.jsonl",
        model_b_dataset_jsonl="data/W01_R11_parse_llm_annotation_small/llm_parsed.jsonl",
        prompt_txt="etc/prompts/compare_registry_name_v1.txt",
        model_config="etc/configs/gpt4o_openai_config.json"
    output:
        jsonl="data/W01_R20_compare_large_vs_small_registry_name/llm_comparison_large_vs_small.jsonl"
    shell:
        """
        python {input.script} \
          --model_a_dataset_jsonl {input.model_a_dataset_jsonl} \
          --model_b_dataset_jsonl {input.model_b_dataset_jsonl} \
          --prompt_txt {input.prompt_txt} \
          --model_config {input.model_config} \
          --output_jsonl {output.jsonl}
        """


rule W01_R21_parse_llm_annotation_comparison_large_vs_small:
    input:
        script="src/scripts/S301_parse_llm_annotation.py",
        llm_annotated_jsonl="data/W01_R20_compare_large_vs_small/llm_comparison_large_vs_small.jsonl"
    output:
        jsonl="data/W01_R21_parse_llm_annotation_comparison_large_vs_small/llm_comparison_parsed_large_vs_small.jsonl"
    shell:
        """
        python {input.script} \
          --llm_annotated_jsonl {input.llm_annotated_jsonl} \
          --output_jsonl {output.jsonl}
        """

rule W01_R21_parse_llm_annotation_comparison_large_vs_small_registry_name:
    input:
        script="src/scripts/S301_parse_llm_annotation.py",
        llm_annotated_jsonl="data/W01_R20_compare_large_vs_small_registry_name/llm_comparison_large_vs_small.jsonl"
    output:
        jsonl="data/W01_R21_parse_llm_annotation_comparison_large_vs_small_registry_name/llm_comparison_parsed_large_vs_small.jsonl"
    shell:
        """
        python {input.script} \
          --llm_annotated_jsonl {input.llm_annotated_jsonl} \
          --output_jsonl {output.jsonl}
        """

rule W01_R30_generate_report_large_vs_small:
    input:
        script="src/scripts/S501_generate_report.py",
        content_a_jsonl="data/W01_R01_parse_llm_annotation_large/llm_parsed.jsonl",
        content_b_jsonl="data/W01_R11_parse_llm_annotation_small/llm_parsed.jsonl",
        comparison_jsonl="data/W01_R21_parse_llm_annotation_comparison_large_vs_small/llm_comparison_parsed_large_vs_small.jsonl"
    output:
        report="data/W01_R30_generate_report_large_vs_small/report.txt",
        plot="data/W01_R30_generate_report_large_vs_small/plot.png"
    shell:
        """
        python {input.script} \
          --content_a_jsonl {input.content_a_jsonl} \
          --content_b_jsonl {input.content_b_jsonl} \
          --comparison_jsonl {input.comparison_jsonl} \
          --output_report {output.report} \
          --output_plot {output.plot}
        """

rule W01_R22_compare_large_vs_small_with_transformers:
    input:
        script="src/scripts/S202_compare_with_transformers.py",
        model_a_dataset_jsonl="data/W01_R01_parse_llm_annotation_large/llm_parsed.jsonl",
        model_b_dataset_jsonl="data/W01_R11_parse_llm_annotation_small/llm_parsed.jsonl"
    output:
        jsonl="data/W01_R22_compare_large_vs_small_with_transformers/llm_comparison_large_vs_small.jsonl"
    shell:
        """
        python {input.script} \
          --model_a_dataset_jsonl {input.model_a_dataset_jsonl} \
          --model_b_dataset_jsonl {input.model_b_dataset_jsonl} \
          --output_jsonl {output.jsonl}
        """