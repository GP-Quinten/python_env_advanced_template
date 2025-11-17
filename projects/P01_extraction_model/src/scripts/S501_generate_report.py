#!/usr/bin/env python
import os
import json
import click
import pandas as pd
import matplotlib.pyplot as plt

@click.command()
@click.option('--content_a_jsonl', type=str, required=True, help="Path to JSONL file for content A.")
@click.option('--content_b_jsonl', type=str, required=True, help="Path to JSONL file for content B.")
@click.option('--comparison_jsonl', type=str, required=True, help="Path to JSONL file with comparison results.")
@click.option('--output_report', type=str, required=True, help="Path to output text report.")
@click.option('--output_plot', type=str, required=True, help="Path to output plot image (PNG).")
def generate_report(content_a_jsonl, content_b_jsonl, comparison_jsonl, output_report, output_plot):
    # Load Content A dataset
    content_a_df = pd.read_json(content_a_jsonl, lines=True)
    if "llm_annotation" in content_a_df.columns and "object_id" in content_a_df.columns:
        content_a_df = pd.DataFrame(content_a_df["llm_annotation"].to_list(), index=content_a_df["object_id"])
    else:
        click.echo("Warning: Content A JSON structure is not as expected.")

    # Load Content B dataset
    content_b_df = pd.read_json(content_b_jsonl, lines=True)
    if "llm_annotation" in content_b_df.columns and "object_id" in content_b_df.columns:
        content_b_df = pd.DataFrame(content_b_df["llm_annotation"].to_list(), index=content_b_df["object_id"])
    else:
        click.echo("Warning: Content B JSON structure is not as expected.")

    # Load Comparison results
    comparison_df = pd.read_json(comparison_jsonl, lines=True)
    if "llm_annotation" in comparison_df.columns and "object_id" in comparison_df.columns:
        comparison_df = pd.DataFrame(comparison_df["llm_annotation"].to_list(), index=comparison_df["object_id"])
    else:
        click.echo("Warning: Comparison JSON structure is not as expected.")

    # Begin constructing report
    report_lines = []
    report_lines.append("Report Summary")
    report_lines.append("=======================")
    report_lines.append(f"Number of records in Content A: {len(content_a_df)}")
    report_lines.append(f"Number of records in Content B: {len(content_b_df)}")
    report_lines.append(f"Number of records in Comparison: {len(comparison_df)}")
    report_lines.append("")

    # Distribution of 'Registry related'
    if "Registry related" in comparison_df.columns:
        registry_counts = comparison_df["Registry related"].value_counts()
        report_lines.append("Distribution of 'Registry related':")
        for key, value in registry_counts.items():
            report_lines.append(f"  {key}: {value}")
    else:
        report_lines.append("Column 'Registry related' not found in comparison results.")
    report_lines.append("")

    # Report value counts for key columns
    key_columns = ["Registry name", "Geographical area", "Outcome measure", "Intervention"]
    for col in key_columns:
        if col in comparison_df.columns:
            report_lines.append(f"Value counts for '{col}':")
            counts = comparison_df[col].value_counts(dropna=False)
            for key, value in counts.items():
                report_lines.append(f"  {key}: {value}")
            report_lines.append("")
        else:
            report_lines.append(f"Column '{col}' not found in comparison results.")
            report_lines.append("")

    # Save report file
    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    click.echo(f"Report saved to {output_report}")

    # Create a bar plot for the "Registry related" distribution if available
    if "Registry related" in comparison_df.columns:
        plt.figure(figsize=(8, 6))
        registry_counts.plot(kind="bar")
        plt.title("Registry related Distribution")
        plt.xlabel("Category")
        plt.ylabel("Count")
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_plot), exist_ok=True)
        plt.savefig(output_plot)
        plt.close()
        click.echo(f"Plot saved to {output_plot}")
    else:
        click.echo("Column 'Registry related' not found. Plot not generated.")

    

    # For each key column, build a combined DataFrame (as in your code snippet) and save it as a TSV file.
    # We only consider records where content A indicates "A Registry related" == "yes".
    output_dir = os.path.dirname(output_report)

    content_a_df.columns = [f"A {col}" for col in content_a_df.columns]
    content_b_df.columns = [f"B {col}" for col in content_b_df.columns]


    # save content_a_df:
    ids = content_b_df.index

    for col in key_columns:
        # Concatenate the relevant columns from content A, content B, and the comparison results.
        try:
            combined_df = pd.concat([
                content_a_df.loc[:, [f"A {col}"]],
                content_b_df.loc[:, [f"B {col}"]],
                comparison_df.loc[:, [col]],
                content_a_df.loc[:, ["A Registry related"]],
                content_b_df.loc[:, ["B Registry related"]],
            ], axis=1).sort_values(by=col)
        except KeyError as e:
            click.echo(f"Column error for {col}: {e}")
            continue

        # Print value counts for the current column (optional logging)
        click.echo(f"Value counts for {col}:")
        click.echo(combined_df[col].value_counts().to_string())

        # Construct a safe file name (replace spaces with underscores)
        json_filename = os.path.join(output_dir, f"{col.lower().replace(' ', '_')}_comparison.json")
        combined_df.reset_index().to_json(json_filename, orient="records", indent=2)
        click.echo(f"JSON file saved to {json_filename}")

if __name__ == "__main__":
    generate_report()
