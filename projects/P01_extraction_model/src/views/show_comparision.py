import streamlit as st
import json
import pandas as pd

# wide
st.set_page_config(layout="wide")

# Path to the comparison JSONL file
comparison_file_path = "data/W01_R22_compare_large_vs_small_with_transformers/llm_comparison_large_vs_small.jsonl"
registry_comparison_file_path = "data/W01_R30_generate_report_large_vs_small/registry_name_comparison.json"

# Function to read JSONL file
def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

# Function to read JSON file
def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Read the comparison data
comparison_data = read_jsonl(comparison_file_path)

# Read the registry comparison data
registry_comparison_data = read_json(registry_comparison_file_path)

# Convert the comparison data to a DataFrame
comparison_df = pd.DataFrame(comparison_data)

# Convert the registry comparison data to a DataFrame
registry_comparison_df = pd.DataFrame(registry_comparison_data)

# Merge the dataframes on 'object_id'
merged_df = pd.merge(comparison_df, registry_comparison_df, on='object_id', how='inner')

# Display the comparison data
st.title("Comparison of Large vs Small Models with Transformers")
st.write("This is the final comparison of the large and small models using transformers.")

# Display the merged data as a table
cols = [
    "model_a_registry_name",
    "model_b_registry_name",
    "Registry name",
    "bi_encoder_score",
    "cross_encoder_score"
]

merged_df = merged_df[cols]
merged_df = merged_df.rename(columns={
    "Registry name": "llm_response"
})

st.write(merged_df)