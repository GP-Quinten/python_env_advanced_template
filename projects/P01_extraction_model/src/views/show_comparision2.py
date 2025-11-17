import streamlit as st
import json
import pandas as pd

# wide
st.set_page_config(layout="wide")

dataset_jsonl = "data/W00_R00_sample_publication_dataset/prod_publication_dataset.jsonl"

dataset_dt = pd.read_json(dataset_jsonl, lines=True).set_index("object_id")


model_a_results_jsonl = "data/W01_R01_parse_llm_annotation_large/llm_parsed.jsonl"
model_b_results_jsonl = "data/W01_R11_parse_llm_annotation_small/llm_parsed.jsonl"

model_a_results_dt = pd.read_json(model_a_results_jsonl, lines=True).set_index("object_id")
model_a_results_dt.loc[:, "a_registry_name"] = model_a_results_dt.llm_annotation.apply(lambda x: x.get("Registry name", "NONE"))
model_b_results_dt = pd.read_json(model_b_results_jsonl, lines=True).set_index("object_id")
model_b_results_dt.loc[:, "b_registry_name"] = model_b_results_dt.llm_annotation.apply(lambda x: x.get("Registry name", "NONE"))

dataset_dt.loc[:, "a_registry_name"] = model_a_results_dt.a_registry_name.str.strip()
dataset_dt.loc[:, "b_registry_name"] = model_b_results_dt.b_registry_name.str.strip()

st.write("## Dataset")
# size
st.write(dataset_dt.shape)

dataset_dt[[
    "title",
    "abstract",
     "a_registry_name",
    "b_registry_name",
]]

dataset_dt.loc[:, "simple_matching"] = model_a_results_dt.a_registry_name == model_b_results_dt.b_registry_name

st.write("Simple matching")
st.write(dataset_dt.simple_matching.value_counts(normalize=True))
matched_dt = dataset_dt[dataset_dt.simple_matching]
st.write(matched_dt.shape)
dataset_dt[[
    "a_registry_name",
    "b_registry_name",
]]

st.write("## LLM as a Judge")

llm_as_a_judge_jsonl = "data/W01_R21_parse_llm_annotation_comparison_large_vs_small_registry_name/llm_comparison_parsed_large_vs_small.jsonl"
llm_as_a_judge_dt = pd.read_json(llm_as_a_judge_jsonl, lines=True).set_index("object_id")
llm_as_a_judge_dt.loc[:, "llm_response"] = llm_as_a_judge_dt.llm_annotation.apply(lambda x: x.get("final_decision"))
llm_as_a_judge_dt.loc[:, "llm_justification"] = llm_as_a_judge_dt.llm_annotation.apply(lambda x: x.get("explanation"))
llm_as_a_judge_dt

dataset_dt.loc[:, "llm_response"] = llm_as_a_judge_dt.llm_response
dataset_dt.loc[:, "llm_justification"] = llm_as_a_judge_dt.llm_justification

llm_dataset_dt = dataset_dt[dataset_dt.llm_response.notnull()]

llm_dataset_dt[[
    "llm_response",
    "a_registry_name",
    "b_registry_name",
    "llm_justification",
]]

mask = (dataset_dt.simple_matching == False) & (dataset_dt.llm_response.isnull())
mask2 = (dataset_dt.a_registry_name == "Not specified") & (dataset_dt.b_registry_name == "NONE")

dataset_dt[mask & (~mask2)][[
    "a_registry_name",
    "b_registry_name",
]]

wtf = dataset_dt[mask & (~mask2)][[
    "title",
    "abstract",
    "a_registry_name",
    "b_registry_name",
]].assign(correct_registry_name = "")


wtf.to_excel("data/W99_annotation/incorrect_registry_name.xlsx")

# make a folder in data/W99_annotation/ called incorrect_registry_names and save each row as formated json
import os
import json

incorrect_registry_names_folder = "data/W99_annotation/incorrect_registry_names"
os.makedirs(incorrect_registry_names_folder, exist_ok=True)

for index, row in wtf.iterrows():
    with open(f"{incorrect_registry_names_folder}/{index}.txt", "w") as f:
        json.dump(row.to_dict(), f, indent=4)

# zip this folder
import shutil
shutil.make_archive(incorrect_registry_names_folder, 'zip', incorrect_registry_names_folder)

# # Path to the comparison JSONL file
# comparison_file_path = "data/W01_R22_compare_large_vs_small_with_transformers/llm_comparison_large_vs_small.jsonl"
# registry_comparison_file_path = "data/W01_R30_generate_report_large_vs_small/registry_name_comparison.json"

# # Function to read JSONL file
# def read_jsonl(file_path):
#     with open(file_path, 'r') as file:
#         return [json.loads(line) for line in file]

# # Function to read JSON file
# def read_json(file_path):
#     with open(file_path, 'r') as file:
#         return json.load(file)

# # Read the comparison data
# comparison_data = read_jsonl(comparison_file_path)

# # Read the registry comparison data
# registry_comparison_data = read_json(registry_comparison_file_path)

# # Convert the comparison data to a DataFrame
# comparison_df = pd.DataFrame(comparison_data)

# # Convert the registry comparison data to a DataFrame
# registry_comparison_df = pd.DataFrame(registry_comparison_data)

# # Merge the dataframes on 'object_id'
# merged_df = pd.merge(comparison_df, registry_comparison_df, on='object_id', how='inner')

# # Display the comparison data
# st.title("Comparison of Large vs Small Models with Transformers")
# st.write("This is the final comparison of the large and small models using transformers.")

# # Display the merged data as a table
# cols = [
#     "model_a_registry_name",
#     "model_b_registry_name",
#     "Registry name",
#     "bi_encoder_score",
#     "cross_encoder_score"
# ]

# merged_df = merged_df[cols]
# merged_df = merged_df.rename(columns={
#     "Registry name": "llm_response"
# })

# st.write(merged_df)