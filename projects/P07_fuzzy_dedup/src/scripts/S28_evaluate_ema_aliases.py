#!/usr/bin/env python3
"""
Evaluate the quality of aliases of EMA registries that have been assigned to clusters.
Uses an LLM to judge whether registry pairs are true aliases of each other.

Based on the approach from P05_refine_dedup but adapted for P07_fuzzy_dedup data.

Input:
  - EMA clusters.json (registry objects + cluster assignments)
  - Object clusters.json (registry objects + cluster assignments)

Output:
  - Excel file with detailed assessment of all EMA aliases
  - Report of the quality assessment with statistics
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import logging
from dotenv import load_dotenv
import click
import llm_backends
from llm_inference.cache.tmp import TmpCacheStorage
import asyncio
from collections import defaultdict


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--ema_clusters", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Path to ema_clusters.json file with EMA registry cluster assignments.")
@click.option("--clusters_json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Path to object_clusters.json with registry cluster assignments.")
@click.option("--output_dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Directory to write outputs.")
@click.option("--prompt_file", type=str, default="etc/prompts/prompt_compare_registry_names_v2.txt",
              help="Path to prompt template for LLM evaluation.")
@click.option("--model_config", type=str, default="etc/configs/gpt4_1_openai_config.json",
              help="Path to LLM model configuration.")
@click.option("--max_sample_size", type=int, default=-1, show_default=True,
              help="Maximum number of pairs to evaluate (-1 for all).")
@click.option("--seed", type=int, default=42, show_default=True,
              help="Random seed for reproducibility.")
def main(
    ema_clusters: Path,
    clusters_json: Path,
    output_dir: Path,
    prompt_file: str,
    model_config: str,
    max_sample_size: int,
    seed: int,
):
    """Evaluate the quality of EMA registry aliases."""
    # Clean logging
    logging.getLogger('httpx').setLevel(logging.WARNING)
    random.seed(seed)
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    click.echo(f"Loading EMA clusters: {ema_clusters}")
    with open(ema_clusters, 'r') as f:
        ema_clusters_list = json.load(f)
    ema_clusters_df = pd.DataFrame(ema_clusters_list)
    click.echo(f'Loaded ema_clusters: {ema_clusters_df.shape[0]} entries')
    
    click.echo(f"Loading registry clusters data: {clusters_json}")
    with open(clusters_json, 'r') as f:
        clusters_data = json.load(f)
    clusters_df = pd.DataFrame(clusters_data)
    click.echo(f'Loaded registry clusters data: {clusters_df.shape[0]} entries')
    
    # 2. Process Data for Analysis
    # Filter to EMA registries that were assigned to a cluster
    ema_assigned_df = ema_clusters_df[ema_clusters_df['cluster_id'].notna()].copy()
    
    # Rename for clarity
    ema_assigned_df = ema_assigned_df.rename(columns={
        'object_id': 'ema_id',
        'registry_name': 'ema_registry_name',
        'acronym': 'ema_acronym'
    })
    
    click.echo(f'EMA registries assigned to a cluster: {len(ema_assigned_df)} / {len(ema_clusters_df)}')
    
    # Create a mapping from cluster_id to member objects
    cluster_to_members = {}
    for _, row in clusters_df.iterrows():
        if pd.notna(row['cluster_id']):
            if row['cluster_id'] not in cluster_to_members:
                cluster_to_members[row['cluster_id']] = []
            cluster_to_members[row['cluster_id']].append({
                'object_id': row['object_id'],
                'registry_name': row.get('registry_name'),
                'acronym': row.get('acronym')
            })
    
    # Print cluster size distribution
    cluster_sizes = [len(members) for members in cluster_to_members.values()]
    click.echo(f'Total clusters: {len(cluster_to_members)}')
    click.echo(f'Avg. cluster size: {np.mean(cluster_sizes):.2f}')
    click.echo(f'Min cluster size: {min(cluster_sizes)}')
    click.echo(f'Max cluster size: {max(cluster_sizes)}')
    click.echo(f'Clusters with size ≥ 20: {sum(1 for s in cluster_sizes if s >= 20)}')
    
    # 3. Prepare Alias Pairs for LLM Assessment
    click.echo("Preparing alias pairs for LLM assessment...")
    alias_pairs = []
    
    for _, ema in tqdm(ema_assigned_df.iterrows(), desc='Preparing alias pairs'):
        cluster_id = ema['cluster_id']
        if cluster_id not in cluster_to_members:
            continue
            
        cluster_members = cluster_to_members[cluster_id]
        # Sample up to 20 members if cluster is large
        nmax=20
        if len(cluster_members) >= nmax:
            selected_members = random.sample(cluster_members, nmax)
        else:
            selected_members = cluster_members
        
        for member in selected_members:
            # Skip self-comparisons (if EMA already in the registry dataset)
            if str(member['object_id']) == str(ema['ema_id']):
                continue
                
            alias_pairs.append({
                'ema_id': ema['ema_id'],
                'ema_registry_name': ema['ema_registry_name'],
                'ema_acronym': ema['ema_acronym'],
                'ema_cluster_id': cluster_id,
                'alias_object_id': member['object_id'],
                'alias_registry_name': member['registry_name'],
                'alias_acronym': member.get('acronym'),
                'cluster_size': len(cluster_members)
            })
    
    alias_pairs_df = pd.DataFrame(alias_pairs)
    click.echo(f'Generated {len(alias_pairs_df)} alias pairs for evaluation')
    
    # 4. LLM Assessment: Prepare Prompts
    click.echo("Preparing prompts for LLM assessment...")
    load_dotenv()
    
    with open(prompt_file, 'r') as pf:
        base_prompt = pf.read().strip()
    
    # Load model config
    with open(model_config, 'r', encoding='utf-8') as f:
        model_cfg = json.load(f)
    
    def construct_prompt(base_prompt: str, name1: str, name2: str) -> str:
        return base_prompt.replace('{{content_a}}', name1).replace('{{content_b}}', name2)
    
    # Set sample size for evaluation
    sample_size = len(alias_pairs_df) if max_sample_size == -1 else min(max_sample_size, len(alias_pairs_df))
    click.echo(f'Sampling {sample_size} pairs for LLM evaluation (max_sample_size={max_sample_size})')
    
    prompts = []
    for idx, row in alias_pairs_df.head(sample_size).iterrows():
        prompt = construct_prompt(base_prompt, row['ema_registry_name'], row['alias_registry_name'])
        prompts.append({
            'prompt': prompt,
            'custom_id': f"{row['ema_id']}|||{row['alias_object_id']}",
            'ema_registry_name': row['ema_registry_name'],
            'alias': row['alias_registry_name'],
            'ema_id': row['ema_id'],
            'ema_cluster_id': row['ema_cluster_id'],
            'cluster_size': row['cluster_size']
        })
    click.echo(f'Prepared {len(prompts)} prompts for LLM assessment.')
    
    # 5. LLM Backend Setup
    cache_storage = TmpCacheStorage()
    if 'openai' in model_config:
        # print if an api key is found
        click.echo(f"Using OpenAI API Key: {'FOUND' if os.getenv('OPENAI_API_KEY') else 'NOT FOUND'}")
        backend = llm_backends.OpenAIAsyncBackend(api_key=os.getenv('OPENAI_API_KEY'), cache_storage=cache_storage)
    elif 'istral' in model_config:
        click.echo(f"Using Mistral API Key: {'FOUND' if os.getenv('MISTRAL_API_KEY') else 'NOT FOUND'}")
        backend = llm_backends.MistralAsyncBackend(api_key=os.getenv('MISTRAL_API_KEY'), cache_storage=cache_storage)
    
    async def run_async_inference(prompts, backend, model_cfg):
        raw_responses = []
        pbar = tqdm(total=len(prompts), desc='LLM Inference')
        for prompt in prompts:
            raw_response = await backend.infer_one(prompt, model_cfg)
            raw_response['custom_id'] = prompt['custom_id']
            raw_responses.append(raw_response)
            pbar.update(1)
        pbar.close()
        return raw_responses
    
    click.echo('LLM backend setup complete.')
    
    # 7. Run LLM Inference
    click.echo("Running LLM inference...")
    loop = asyncio.get_event_loop()
    raw_responses = loop.run_until_complete(run_async_inference(prompts, backend, model_cfg))
    click.echo(f'LLM inference complete. {len(raw_responses)} responses collected.')
    
    # 8. Parse and Process Results
    click.echo("Processing LLM responses...")
    prompt_map = {p['custom_id']: p for p in prompts}
    llm_responses = []
    
    for raw_response in tqdm(raw_responses, desc='Processing responses'):
        custom_id = raw_response.get('custom_id', '')
        prompt_obj = prompt_map.get(custom_id)
        if prompt_obj:
            parsed_response = backend._parse_response(raw_response)
            parsed_response['custom_id'] = custom_id
            parsed_response['ema_id'] = prompt_obj['ema_id']
            parsed_response['ema_registry_name'] = prompt_obj['ema_registry_name']
            parsed_response['alias'] = prompt_obj['alias']
            parsed_response['ema_cluster_id'] = prompt_obj['ema_cluster_id']
            parsed_response['cluster_size'] = prompt_obj['cluster_size']
            llm_responses.append(parsed_response)
    
    llm_responses_df = pd.DataFrame(llm_responses)
    click.echo(f'Parsed {llm_responses_df.shape[0]} responses.')
    
    # Create get_final_label function that was missing in the code
    def get_final_label(decision):
        """Convert LLM decision to binary label: 1 for same/very_close, 0 for different"""
        if decision in ['same', 'very_close']:
            return 1
        else:
            return 0

    # Add after creating llm_responses_df
    # Create final_label column for statistics
    llm_responses_df['final_label'] = llm_responses_df['final_decision'].map(get_final_label)
    
    # Add cluster size bins for analysis BEFORE we try to use it
    llm_responses_df['cluster_size_bin'] = pd.cut(
        llm_responses_df['cluster_size'],
        bins=[0, 2, 5, 10, 20, 1000],
        labels=['2', '3-5', '6-10', '11-20', '>20']
    )
    
    # 9. Calculate Statistics
    click.echo("Calculating statistics...")
    # there 3 options; 'same', 'different', 'very_close', print how many of each
    click.echo("Final decision counts:")
    click.echo(llm_responses_df['final_decision'].value_counts())
    
    # Calculate aliases quality score per cluster
    click.echo("Calculating aliases quality scores...")
    cluster_quality_scores = defaultdict(lambda: {'total': 0, 'good': 0})
    
    for _, row in llm_responses_df.iterrows():
        cluster_id = row['ema_cluster_id']
        cluster_size = row['cluster_size']
        # For clusters ≥ 20, we need to adjust the score since we only sampled 20
        scaling_factor = cluster_size / 20 if cluster_size >= 20 else 1
        
        cluster_quality_scores[cluster_id]['total'] += scaling_factor
        if row['final_label'] == 1:  # same or very_close
            cluster_quality_scores[cluster_id]['good'] += scaling_factor
    
    # Calculate quality score for each cluster
    cluster_quality_df = pd.DataFrame([
        {
            'cluster_id': cid,
            'quality_score': scores['good'] / scores['total'] if scores['total'] > 0 else 0,
            'cluster_size': llm_responses_df[llm_responses_df['ema_cluster_id'] == cid]['cluster_size'].iloc[0]
        }
        for cid, scores in cluster_quality_scores.items()
    ])
    
    # Calculate mean aliases quality score weighted by cluster size
    total_weighted_score = sum(row['quality_score'] * row['cluster_size'] 
                             for _, row in cluster_quality_df.iterrows())
    total_size = sum(row['cluster_size'] for _, row in cluster_quality_df.iterrows())
    mean_aliases_quality_score = total_weighted_score / total_size if total_size > 0 else 0
    
    # Calculate quality scores by cluster size bins
    bin_quality_scores = defaultdict(lambda: {'total': 0, 'good': 0})
    
    # Now this line won't cause a KeyError because the column exists
    for _, row in llm_responses_df.iterrows():
        bin_name = row['cluster_size_bin']
        cluster_size = row['cluster_size']
        scaling_factor = cluster_size / 20 if cluster_size >= 20 else 1
        
        bin_quality_scores[bin_name]['total'] += scaling_factor
        if row['final_label'] == 1:
            bin_quality_scores[bin_name]['good'] += scaling_factor
    
    bin_quality_df = pd.DataFrame([
        {
            'bin': bin_name,
            'quality_score': scores['good'] / scores['total'] if scores['total'] > 0 else 0,
            'count': int(scores['total'])
        }
        for bin_name, scores in bin_quality_scores.items()
    ]).sort_index()

    # 10. Analysis of Alias Quality
    total = len(llm_responses_df)
    same_count = (llm_responses_df.final_label == 1).sum()
    diff_count = (llm_responses_df.final_label == 0).sum()
    
    # Calculate stats by cluster size
    cluster_size_stats = llm_responses_df.groupby('cluster_size_bin')['final_label'].agg(['mean', 'count'])
    cluster_size_stats = cluster_size_stats.rename(columns={'mean': 'good_aliases_%'})
    
    # Calculate per-EMA registry statistics
    ema_stats = llm_responses_df.groupby('ema_id').agg({
        'final_label': ['mean', 'count', 'sum'],
        'ema_registry_name': 'first',
        'ema_cluster_id': 'first',
        'cluster_size': 'first'
    })
    
    ema_stats.columns = ['good_aliases_%', 'tested_pairs', 'correct_pairs', 'ema_registry_name', 'cluster_id', 'cluster_size']
    ema_stats = ema_stats.sort_values('cluster_size', ascending=False)
    
    # 11. Save Results
    # Save to Excel
    excel_path = output_dir / "ema_aliases_assessment.xlsx"
    click.echo(f"Saving detailed assessment to: {excel_path}")
    
    with pd.ExcelWriter(excel_path) as writer:
        # Assessments data
        llm_responses_df[[
            'custom_id', 'ema_id', 'ema_cluster_id', 'cluster_size', 'cluster_size_bin',
            'ema_registry_name', 'alias', 'final_decision', 'explanation'
        ]].to_excel(writer, sheet_name='All_Assessments', index=False)
        
        # Cluster size statistics with aliases quality scores
        bin_quality_df.to_excel(writer, sheet_name='Cluster_Size_Stats', index=False)
        
        # Add cluster quality scores
        cluster_quality_df.to_excel(writer, sheet_name='Cluster_Quality_Scores', index=False)
        
        # EMA registry statistics
        ema_stats.to_excel(writer, sheet_name='EMA_Stats')
    
    # Generate report
    report_path = output_dir / "report.md"
    click.echo(f"Generating report: {report_path}")
    
    report_content = f"""# EMA Registry Aliases Quality Assessment

## Summary Statistics

- **Total EMA Registries**: {len(ema_clusters_df)}
- **EMA Registries with Cluster Assignment**: {len(ema_assigned_df)} ({len(ema_assigned_df)/len(ema_clusters_df):.1%})
- **Total Evaluated Pairs**: {total}
- **Same**: {same_count} ({same_count/total:.1%})
- **Different**: {diff_count} ({diff_count/total:.1%})
- **Mean Aliases Quality Score**: {mean_aliases_quality_score:.1%}

## Aliases Quality Score by Cluster Size

| Cluster Size | Count | Aliases Quality Score |
|--------------|-------|---------------------|
"""
    
    for _, row in bin_quality_df.iterrows():
        report_content += f"| {row['bin']} | {int(row['count'])} | {row['quality_score']:.1%} |\n"
    
    # Add more sections to the report
    report_content += f"""
## EMA Registry Analysis

- **Total EMA Registries Evaluated**: {len(ema_stats)}
- **EMA Registries with >90% Good aliases**: {(ema_stats['good_aliases_%'] > 0.9).sum()}
- **EMA Registries with <50% Good aliases**: {(ema_stats['good_aliases_%'] < 0.5).sum()}

## Conclusion

This analysis assessed the quality of EMA registry aliases by evaluating whether pairs of registries within the same cluster are true aliases of each other. The overall mean aliases quality score is {mean_aliases_quality_score:.1%}, indicating the weighted average quality across all clusters. The LLM-based assessment provides insights into the accuracy of the fuzzy deduplication process.
"""

    # Write the report
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    click.echo(f"Evaluation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()