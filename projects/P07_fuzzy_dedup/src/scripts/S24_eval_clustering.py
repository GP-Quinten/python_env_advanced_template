#!/usr/bin/env python3
"""
Evaluate clustering output against labeled test pairs.
Generates:
 - predictions_with_labels.csv (left/right ids, names, label, y_pred)
 - report.md (metrics + confusion matrix)
 - false_negatives_positives_report.md (markdown FN/FP)
"""
from __future__ import annotations
import json
from pathlib import Path
import click
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

def _write_report(metrics: dict, out_path: Path, df: pd.DataFrame):
    tn, fp, fn, tp = metrics['confusion'].ravel()
    lines = [
        "Clustering Evaluation Report\n",
        f"Samples: {metrics['n']}\n",
        f"Number of clusters: {metrics['n_clusters']}\n",
        f"Total unique objects (left_object_id + right_object_id union): {metrics['n_objects']}\n",
        f"Number of valid objects (found in clusters): {metrics['n_valid_objects']}\n",
        f"Accuracy: {metrics['accuracy']:.4f}\n",
        f"Precision: {metrics['precision']:.4f}\n",
        f"Recall: {metrics['recall']:.4f}\n",
        f"F1: {metrics['f1']:.4f}\n",
        "\nConfusion Matrix:\n",
        "           Pred 0   Pred 1\n",
        f"True 0   {tn:7d}  {fp:7d}\n",
        f"True 1   {fn:7d}  {tp:7d}\n",
        "\nFull Classification Report:\n",
        metrics['classification_report'],
        "\n"
    ]
    fp_df = df[(df['label'] == 0) & (df['y_pred'] == 1)]
    fn_df = df[(df['label'] == 1) & (df['y_pred'] == 0)]
    fp_sample = fp_df.sample(n=min(10, len(fp_df)), random_state=42) if not fp_df.empty else pd.DataFrame()
    fn_sample = fn_df.sample(n=min(10, len(fn_df)), random_state=42) if not fn_df.empty else pd.DataFrame()
    cols = [c for c in ['pair_hash_id', 'left_name', 'right_name', 'label', 'y_pred'] if c in df.columns]
    def nested_list(rows):
        out = []
        for _, row in rows.iterrows():
            if 'pair_hash_id' in cols:
                out.append(f"- pair_hash_id: {row['pair_hash_id']}")
                for field in [c for c in cols if c != 'pair_hash_id']:
                    out.append(f"    - {field}: {row[field]}")
            else:
                out.append("- " + ", ".join(f"{col}: {row[col]}" for col in cols))
        return out
    lines.append("\n10 Random False Positives:\n")
    if not fp_sample.empty:
        lines.extend(nested_list(fp_sample[cols]))
    else:
        lines.append("No False Positives found.")
    lines.append("\n10 Random False Negatives:\n")
    if not fn_sample.empty:
        lines.extend(nested_list(fn_sample[cols]))
    else:
        lines.append("No False Negatives found.")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_fn_fp(df: pd.DataFrame, out_path: Path):
    # Ensure required columns exist and extract acronyms
    if 'left_name' in df.columns:
        df['left_acronym'] = df['left_name'].str.extract(r'\(([^)]+)\)', expand=False).fillna('')
    if 'right_name' in df.columns:
        df['right_acronym'] = df['right_name'].str.extract(r'\(([^)]+)\)', expand=False).fillna('')
    
    df_fn = df[(df['label'] == 1) & (df['y_pred'] == 0)]
    df_fp = df[(df['label'] == 0) & (df['y_pred'] == 1)]
    
    # Add left_object_id and right_object_id to the columns
    cols = [
        'pair_hash_id', 'left_object_id', 'right_object_id',
        'left_name', 'left_acronym', 'right_name', 'right_acronym', 'label', 'y_pred'
    ]
    cols = [c for c in cols if c in df.columns]
    def nested_list(rows):
        out = []
        for _, row in rows.iterrows():
            if 'pair_hash_id' in cols:
                out.append(f"- pair_hash_id: {row['pair_hash_id']}")
                for field in [c for c in cols if c != 'pair_hash_id']:
                    out.append(f"    - {field}: {row[field]}")
            else:
                out.append("- " + ", ".join(f"{col}: {row[col]}" for col in cols))
        return out
    md = [
        "False Negatives and False Positives Report\n",
        "False Negatives:\n"
    ]
    if not df_fn.empty:
        md.extend(nested_list(df_fn[cols]))
    else:
        md.append("No False Negatives found.")
    md.append("\nFalse Positives:\n")
    if not df_fp.empty:
        md.extend(nested_list(df_fp[cols]))
    else:
        md.append("No False Positives found.")
    out_path.write_text("\n".join(md), encoding="utf-8")

@click.command()
@click.option('--clusters_json', type=click.Path(exists=True, path_type=Path), required=True)
@click.option('--test_data', type=click.Path(exists=True, path_type=Path), required=True)
@click.option('--test_indices', type=click.Path(exists=True, path_type=Path), required=True)
@click.option('--output_dir', type=click.Path(path_type=Path), required=True)
@click.option('--cluster_prefix', default='C')
@click.option('--pred_column', default='proba')
@click.option('--seed', default=42)
def main(clusters_json: Path, test_data: Path, test_indices: Path, output_dir: Path, cluster_prefix: str, pred_column: str, seed: int):
    out = output_dir
    out.mkdir(parents=True, exist_ok=True)

    # load clusters
    clusters = pd.read_json(clusters_json)
    mapping = clusters.set_index('object_id')['cluster_id'].to_dict()
    n_clusters = clusters['cluster_id'].nunique()

    # load test pairs
    df = pd.read_json(test_data)
    # indices available but not used
    _ = np.load(test_indices)

    # Calculate n_objects (union of left/right object ids)
    left_ids = set(df['left_object_id'].unique())
    right_ids = set(df['right_object_id'].unique())
    all_ids = left_ids | right_ids
    n_objects = len(all_ids)

    # Calculate n_valid_objects (unique object_ids found in clusters)
    cluster_object_ids = set(clusters['object_id'].unique())
    valid_object_ids = all_ids & cluster_object_ids
    n_valid_objects = len(valid_object_ids)

    # Get cluster IDs, handling null values
    left_cid = df['left_object_id'].map(mapping)
    right_cid = df['right_object_id'].map(mapping)
    
    # Add columns safely
    df['left_cid'] = left_cid
    df['right_cid'] = right_cid
    
    # predict same cluster - only when both objects have valid cluster IDs and they match
    df['y_pred'] = ((df['left_cid'].notna()) & 
                    (df['right_cid'].notna()) & 
                    (df['left_cid'] == df['right_cid'])).astype(int)

    # save predictions
    cols = [c for c in ['left_object_id','right_object_id','left_name','right_name','label','y_pred'] if c in df.columns]
    df[cols].to_csv(out / 'predictions_with_labels.csv', index=False)

    # compute metrics
    y_true = df['label'].astype(int).to_numpy()
    y_pred = df['y_pred'].to_numpy()
    cm = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, zero_division=0)
    metrics = {
        'n': int(len(df)),
        'n_clusters': int(n_clusters),
        'n_objects': int(n_objects),
        'n_valid_objects': int(n_valid_objects),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'confusion': cm,
        'classification_report': class_report
    }
    # write report and fn/fp
    _write_report(metrics, out / 'report.txt', df)
    _write_fn_fp(df, out / 'false_negatives_positives_report.txt')

    click.echo(f"[SUCCESS] Wrote evaluation to {out}")

if __name__ == '__main__':
    main()
