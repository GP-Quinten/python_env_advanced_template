#!/usr/bin/env python3
"""
Comprehensive Model Evaluation for Registry Deduplication

This script evaluates the performance of deduplication predictions against ground truth labels.
It generates comprehensive metrics, visualizations, and detailed analysis reports including
ROC curves, precision-recall curves, calibration plots, and false positive/negative analysis.

Input Requirements:
    - dataset.parquet: Original dataset with ground truth labels
    - predictions.csv: Model predictions with scores from S05_make_inference
    
Output:
    - Comprehensive metrics and performance reports
    - Multiple evaluation plots and visualizations
    - Detailed analysis of model errors (false positives/negatives)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve, roc_curve,
    confusion_matrix, classification_report, accuracy_score, precision_score,
    recall_score, f1_score, brier_score_loss, log_loss,
)
from sklearn.calibration import calibration_curve

from deduplication_model.config import COLUMN_CONFIG, COL_PAIR_ID, COL_LABEL, LEFT_PREFIX, RIGHT_PREFIX


def align_predictions_with_ground_truth(
    predictions_dataframe: pd.DataFrame, 
    ground_truth_dataframe: pd.DataFrame, 
    left_column_name: str, 
    right_column_name: str
) -> pd.DataFrame:
    """
    Align prediction data with ground truth labels using available join keys.
    
    This function attempts multiple alignment strategies in order of preference:
    1. Join on pair_id (most reliable)
    2. Join on left/right name columns
    3. Align by row order (fallback)
    
    Args:
        predictions_dataframe: DataFrame containing model predictions
        ground_truth_dataframe: DataFrame containing true labels
        left_column_name: Name of the left registry column
        right_column_name: Name of the right registry column
        
    Returns:
        DataFrame with predictions aligned to ground truth labels
        
    Raises:
        click.ClickException: If alignment cannot be performed
    """
    predictions_copy = predictions_dataframe.copy()
    ground_truth_copy = ground_truth_dataframe.copy()

    # Strategy 1: Join on pair_id (preferred method)
    if COL_PAIR_ID in predictions_copy.columns and COL_PAIR_ID in ground_truth_copy.columns:
        click.echo(f"[INFO] Aligning data using pair_id column: {COL_PAIR_ID}")
        return predictions_copy.merge(
            ground_truth_copy[[COL_PAIR_ID]], on=COL_PAIR_ID, how="left"
        ).merge(
            ground_truth_copy, on=COL_PAIR_ID, how="left", suffixes=("", "_ground_truth")
        )

    # Strategy 2: Join on left/right name columns
    left_right_columns = {left_column_name, right_column_name}
    if (left_right_columns.issubset(predictions_copy.columns) and 
        left_right_columns.issubset(ground_truth_copy.columns)):
        click.echo(f"[INFO] Aligning data using name columns: {left_column_name}, {right_column_name}")
        return predictions_copy.merge(
            ground_truth_copy[[left_column_name, right_column_name]],
            on=[left_column_name, right_column_name], how="left"
        ).merge(
            ground_truth_copy,
            on=[left_column_name, right_column_name], how="left", 
            suffixes=("", "_ground_truth")
        )

    # Strategy 3: Align by row order (fallback)
    if len(predictions_copy) == len(ground_truth_copy):
        click.echo("[INFO] Aligning data by row order (fallback method)")
        predictions_copy = predictions_copy.reset_index(drop=True)
        ground_truth_copy = ground_truth_copy.reset_index(drop=True)
        
        for column in ground_truth_copy.columns:
            predictions_copy[column] = ground_truth_copy[column]
        return predictions_copy

    # If no alignment strategy works, raise an error
    raise click.ClickException(
        "Unable to align predictions with ground truth data. "
        "Ensure datasets have matching pair_id, name columns, or same row order/length."
    )


def determine_optimal_classification_threshold(
    true_labels: np.ndarray, 
    prediction_probabilities: np.ndarray,
    optimization_strategy: str = "none", 
    fixed_threshold: Optional[float] = None
) -> float:
    """
    Determine the optimal classification threshold based on the specified strategy.
    
    Args:
        true_labels: Ground truth binary labels
        prediction_probabilities: Model prediction probabilities
        optimization_strategy: Strategy for threshold selection ("f1", "youden", or "none")
        fixed_threshold: Fixed threshold value (overrides optimization_strategy)
        
    Returns:
        Optimal threshold value
    """
    if fixed_threshold is not None:
        return float(fixed_threshold)
    
    optimization_strategy = (optimization_strategy or "none").lower()
    
    if optimization_strategy == "f1":
        # Optimize for F1 score
        precision_values, recall_values, threshold_values = precision_recall_curve(true_labels, prediction_probabilities)
        precision_values, recall_values = precision_values[:-1], recall_values[:-1]
        
        if len(precision_values) == 0:
            return 0.5
            
        f1_scores = (2 * precision_values * recall_values) / np.maximum(precision_values + recall_values, 1e-12)
        optimal_index = np.argmax(f1_scores)
        return float(threshold_values[optimal_index]) if len(threshold_values) > 0 else 0.5
    
    elif optimization_strategy == "youden":
        # Optimize using Youden's J statistic (sensitivity + specificity - 1)
        false_positive_rates, true_positive_rates, threshold_values = roc_curve(true_labels, prediction_probabilities)
        youden_statistics = true_positive_rates - false_positive_rates
        optimal_index = np.argmax(youden_statistics)
        return float(threshold_values[optimal_index]) if len(threshold_values) > 0 else 0.5
    
    # Default threshold
    return 0.5


def create_individual_evaluation_plots(
    recall_values: np.ndarray, 
    precision_values: np.ndarray, 
    average_precision: float,
    false_positive_rates: np.ndarray, 
    true_positive_rates: np.ndarray, 
    roc_auc: float,
    true_labels: np.ndarray, 
    prediction_probabilities: np.ndarray, 
    output_directory: Path
) -> None:
    """
    Create and save individual evaluation plots (PR curve, ROC curve, calibration, histogram).
    
    Args:
        recall_values: Recall values from precision-recall curve
        precision_values: Precision values from precision-recall curve
        average_precision: Average precision score
        false_positive_rates: FPR values from ROC curve
        true_positive_rates: TPR values from ROC curve
        roc_auc: ROC AUC score
        true_labels: Ground truth labels
        prediction_probabilities: Model prediction probabilities
        output_directory: Directory to save plots
    """
    # Precision-Recall curve with high-precision region highlighted
    plt.figure(figsize=(8, 6))
    plt.step(recall_values, precision_values, where="post", linewidth=2)
    
    # Highlight precision ≥ 0.95 region
    high_precision_mask = precision_values[:-1] >= 0.95
    plt.fill_between(
        recall_values[:-1], precision_values[:-1], 0.95, 
        where=high_precision_mask, step="post", alpha=0.3, 
        label="Precision ≥ 0.95", color="lightgreen"
    )
    
    plt.axhline(0.95, linestyle="--", linewidth=1, color="gray", alpha=0.7)
    plt.xlabel("Recall")
    plt.ylabel("Precision") 
    plt.title(f"Precision-Recall Curve (AP = {average_precision:.4f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_directory / "pr_curve.png", dpi=150)
    plt.close()

    # ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(false_positive_rates, true_positive_rates, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7, label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_directory / "roc_curve.png", dpi=150)
    plt.close()

    # Calibration plot
    plt.figure(figsize=(8, 6))
    fraction_of_positives, mean_predicted_value = calibration_curve(
        true_labels, prediction_probabilities, n_bins=10, strategy="uniform"
    )
    plt.plot(mean_predicted_value, fraction_of_positives, marker="o", linewidth=2, markersize=8, label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7, label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_directory / "calibration.png", dpi=150)
    plt.close()

    # Score distribution histogram
    plt.figure(figsize=(8, 6))
    plt.hist(prediction_probabilities[true_labels == 0], bins=30, alpha=0.6, label="Negative Class", color="red")
    plt.hist(prediction_probabilities[true_labels == 1], bins=30, alpha=0.6, label="Positive Class", color="blue")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Score Distribution by True Class")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_directory / "proba_hist.png", dpi=150)
    plt.close()


def create_combined_evaluation_figure(
    recall_values: np.ndarray, 
    precision_values: np.ndarray, 
    average_precision: float,
    false_positive_rates: np.ndarray, 
    true_positive_rates: np.ndarray, 
    roc_auc: float,
    true_labels: np.ndarray, 
    prediction_probabilities: np.ndarray, 
    output_path: Path
) -> None:
    """
    Create a combined figure with all evaluation plots in a 2x2 grid.
    
    Args:
        recall_values: Recall values from precision-recall curve
        precision_values: Precision values from precision-recall curve  
        average_precision: Average precision score
        false_positive_rates: FPR values from ROC curve
        true_positive_rates: TPR values from ROC curve
        roc_auc: ROC AUC score
        true_labels: Ground truth labels
        prediction_probabilities: Model prediction probabilities
        output_path: Path to save the combined figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Precision-Recall curve (top-left)
    ax = axes[0, 0]
    ax.step(recall_values, precision_values, where="post", linewidth=2, label="PR Curve")
    high_precision_mask = precision_values[:-1] >= 0.95
    ax.fill_between(
        recall_values[:-1], precision_values[:-1], 0.95,
        where=high_precision_mask, step="post", alpha=0.3, 
        label="Precision ≥ 0.95", color="lightgreen"
    )
    ax.axhline(0.95, linestyle="--", linewidth=1, color="gray", alpha=0.7)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve (AP = {average_precision:.4f})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # ROC curve (top-right)
    ax = axes[0, 1]
    ax.plot(false_positive_rates, true_positive_rates, linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.grid(True, alpha=0.3)

    # Calibration plot (bottom-left)
    ax = axes[1, 0]
    fraction_of_positives, mean_predicted_value = calibration_curve(
        true_labels, prediction_probabilities, n_bins=10, strategy="uniform"
    )
    ax.plot(mean_predicted_value, fraction_of_positives, marker="o", linewidth=2, markersize=8)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Plot")
    ax.grid(True, alpha=0.3)

    # Score distribution (bottom-right)
    ax = axes[1, 1]
    ax.hist(prediction_probabilities[true_labels == 0], bins=30, alpha=0.6, label="Negative", color="red")
    ax.hist(prediction_probabilities[true_labels == 1], bins=30, alpha=0.6, label="Positive", color="blue")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution by True Class")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_precision_recall_thresholds_analysis(
    true_labels: np.ndarray, 
    prediction_probabilities: np.ndarray,
    output_directory: Path
) -> dict:
    """
    Generate analysis of thresholds needed to achieve specific precision levels.
    
    Args:
        true_labels: Ground truth labels
        prediction_probabilities: Model prediction probabilities
        output_directory: Directory to save results
        
    Returns:
        Dictionary mapping precision targets to thresholds and recall values
    """
    precision_values, recall_values, threshold_values = precision_recall_curve(true_labels, prediction_probabilities)
    
    # Align arrays (precision_recall_curve returns different lengths)
    precision_array = precision_values[1:]  # Skip the last precision value
    recall_array = recall_values[1:]       # Skip the last recall value
    
    precision_threshold_mapping = {}
    
    # Analyze precision targets from 0.85 to 0.96
    for target_precision in np.arange(0.85, 0.97, 0.01):
        target_precision = round(target_precision, 2)
        
        # Find indices where precision meets or exceeds target
        qualifying_indices = np.where(precision_array >= target_precision)[0]
        
        if len(qualifying_indices) > 0:
            # Choose the threshold that maximizes recall for the given precision
            optimal_index = qualifying_indices[0]
            precision_threshold_mapping[target_precision] = {
                "threshold": float(threshold_values[optimal_index]),
                "recall": float(recall_array[optimal_index])
            }
        else:
            precision_threshold_mapping[target_precision] = {
                "threshold": None,
                "recall": None
            }
    
    # Save the analysis
    threshold_analysis_path = output_directory / "precision_recall_thresholds.json"
    with open(threshold_analysis_path, "w") as analysis_file:
        json.dump(precision_threshold_mapping, analysis_file, indent=2)
    
    return precision_threshold_mapping


def create_detailed_error_analysis_report(
    predictions_with_labels: pd.DataFrame, 
    output_directory: Path
) -> None:
    """
    Create detailed analysis of false positives and false negatives with registry names.
    
    Args:
        predictions_with_labels: DataFrame with predictions, labels, and registry names
        output_directory: Directory to save the report
    """
    analysis_dataframe = predictions_with_labels.copy()
    
    # Extract acronyms from registry names for better analysis
    if "left_name" in analysis_dataframe.columns:
        analysis_dataframe["left_acronym_extracted"] = (
            analysis_dataframe["left_name"].str.extract(r"\(([^)]+)\)", expand=False).fillna("")
        )
    if "right_name" in analysis_dataframe.columns:
        analysis_dataframe["right_acronym_extracted"] = (
            analysis_dataframe["right_name"].str.extract(r"\(([^)]+)\)", expand=False).fillna("")
        )

    # Classify prediction outcomes
    analysis_dataframe["prediction_outcome"] = np.where(
        (analysis_dataframe["label"] == 1) & (analysis_dataframe["y_pred"] == 0), "False Negative",
        np.where((analysis_dataframe["label"] == 0) & (analysis_dataframe["y_pred"] == 1), "False Positive", "Correct")
    )

    # Separate false negatives and false positives
    false_negatives = analysis_dataframe[analysis_dataframe["prediction_outcome"] == "False Negative"].sort_values(by="proba", ascending=True)
    false_positives = analysis_dataframe[analysis_dataframe["prediction_outcome"] == "False Positive"].sort_values(by="proba", ascending=False)

    # Create markdown report
    report_lines = [
        "# Model Error Analysis Report\n",
        f"## Summary\n",
        f"- Total Predictions: {len(analysis_dataframe):,}\n",
        f"- False Negatives: {len(false_negatives):,}\n", 
        f"- False Positives: {len(false_positives):,}\n",
        f"- Correct Predictions: {len(analysis_dataframe) - len(false_negatives) - len(false_positives):,}\n\n",
        "## False Negatives (True Matches Missed)\n",
        "_Sorted by prediction score (lowest confidence first)_\n\n"
    ]
    
    # Add false negatives table
    if not false_negatives.empty:
        display_columns = ["left_name", "left_acronym_extracted", "right_name", "right_acronym_extracted", "label", "proba", "y_pred"]
        available_columns = [col for col in display_columns if col in false_negatives.columns]
        report_lines.append(false_negatives[available_columns].to_markdown(index=False, tablefmt="pipe"))
    else:
        report_lines.append("No false negatives found.")
    
    report_lines.extend(["\n\n## False Positives (Non-Matches Incorrectly Flagged)\n", 
                        "_Sorted by prediction score (highest confidence first)_\n\n"])
    
    # Add false positives table
    if not false_positives.empty:
        display_columns = ["left_name", "left_acronym_extracted", "right_name", "right_acronym_extracted", "label", "proba", "y_pred"]
        available_columns = [col for col in display_columns if col in false_positives.columns]
        report_lines.append(false_positives[available_columns].to_markdown(index=False, tablefmt="pipe"))
    else:
        report_lines.append("No false positives found.")
    
    # Save the report
    error_analysis_path = output_directory / "detailed_error_analysis.md"
    error_analysis_path.write_text("\n".join(report_lines), encoding="utf-8")


def generate_comprehensive_markdown_report(
    output_directory: Path, 
    evaluation_metrics: dict
) -> None:
    """
    Generate a comprehensive markdown report with all evaluation results.
    
    Args:
        output_directory: Directory to save the report
        evaluation_metrics: Dictionary containing all computed metrics
    """
    confusion_matrix_data = evaluation_metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    true_negatives, false_positives = confusion_matrix_data[0]
    false_negatives, true_positives = confusion_matrix_data[1]
    
    report_sections = [
        "# Model Evaluation Report\n",
        "## Performance Summary\n",
        f"- **Dataset Size**: {evaluation_metrics['n_samples']:,} registry pairs\n",
        f"- **Class Prevalence**: {evaluation_metrics['prevalence']:.4f} (positive rate)\n",
        f"- **ROC AUC**: {evaluation_metrics['auroc']:.4f}\n",
        f"- **Average Precision**: {evaluation_metrics['average_precision']:.4f}\n\n",
        "## Classification Performance\n",
        f"- **Threshold**: {evaluation_metrics['threshold']:.4f} (strategy: `{evaluation_metrics['strategy']}`)\n",
        f"- **Accuracy**: {evaluation_metrics['accuracy_at_threshold']:.4f}\n",
        f"- **Precision**: {evaluation_metrics['precision_at_threshold']:.4f}\n",
        f"- **Recall**: {evaluation_metrics['recall_at_threshold']:.4f}\n",
        f"- **F1 Score**: {evaluation_metrics['f1_at_threshold']:.4f}\n\n",
        "## Confusion Matrix\n\n",
        "|        | Pred Negative | Pred Positive |\n",
        "|--------|---------------|---------------|\n",
        f"| True Negative | {true_negatives:,} | {false_positives:,} |\n",
        f"| True Positive | {false_negatives:,} | {true_positives:,} |\n\n",
        "## Evaluation Visualizations\n\n",
        "![Combined Evaluation Plots](evaluation_plots.png)\n\n",
        "![High-Precision PR Curve](pr_curve_highlight.png)\n\n",
        "## Precision-Recall Threshold Analysis\n\n",
        "| Target Precision | Required Threshold | Achieved Recall |\n",
        "|------------------|-------------------|------------------|\n"
    ]
    
    # Add precision-recall threshold table
    precision_thresholds = evaluation_metrics.get("pr_thresholds", {})
    for precision_target in sorted(precision_thresholds.keys()):
        threshold_info = precision_thresholds[precision_target]
        threshold_value = threshold_info["threshold"]
        recall_value = threshold_info["recall"]
        
        threshold_display = f"{threshold_value:.4f}" if threshold_value is not None else "N/A"
        recall_display = f"{recall_value:.4f}" if recall_value is not None else "N/A"
        
        report_sections.append(f"| {precision_target:.2f} | {threshold_display} | {recall_display} |\n")
    
    # Save the comprehensive report
    report_path = output_directory / "comprehensive_evaluation_report.md"
    report_path.write_text("".join(report_sections), encoding="utf-8")


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--dataset_parquet", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Path to original dataset with ground truth labels (Parquet format)")
@click.option("--predictions_csv", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Path to model predictions CSV (from S05_make_inference)")
@click.option("--output_dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Directory to save evaluation results and reports")
@click.option("--label_col", default=COL_LABEL, show_default=True,
              help="Name of the ground truth label column")
@click.option("--left_col", default=LEFT_PREFIX, show_default=True,
              help="Name of the left registry name column")
@click.option("--right_col", default=RIGHT_PREFIX, show_default=True,
              help="Name of the right registry name column")
@click.option("--optimize", type=click.Choice(["none", "f1", "youden"], case_sensitive=False),
              default="none", show_default=True,
              help="Threshold optimization strategy")
@click.option("--threshold", type=float, default=None,
              help="Fixed classification threshold (overrides --optimize)")
@click.option("--target_precision", type=float, default=0.95, show_default=True,
              help="Target precision level for visualization highlighting")
def main(dataset_parquet, predictions_csv, output_dir, label_col, left_col, right_col, 
         optimize, threshold, target_precision):
    """
    Comprehensive evaluation of registry deduplication model performance.
    
    This script provides detailed analysis of model predictions including:
    - Performance metrics (ROC AUC, Average Precision, F1, etc.)
    - Threshold optimization for different objectives
    - Comprehensive visualizations (ROC, PR curves, calibration)
    - Error analysis (false positives/negatives with details)
    - Precision-recall threshold mapping
    
    Example usage:
        python S06_eval_model.py \\
            --dataset_parquet data/test_dataset.parquet \\
            --predictions_csv data/predictions.csv \\
            --output_dir data/evaluation_results/ \\
            --optimize f1 \\
            --target_precision 0.95
    """
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Load datasets
    click.echo(f"[INFO] Loading ground truth data from: {dataset_parquet}")
    ground_truth_data = pd.read_parquet(dataset_parquet)
    
    click.echo(f"[INFO] Loading predictions from: {predictions_csv}")
    predictions_data = pd.read_csv(predictions_csv)

    # Standardize prediction score column name
    if "proba" not in predictions_data.columns:
        if "score" in predictions_data.columns:
            predictions_data = predictions_data.rename(columns={"score": "proba"})
        else:
            raise click.ClickException("Predictions CSV must contain 'proba' or 'score' column.")

    # Align predictions with ground truth
    click.echo("[INFO] Aligning predictions with ground truth labels...")
    aligned_data = align_predictions_with_ground_truth(predictions_data, ground_truth_data, left_col, right_col)
    
    if label_col not in aligned_data.columns:
        raise click.ClickException(f"Label column '{label_col}' not found after alignment.")

    # Extract labels and probabilities
    true_labels = aligned_data[label_col].astype(int).to_numpy()
    prediction_probabilities = aligned_data["proba"].astype(float).to_numpy()
    
    click.echo(f"[INFO] Evaluating {len(true_labels):,} predictions")

    # Compute performance curves and metrics
    click.echo("[INFO] Computing performance metrics and curves...")
    roc_auc = roc_auc_score(true_labels, prediction_probabilities)
    average_precision = average_precision_score(true_labels, prediction_probabilities)
    precision_values, recall_values, pr_thresholds = precision_recall_curve(true_labels, prediction_probabilities)
    false_positive_rates, true_positive_rates, roc_thresholds = roc_curve(true_labels, prediction_probabilities)

    # Determine optimal threshold
    optimal_threshold = determine_optimal_classification_threshold(true_labels, prediction_probabilities, optimize, threshold)
    binary_predictions = (prediction_probabilities >= optimal_threshold).astype(int)

    # Compute classification metrics
    accuracy = accuracy_score(true_labels, binary_predictions)
    precision = precision_score(true_labels, binary_predictions, zero_division=0)
    recall = recall_score(true_labels, binary_predictions, zero_division=0)
    f1 = f1_score(true_labels, binary_predictions, zero_division=0)
    confusion_mat = confusion_matrix(true_labels, binary_predictions).tolist()
    brier_score = brier_score_loss(true_labels, binary_predictions)
    
    # Compute log loss with error handling
    try:
        log_loss_value = log_loss(true_labels, np.vstack([1 - prediction_probabilities, prediction_probabilities]).T, labels=[0, 1])
    except Exception:
        log_loss_value = None

    # Save performance curves
    click.echo("[INFO] Saving performance curves...")
    pd.DataFrame({"recall": recall_values, "precision": precision_values, "threshold": np.r_[pr_thresholds, np.nan]}).to_csv(output_directory / "pr_curve.csv", index=False)
    pd.DataFrame({"fpr": false_positive_rates, "tpr": true_positive_rates, "threshold": roc_thresholds}).to_csv(output_directory / "roc_curve.csv", index=False)

    # Save aligned predictions with labels
    essential_columns = [COL_PAIR_ID, left_col, right_col] if left_col in aligned_data.columns and right_col in aligned_data.columns else [COL_PAIR_ID]
    output_columns = [col for col in essential_columns if col in aligned_data.columns]
    predictions_with_labels = aligned_data[output_columns + [label_col, "proba"]].copy() if output_columns else aligned_data[[label_col, "proba"]].copy()
    predictions_with_labels["y_pred"] = binary_predictions
    predictions_with_labels.to_csv(output_directory / "predictions_with_labels.csv", index=False)

    # Generate visualizations
    click.echo("[INFO] Creating evaluation visualizations...")
    create_individual_evaluation_plots(recall_values, precision_values, average_precision, false_positive_rates, true_positive_rates, roc_auc, true_labels, prediction_probabilities, output_directory)
    create_combined_evaluation_figure(recall_values, precision_values, average_precision, false_positive_rates, true_positive_rates, roc_auc, true_labels, prediction_probabilities, output_directory / "evaluation_plots.png")

    # Generate precision-recall threshold analysis
    click.echo("[INFO] Analyzing precision-recall thresholds...")
    pr_threshold_analysis = generate_precision_recall_thresholds_analysis(true_labels, prediction_probabilities, output_directory)

    # Compile comprehensive metrics
    comprehensive_metrics = {
        "evaluation_summary": {
            "n_samples": int(len(true_labels)),
            "prevalence": float(np.mean(true_labels)),
            "auroc": float(roc_auc),
            "average_precision": float(average_precision),
        },
        "threshold_analysis": {
            "strategy": optimize,
            "threshold": float(optimal_threshold),
        },
        "classification_performance": {
            "accuracy_at_threshold": float(accuracy),
            "precision_at_threshold": float(precision),
            "recall_at_threshold": float(recall),
            "f1_at_threshold": float(f1),
            "brier_score": float(brier_score),
            "log_loss": float(log_loss_value) if log_loss_value is not None else None,
            "confusion_matrix": confusion_mat,
        },
        "pr_thresholds": pr_threshold_analysis,
    }

    # Save metrics
    metrics_path = output_directory / "comprehensive_metrics.json"
    with open(metrics_path, "w") as metrics_file:
        json.dump(comprehensive_metrics, metrics_file, indent=2)

    # Generate reports
    click.echo("[INFO] Generating detailed reports...")
    
    # Flatten metrics for report generation
    flattened_metrics = {
        **comprehensive_metrics["evaluation_summary"],
        **comprehensive_metrics["threshold_analysis"], 
        **comprehensive_metrics["classification_performance"],
        "pr_thresholds": comprehensive_metrics["pr_thresholds"]
    }
    
    generate_comprehensive_markdown_report(output_directory, flattened_metrics)
    create_detailed_error_analysis_report(predictions_with_labels, output_directory)

    # Save additional analysis files
    false_positives = predictions_with_labels[(predictions_with_labels[label_col] == 0) & (predictions_with_labels["y_pred"] == 1)]
    false_negatives = predictions_with_labels[(predictions_with_labels[label_col] == 1) & (predictions_with_labels["y_pred"] == 0)]
    false_positives.to_csv(output_directory / "false_positives.csv", index=False)
    false_negatives.to_csv(output_directory / "false_negatives.csv", index=False)

    # Generate classification report
    classification_report_text = classification_report(true_labels, binary_predictions, digits=4, zero_division=0)
    (output_directory / "classification_report.txt").write_text(classification_report_text)

    click.echo(f"\n[SUCCESS] Comprehensive evaluation completed!")
    click.echo(f"[SUCCESS] ROC AUC: {roc_auc:.4f}")
    click.echo(f"[SUCCESS] Average Precision: {average_precision:.4f}")
    click.echo(f"[SUCCESS] F1 Score: {f1:.4f}")
    click.echo(f"[SUCCESS] Results saved to: {output_directory}")

if __name__ == "__main__":
    main()
