#!/usr/bin/env python3
"""
Production Inference Engine for Registry Deduplication

This script runs inference using a trained sklearn pipeline to generate deduplication
predictions on registry pairs. It processes raw registry data through the complete
pipeline (text normalization → feature extraction → classification) and outputs
predictions with confidence scores.

Input Requirements:
    - inference_pipeline.joblib: Complete trained pipeline from S04_make_pipeline
    - dataset.parquet: Registry pairs dataset with required columns
    
Output:
    - predictions.csv: Registry pairs with similarity scores
    - score_distribution.png: Distribution plot of prediction scores
"""

from __future__ import annotations
from pathlib import Path
import click
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from deduplication_model.config import COLUMN_CONFIG, COL_PAIR_ID


def standardize_column_names_for_inference(input_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to ensure compatibility with the inference pipeline.
    
    Some datasets may have singular column names (medical_condition, geographical_area)
    while the pipeline expects plural forms. This function handles the conversion.
    
    Args:
        input_dataframe: Raw input dataframe with potentially inconsistent column names
        
    Returns:
        DataFrame with standardized column names expected by the pipeline
    """
    standardized_dataframe = input_dataframe.copy()
    
    for column_type in ["medical_condition", "geographical_area"]:
        for side in ["left", "right"]:
            expected_plural_column = COLUMN_CONFIG[column_type][side]
            potential_singular_column = expected_plural_column.replace("s", "")
            
            # Case 1: Singular exists but plural doesn't - rename to plural
            if (potential_singular_column in standardized_dataframe.columns and 
                expected_plural_column not in standardized_dataframe.columns):
                click.echo(f"[INFO] Standardizing column name: {potential_singular_column} → {expected_plural_column}")
                standardized_dataframe = standardized_dataframe.rename(
                    columns={potential_singular_column: expected_plural_column}
                )
            
            # Case 2: Neither exists - create empty plural column
            elif (potential_singular_column not in standardized_dataframe.columns and 
                  expected_plural_column not in standardized_dataframe.columns):
                click.echo(f"[INFO] Creating missing column: {expected_plural_column}")
                standardized_dataframe[expected_plural_column] = ""
    
    return standardized_dataframe


def validate_pipeline_input_requirements(
    input_dataframe: pd.DataFrame, 
    required_columns: list
) -> None:
    """
    Validate that the input dataframe contains all columns required by the pipeline.
    
    Args:
        input_dataframe: DataFrame to validate
        required_columns: List of column names required by the pipeline
        
    Raises:
        click.ClickException: If any required columns are missing
    """
    missing_columns = [col for col in required_columns if col not in input_dataframe.columns]
    if missing_columns:
        available_columns = sorted(input_dataframe.columns.tolist())
        raise click.ClickException(
            f"Missing required columns: {missing_columns}\n"
            f"Available columns: {available_columns}"
        )


def generate_prediction_scores_safely(inference_pipeline, input_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Generate prediction scores using the inference pipeline with comprehensive error handling.
    
    Args:
        inference_pipeline: Trained sklearn pipeline
        input_dataframe: Input data for inference
        
    Returns:
        DataFrame with prediction scores
        
    Raises:
        click.ClickException: If inference fails with detailed error information
    """
    try:
        # Generate probability scores (confidence in "match" class)
        probability_scores = inference_pipeline.predict_proba(input_dataframe)[:, 1]
        return probability_scores
        
    except Exception as inference_error:
        click.echo(f"[ERROR] Inference pipeline execution failed: {inference_error}")
        
        # Provide detailed debugging information for pipeline components
        if hasattr(inference_pipeline, "steps"):
            click.echo("[DEBUG] Pipeline component analysis:")
            for step_name, step_component in inference_pipeline.steps:
                click.echo(f"[DEBUG] Step '{step_name}': {type(step_component).__name__}")
                
                # Special handling for feature extraction step
                if step_name == "feature_extraction" and hasattr(step_component, "featurizers"):
                    for featurizer_name, featurizer in step_component.featurizers:
                        if hasattr(featurizer, "left_col") and hasattr(featurizer, "right_col"):
                            left_column = featurizer.left_col
                            right_column = featurizer.right_col
                            click.echo(f"[DEBUG] Featurizer '{featurizer_name}' expects: {left_column}, {right_column}")
                            
                            if left_column not in input_dataframe.columns:
                                click.echo(f"[DEBUG] Missing left column: {left_column}")
                            if right_column not in input_dataframe.columns:
                                click.echo(f"[DEBUG] Missing right column: {right_column}")
        
        raise click.ClickException(f"Pipeline inference failed: {inference_error}")


def create_score_distribution_visualization(
    prediction_scores: pd.Series, 
    output_directory: Path
) -> Path:
    """
    Create and save a visualization of the prediction score distribution.
    
    Args:
        prediction_scores: Array of prediction scores (0-1 range)
        output_directory: Directory to save the plot
        
    Returns:
        Path to the saved plot file
    """
    plt.figure(figsize=(10, 6))
    
    # Create histogram with appropriate binning
    plt.hist(prediction_scores, bins=50, edgecolor="black", alpha=0.7, color="skyblue")
    
    # Add statistical information
    mean_score = prediction_scores.mean()
    median_score = prediction_scores.median()
    
    plt.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.3f}')
    plt.axvline(median_score, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_score:.3f}')
    
    # Formatting
    plt.title("Distribution of Deduplication Prediction Scores", fontsize=14, fontweight='bold')
    plt.xlabel("Prediction Score (Probability of Match)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add summary statistics as text
    stats_text = (f"Total Pairs: {len(prediction_scores):,}\n"
                 f"Min Score: {prediction_scores.min():.3f}\n"
                 f"Max Score: {prediction_scores.max():.3f}\n"
                 f"Std Dev: {prediction_scores.std():.3f}")
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_output_path = output_directory / "score_distribution.png"
    plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_output_path


def save_predictions_with_metadata(
    input_dataframe: pd.DataFrame,
    prediction_scores: pd.Series,
    output_directory: Path,
    required_columns: list
) -> Path:
    """
    Save predictions along with original data and comprehensive metadata.
    
    Args:
        input_dataframe: Original input dataframe
        prediction_scores: Generated prediction scores
        output_directory: Directory to save outputs
        required_columns: Columns that were required for inference
        
    Returns:
        Path to the saved predictions file
    """
    # Create output dataframe with essential columns and predictions
    prediction_columns = [col for col in required_columns if col in input_dataframe.columns]
    output_dataframe = input_dataframe[prediction_columns].copy()
    output_dataframe["score"] = prediction_scores
    
    # Save predictions
    predictions_output_path = output_directory / "predictions.csv"
    output_dataframe.to_csv(predictions_output_path, index=False)
    
    return predictions_output_path


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--pipeline",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the complete inference pipeline (joblib file from S04_make_pipeline)"
)
@click.option(
    "--dataset_parquet",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to registry pairs dataset (Parquet format) for inference"
)
@click.option(
    "--output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory to save prediction results and visualizations"
)
def main(pipeline: Path, dataset_parquet: Path, output_dir: Path):
    """
    Generate deduplication predictions for registry pairs using a trained pipeline.
    
    This script processes registry pairs through a complete machine learning pipeline
    to generate similarity scores indicating the likelihood that two registries are
    duplicates of each other.
    
    The pipeline performs:
    1. Text normalization of registry names
    2. Feature extraction (similarity measures, embeddings, etc.)
    3. Classification using a trained gradient boosting model
    
    Output scores range from 0 (definitely not a match) to 1 (definitely a match).
    
    Example usage:
        python S05_make_inference.py \\
            --pipeline data/production_pipeline/inference_pipeline.joblib \\
            --dataset_parquet data/registry_pairs_to_score.parquet \\
            --output_dir data/inference_results/
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the trained inference pipeline
    click.echo(f"[INFO] Loading inference pipeline from: {pipeline}")
    trained_inference_pipeline = joblib.load(pipeline)
    click.echo("[INFO] Pipeline loaded successfully")

    # Load and examine the input dataset
    click.echo(f"[INFO] Loading dataset from: {dataset_parquet}")
    input_dataset = pd.read_parquet(dataset_parquet)
    click.echo(f"[INFO] Loaded {len(input_dataset):,} registry pairs for inference")
    click.echo(f"[INFO] Dataset columns: {sorted(input_dataset.columns.tolist())}")

    # Standardize column names for pipeline compatibility
    standardized_dataset = standardize_column_names_for_inference(input_dataset)

    # Define required columns for the pipeline
    essential_columns = [
        COL_PAIR_ID,
        COLUMN_CONFIG["name"]["left"],
        COLUMN_CONFIG["name"]["right"]
    ]

    # Validate that all required columns are present
    validate_pipeline_input_requirements(standardized_dataset, essential_columns)

    # Generate prediction scores
    click.echo("[INFO] Running inference pipeline to generate prediction scores...")
    prediction_scores = generate_prediction_scores_safely(trained_inference_pipeline, standardized_dataset)
    
    click.echo(f"[INFO] Generated scores for {len(prediction_scores):,} pairs")
    click.echo(f"[INFO] Score statistics - Mean: {prediction_scores.mean():.3f}, "
               f"Median: {pd.Series(prediction_scores).median():.3f}, "
               f"Range: [{prediction_scores.min():.3f}, {prediction_scores.max():.3f}]")

    # Create and save score distribution visualization
    click.echo("[INFO] Creating score distribution visualization...")
    distribution_plot_path = create_score_distribution_visualization(
        pd.Series(prediction_scores), output_dir
    )
    click.echo(f"[SUCCESS] Score distribution plot saved to: {distribution_plot_path}")

    # Save predictions with metadata
    click.echo("[INFO] Saving predictions with metadata...")
    predictions_file_path = save_predictions_with_metadata(
        standardized_dataset, prediction_scores, output_dir, essential_columns
    )
    click.echo(f"[SUCCESS] Predictions saved to: {predictions_file_path}")

    # Summary report
    click.echo(f"\n[SUCCESS] Inference completed successfully!")
    click.echo(f"[SUCCESS] Processed {len(input_dataset):,} registry pairs")
    click.echo(f"[SUCCESS] Results saved to: {output_dir}")
    click.echo(f"[SUCCESS] Use the scores to identify potential duplicate registries")

if __name__ == "__main__":
    main()
