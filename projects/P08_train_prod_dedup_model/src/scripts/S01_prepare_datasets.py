#!/usr/bin/env python3
"""
Dataset Preparation: Train/Test Split with Data Augmentation

This script splits a labeled dataset of registry pairs into stratified train/test sets
and augments the training data with synthetic exact match samples to improve model learning.

Input Requirements:
    - Parquet file with registry pairs containing:
        * pair_hash_id: Unique identifier for each pair
        * label: Binary label (0=no match, 1=match)
        * left_name, right_name: Registry names to compare
        * left_embedding, right_embedding: Semantic embeddings
        * left_medical_conditions, right_medical_conditions: Medical condition data
        * left_geographical_areas, right_geographical_areas: Geographic data

Output:
    - train_dataset.parquet: Training set with augmented exact matches
    - test_dataset.parquet: Test set
    - train_indices.npy: Original indices used for training
    - test_indices.npy: Original indices used for testing  
    - metadata.json: Split statistics and configuration
"""

import json
from pathlib import Path
import click
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import hashlib
from deduplication_model.config import (
    COLUMN_CONFIG,
    COL_PAIR_ID,
    COL_LABEL,
    LEFT_PREFIX,
    RIGHT_PREFIX,
    COL_NAME, COL_ACRONYM,
    COL_OBJECT_ID,
    COL_EMBEDDING,
    COL_MEDICAL_CONDITION,
    COL_GEOGRAPHICAL_AREA,
)

def generate_deterministic_pair_hash(left_registry_id: str, right_registry_id: str) -> str:
    """
    Generate a deterministic hash for a pair of registry IDs.
    
    Args:
        left_registry_id: ID of the left registry
        right_registry_id: ID of the right registry
        
    Returns:
        MD5 hash string representing the pair
    """
    pair_identifier = f"{left_registry_id}::{right_registry_id}"
    return hashlib.md5(pair_identifier.encode("utf-8")).hexdigest()

def create_synthetic_exact_match_samples(training_dataframe: pd.DataFrame, random_seed: int) -> tuple[pd.DataFrame, int]:
    """
    Create synthetic exact match samples by duplicating left-side data to right-side.
    
    This augmentation helps the model learn to identify perfect matches by providing
    examples where all features are identical between left and right registries.
    
    Args:
        training_dataframe: Original training dataset
        random_seed: Random seed for reproducible sampling
        
    Returns:
        Tuple of (augmented_dataframe, number_of_samples_added)
    """
    # Sample 5% of training data for exact match augmentation
    sampling_fraction = 0.05
    sampled_registries = training_dataframe.sample(
        frac=sampling_fraction, 
        random_state=random_seed
    ).reset_index(drop=True)
    
    # Extract only left-side columns for duplication
    left_side_columns = [
        f"{LEFT_PREFIX}_{COL_NAME}", 
        f"{LEFT_PREFIX}_{COL_ACRONYM}", 
        f"{LEFT_PREFIX}_{COL_OBJECT_ID}", 
        f"{LEFT_PREFIX}_{COL_EMBEDDING}",
        f"{LEFT_PREFIX}_{COL_MEDICAL_CONDITION}", 
        f"{LEFT_PREFIX}_{COL_GEOGRAPHICAL_AREA}"
    ]
    
    exact_match_samples = sampled_registries[left_side_columns].copy()
    
    # Duplicate left columns to create matching right columns
    for left_column in exact_match_samples.columns:
        right_column = left_column.replace(f"{LEFT_PREFIX}_", f"{RIGHT_PREFIX}_")
        exact_match_samples[right_column] = exact_match_samples[left_column]
        
    # Generate unique pair identifiers for the synthetic samples
    exact_match_samples[COL_PAIR_ID] = [
        generate_deterministic_pair_hash(str(left_id), str(right_id))
        for left_id, right_id in zip(
            exact_match_samples[f"{LEFT_PREFIX}_{COL_OBJECT_ID}"],
            exact_match_samples[f"{RIGHT_PREFIX}_{COL_OBJECT_ID}"]
        )
    ]
    
    # Set all synthetic samples as positive matches
    exact_match_samples[COL_LABEL] = 1
    
    # Ensure column order matches original dataset
    exact_match_samples = exact_match_samples[training_dataframe.columns]
    
    # Combine original training data with synthetic exact matches
    augmented_training_data = pd.concat(
        [training_dataframe, exact_match_samples], 
        ignore_index=True
    )
    
    number_of_synthetic_samples = len(exact_match_samples)
    
    return augmented_training_data, number_of_synthetic_samples

def validate_required_dataset_columns(dataset: pd.DataFrame, dataset_name: str) -> None:
    """
    Validate that the dataset contains all required columns for processing.
    
    Args:
        dataset: DataFrame to validate
        dataset_name: Name of the dataset (for error messages)
        
    Raises:
        click.ClickException: If any required columns are missing
    """
    required_columns = [
        COL_PAIR_ID,
        COL_LABEL,
        f"{LEFT_PREFIX}_{COL_NAME}",
        f"{RIGHT_PREFIX}_{COL_NAME}",
        f"{LEFT_PREFIX}_{COL_EMBEDDING}",
        f"{RIGHT_PREFIX}_{COL_EMBEDDING}",
        f"{LEFT_PREFIX}_{COL_MEDICAL_CONDITION}",
        f"{RIGHT_PREFIX}_{COL_MEDICAL_CONDITION}",
        f"{LEFT_PREFIX}_{COL_GEOGRAPHICAL_AREA}",
        f"{RIGHT_PREFIX}_{COL_GEOGRAPHICAL_AREA}",
    ]
    
    missing_columns = [col for col in required_columns if col not in dataset.columns]
    if missing_columns:
        raise click.ClickException(
            f"Dataset '{dataset_name}' is missing required columns: {missing_columns}"
        )

def save_dataset_split_metadata(
    output_directory: Path,
    source_file_path: Path,
    split_configuration: dict,
    training_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame
) -> None:
    """
    Save comprehensive metadata about the dataset split process.
    
    Args:
        output_directory: Directory where metadata will be saved
        source_file_path: Path to the original dataset file
        split_configuration: Configuration parameters used for splitting
        training_dataset: Training set after augmentation
        test_dataset: Test set
    """
    # Calculate class distributions for both sets
    training_class_distribution = {
        "positive_matches": int(training_dataset[COL_LABEL].sum()),
        "negative_matches": int(len(training_dataset) - training_dataset[COL_LABEL].sum()),
    }
    
    test_class_distribution = {
        "positive_matches": int(test_dataset[COL_LABEL].sum()),
        "negative_matches": int(len(test_dataset) - test_dataset[COL_LABEL].sum()),
    }
    
    metadata = {
        "source_information": {
            "original_file": str(source_file_path),
            "total_samples": split_configuration["original_dataset_size"],
        },
        "split_configuration": {
            "test_size_fraction": split_configuration["test_size"],
            "random_seed": split_configuration["random_state"],
            "stratification_enabled": True,
            "exact_match_augmentation_enabled": True,
        },
        "resulting_datasets": {
            "training_set": {
                "total_samples": len(training_dataset),
                "original_samples": split_configuration["original_training_size"],
                "synthetic_exact_matches": split_configuration["synthetic_samples_added"],
                "class_distribution": training_class_distribution,
            },
            "test_set": {
                "total_samples": len(test_dataset),
                "class_distribution": test_class_distribution,
            },
        },
        "quality_metrics": {
            "training_positive_rate": training_class_distribution["positive_matches"] / len(training_dataset),
            "test_positive_rate": test_class_distribution["positive_matches"] / len(test_dataset),
        }
    }
    
    metadata_file_path = output_directory / "metadata.json"
    with open(metadata_file_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--eval_dataset_parquet",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the labeled registry pairs dataset (Parquet format)",
)
@click.option(
    "--test_size", 
    default=0.2, 
    show_default=True, 
    type=float,
    help="Fraction of data to reserve for testing (0.0 to 1.0)"
)
@click.option(
    "--random_state", 
    default=42, 
    show_default=True, 
    type=int,
    help="Random seed for reproducible train/test splits"
)
@click.option(
    "--output_dir", 
    type=click.Path(file_okay=False, path_type=Path), 
    required=True,
    help="Directory where split datasets and metadata will be saved"
)
def main(
    eval_dataset_parquet: Path, 
    test_size: float, 
    random_state: int, 
    output_dir: Path
):
    """
    Split labeled registry pairs into stratified train/test sets with data augmentation.
    
    This script performs stratified splitting to maintain class balance and augments
    the training set with synthetic exact match samples to improve model learning.
    
    The process includes:
    1. Loading and validating the input dataset
    2. Performing stratified train/test split to maintain class balance
    3. Augmenting training data with synthetic exact match samples
    4. Saving split datasets and comprehensive metadata
    
    Example usage:
        python S01_prepare_datasets.py \\
            --eval_dataset_parquet data/registry_pairs.parquet \\
            --test_size 0.2 \\
            --random_state 42 \\
            --output_dir data/splits/
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and validate the evaluation dataset
    click.echo(f"[INFO] Loading dataset from: {eval_dataset_parquet}")
    original_dataset = pd.read_parquet(eval_dataset_parquet)
    click.echo(f"[INFO] Loaded {len(original_dataset):,} registry pairs")

    validate_required_dataset_columns(original_dataset, "input dataset")

    # Prepare data for stratified splitting
    class_labels = original_dataset[COL_LABEL].astype(int).values
    sample_indices = np.arange(len(original_dataset))

    # Perform stratified train/test split
    click.echo(f"[INFO] Performing stratified split (test_size={test_size:.1%})")
    training_indices, test_indices = train_test_split(
        sample_indices, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=class_labels
    )

    # Create initial training and test sets
    initial_training_set = original_dataset.iloc[training_indices].reset_index(drop=True)
    final_test_set = original_dataset.iloc[test_indices].reset_index(drop=True)

    # Augment training set with synthetic exact match samples
    click.echo("[INFO] Augmenting training set with synthetic exact match samples...")
    final_training_set, synthetic_samples_count = create_synthetic_exact_match_samples(
        initial_training_set, random_state
    )
    
    click.echo(f"[INFO] Added {synthetic_samples_count:,} synthetic exact match samples")
    click.echo(f"[INFO] Final training set size: {len(final_training_set):,} samples")
    click.echo(f"[INFO] Test set size: {len(final_test_set):,} samples")

    # Save the split datasets
    training_output_path = output_dir / "train_dataset.parquet"
    test_output_path = output_dir / "test_dataset.parquet"
    
    final_training_set.to_parquet(training_output_path)
    final_test_set.to_parquet(test_output_path)
    
    # Save the original indices for reference
    np.save(output_dir / "train_indices.npy", training_indices)
    np.save(output_dir / "test_indices.npy", test_indices)

    # Save comprehensive metadata
    split_configuration = {
        "original_dataset_size": len(original_dataset),
        "original_training_size": len(initial_training_set),
        "test_size": test_size,
        "random_state": random_state,
        "synthetic_samples_added": synthetic_samples_count,
    }
    
    save_dataset_split_metadata(
        output_dir, 
        eval_dataset_parquet, 
        split_configuration,
        final_training_set, 
        final_test_set
    )

    click.echo(f"[SUCCESS] Dataset split completed successfully!")
    click.echo(f"[SUCCESS] Training data saved to: {training_output_path}")
    click.echo(f"[SUCCESS] Test data saved to: {test_output_path}")
    click.echo(f"[SUCCESS] Metadata saved to: {output_dir / 'metadata.json'}")

if __name__ == "__main__":
    main()