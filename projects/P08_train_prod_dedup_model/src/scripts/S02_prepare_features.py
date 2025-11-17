#!/usr/bin/env python3
"""
Feature Engineering Pipeline for Registry Deduplication

This script processes registry pair datasets to extract comprehensive similarity features
for training a deduplication model. It applies text normalization and generates multiple
types of features including exact matches, string similarities, semantic similarities,
and missing pattern indicators.

Input Requirements:
    - train_dataset.parquet: Training pairs with labels
    - test_dataset.parquet: Test pairs with labels
    
Output:
    - Feature matrices (X_train.npy, X_test.npy) 
    - Label arrays (y_train.npy, y_test.npy)
    - Feature pipeline and normalizer (for inference)
    - Feature metadata and distribution plots
    - Debug CSV files with all features
"""

import json
import pickle
from pathlib import Path

import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import FeatureUnion

from deduplication_model.normalization.text_normalizer import RegistryNameNormalizer

# String similarity featurizers
from deduplication_model.features.jw_featurizer import JaroWinklerFeaturizer
from deduplication_model.features.levenshtein_featurizer import LevenshteinNormSimFeaturizer
from deduplication_model.features.lcs_featurizer import LCSNormalizedFeaturizer
from deduplication_model.features.jaccard_featurizer import TokenJaccardFeaturizer
from deduplication_model.features.char_ngram_jaccard_featurizer import CharNgramJaccardFeaturizer

# TF-IDF based featurizers
from deduplication_model.features.tfidf_token_cosine_featurizer import TfidfTokenCosineFeaturizer
from deduplication_model.features.tfidf_char_cosine_featurizer import TfidfCharCosineFeaturizer
from deduplication_model.features.soft_tfidf_featurizer import SoftTfidfFeaturizer

# Specialized featurizers
from deduplication_model.features.token_stats_featurizer import TokenStatsFeaturizer
from deduplication_model.features.shared_rare_token_frac_featurizer import SharedRareTokenFracFeaturizer
from deduplication_model.features.exact_match_featurizer import ExactNormalizedMatchFeaturizer
from deduplication_model.features.token_bag_match_featurizer import TokenBagMatchFeaturizer
from deduplication_model.features.semantic_featurizer import SemanticCosineFeaturizer

# Acronym-specific featurizers
from deduplication_model.features.acronym_featurizer import (
    AcronymExactMatchFeaturizer,
    AcronymLengthDiffFeaturizer,
    AcronymJaccardFeaturizer,
)

# Missing pattern featurizers
from deduplication_model.features.missing_pattern_featurizer import (
    BothPresentFeaturizer,
    BothMissingFeaturizer,
    OneMissingFeaturizer,
)

from deduplication_model.normalization.rules import DEFAULT_STOPWORDS
from deduplication_model.config import COLUMN_CONFIG, COL_PAIR_ID, COL_LABEL


def create_comprehensive_feature_dataframe(
    feature_matrix: np.ndarray, 
    feature_names: list, 
    normalized_dataframe: pd.DataFrame, 
    label_column: str, 
    labels: np.ndarray, 
    text_normalizer: RegistryNameNormalizer
) -> pd.DataFrame:
    """
    Create a comprehensive dataframe combining features with original data columns.
    
    Args:
        feature_matrix: Computed feature matrix (n_samples x n_features)
        feature_names: Names of the computed features
        normalized_dataframe: Original dataframe with normalized text columns
        label_column: Name of the label column
        labels: Array of binary labels
        text_normalizer: Fitted text normalizer instance
        
    Returns:
        DataFrame with features, labels, and original columns
    """
    # Start with the feature matrix as base dataframe
    comprehensive_dataframe = pd.DataFrame(feature_matrix, columns=feature_names)
    comprehensive_dataframe[label_column] = labels
    
    # Add all original columns from the configuration
    for column_type, column_mapping in COLUMN_CONFIG.items():
        for side in ['left', 'right']:
            original_column = column_mapping[side]
            if original_column in normalized_dataframe.columns:
                comprehensive_dataframe[original_column] = (
                    normalized_dataframe[original_column].astype(str).to_numpy()
                )
    
    # Add normalized text columns created by the text normalizer
    normalized_column_mapping = {
        f"{COLUMN_CONFIG['name']['left']}{text_normalizer.normalized_suffix}": 
            normalized_dataframe[f"{COLUMN_CONFIG['name']['left']}{text_normalizer.normalized_suffix}"].astype(str).to_numpy(),
        f"{COLUMN_CONFIG['name']['right']}{text_normalizer.normalized_suffix}": 
            normalized_dataframe[f"{COLUMN_CONFIG['name']['right']}{text_normalizer.normalized_suffix}"].astype(str).to_numpy()
    }
    comprehensive_dataframe.update(normalized_column_mapping)
    
    # Replace empty strings with None for better data quality
    return comprehensive_dataframe.replace("", None)


def create_feature_distribution_plots(
    training_features: np.ndarray, 
    test_features: np.ndarray, 
    feature_names: list, 
    output_directory: Path
) -> None:
    """
    Generate distribution plots for each feature to analyze data quality and differences.
    
    Args:
        training_features: Training feature matrix
        test_features: Test feature matrix  
        feature_names: List of feature names
        output_directory: Directory to save plots
    """
    plots_directory = output_directory / "feature_distributions"
    plots_directory.mkdir(exist_ok=True, parents=True)
    
    for feature_index, feature_name in enumerate(feature_names):
        plt.figure(figsize=(10, 6))
        
        # Extract feature values for train and test sets
        train_feature_values = training_features[:, feature_index]
        test_feature_values = test_features[:, feature_index]
        
        # Calculate missing value statistics
        train_missing_percentage = np.sum(np.isnan(train_feature_values)) / len(train_feature_values) * 100
        test_missing_percentage = np.sum(np.isnan(test_feature_values)) / len(test_feature_values) * 100
        
        # Filter out NaN values for plotting
        train_values_clean = train_feature_values[~np.isnan(train_feature_values)]
        test_values_clean = test_feature_values[~np.isnan(test_feature_values)]
        
        # Create histograms if data is available after removing NaNs
        if len(train_values_clean) > 0:
            plt.hist(train_values_clean, bins=30, alpha=0.6, label="Training", density=True, color='blue')
        if len(test_values_clean) > 0:
            plt.hist(test_values_clean, bins=30, alpha=0.6, label="Test", density=True, color='orange')
        
        # Add statistical information
        plt.title(f"Feature Distribution: {feature_name}")
        plt.xlabel(f"Feature Value ({feature_name})")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add missing data information as text
        missing_data_text = f"Missing Values - Training: {train_missing_percentage:.1f}%, Test: {test_missing_percentage:.1f}%"
        plt.figtext(0.5, 0.02, missing_data_text, ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plot_save_path = plots_directory / f"{feature_name.replace('/', '_').replace(' ', '_')}.png"
        plt.savefig(plot_save_path, dpi=150, bbox_inches='tight')
        plt.close()


def configure_registry_name_text_normalizer() -> RegistryNameNormalizer:
    """
    Configure the text normalizer for registry names with standard preprocessing.
    
    Returns:
        Configured RegistryNameNormalizer instance
    """
    return RegistryNameNormalizer(
        left_col=COLUMN_CONFIG["name"]["left"],
        right_col=COLUMN_CONFIG["name"]["right"],
        output_mode="append",  # Add normalized columns alongside originals
        remove_drop_terms=True,  # Remove common drop terms
        remove_stopwords=True,   # Remove stopwords
        sort_tokens=True,        # Sort tokens alphabetically for consistency
    )


def build_comprehensive_feature_pipeline() -> FeatureUnion:
    """
    Build a comprehensive feature extraction pipeline with multiple similarity measures.
    
    Returns:
        Configured FeatureUnion with all featurizers
    """
    # Get reference to the text normalizer to get the correct suffix
    text_normalizer = configure_registry_name_text_normalizer()
    
    # Get column names for normalized text using the normalizer's actual suffix
    normalized_left_column = f"{COLUMN_CONFIG['name']['left']}{text_normalizer.normalized_suffix}"
    normalized_right_column = f"{COLUMN_CONFIG['name']['right']}{text_normalizer.normalized_suffix}"
    
    # Original name column featurizers
    original_name_featurizers = [
        ("raw_exact_match", ExactNormalizedMatchFeaturizer(
            left_col=COLUMN_CONFIG["name"]["left"], 
            right_col=COLUMN_CONFIG["name"]["right"]
        )),
        ("raw_token_bag_match", TokenBagMatchFeaturizer(
            left_col=COLUMN_CONFIG["name"]["left"], 
            right_col=COLUMN_CONFIG["name"]["right"]
        )),
        ("raw_jaro_winkler", JaroWinklerFeaturizer(
            left_col=COLUMN_CONFIG["name"]["left"], 
            right_col=COLUMN_CONFIG["name"]["right"]
        )),
        ("raw_levenshtein", LevenshteinNormSimFeaturizer(
            left_col=COLUMN_CONFIG["name"]["left"], 
            right_col=COLUMN_CONFIG["name"]["right"]
        )),
        ("raw_token_jaccard", TokenJaccardFeaturizer(
            left_col=COLUMN_CONFIG["name"]["left"], 
            right_col=COLUMN_CONFIG["name"]["right"], 
            stopwords=DEFAULT_STOPWORDS
        )),
        ("raw_tfidf_token_cosine", TfidfTokenCosineFeaturizer(
            left_col=COLUMN_CONFIG["name"]["left"], 
            right_col=COLUMN_CONFIG["name"]["right"]
        )),
        ("raw_tfidf_char_cosine", TfidfCharCosineFeaturizer(
            left_col=COLUMN_CONFIG["name"]["left"], 
            right_col=COLUMN_CONFIG["name"]["right"]
        )),
        ("raw_soft_tfidf", SoftTfidfFeaturizer(
            left_col=COLUMN_CONFIG["name"]["left"], 
            right_col=COLUMN_CONFIG["name"]["right"]
        )),
        ("raw_token_statistics", TokenStatsFeaturizer(
            left_col=COLUMN_CONFIG["name"]["left"], 
            right_col=COLUMN_CONFIG["name"]["right"], 
            stopwords=DEFAULT_STOPWORDS
        )),
        ("raw_lcs_normalized", LCSNormalizedFeaturizer(
            left_col=COLUMN_CONFIG["name"]["left"], 
            right_col=COLUMN_CONFIG["name"]["right"]
        )),
        ("raw_rare_token_fraction", SharedRareTokenFracFeaturizer(
            left_col=COLUMN_CONFIG["name"]["left"], 
            right_col=COLUMN_CONFIG["name"]["right"], 
            stopwords=DEFAULT_STOPWORDS
        )),
        ("raw_char_ngram_jaccard", CharNgramJaccardFeaturizer(
            left_col=COLUMN_CONFIG["name"]["left"], 
            right_col=COLUMN_CONFIG["name"]["right"]
        )),
    ]
    
    # Normalized name column featurizers (same algorithms on cleaned text)
    normalized_name_featurizers = [
        ("normalized_exact_match", ExactNormalizedMatchFeaturizer(
            left_col=normalized_left_column, 
            right_col=normalized_right_column
        )),
        ("normalized_token_bag_match", TokenBagMatchFeaturizer(
            left_col=normalized_left_column, 
            right_col=normalized_right_column
        )),
        ("normalized_jaro_winkler", JaroWinklerFeaturizer(
            left_col=normalized_left_column, 
            right_col=normalized_right_column
        )),
        ("normalized_levenshtein", LevenshteinNormSimFeaturizer(
            left_col=normalized_left_column, 
            right_col=normalized_right_column
        )),
        ("normalized_token_jaccard", TokenJaccardFeaturizer(
            left_col=normalized_left_column, 
            right_col=normalized_right_column, 
            stopwords=DEFAULT_STOPWORDS
        )),
        ("normalized_tfidf_token_cosine", TfidfTokenCosineFeaturizer(
            left_col=normalized_left_column, 
            right_col=normalized_right_column
        )),
        ("normalized_tfidf_char_cosine", TfidfCharCosineFeaturizer(
            left_col=normalized_left_column, 
            right_col=normalized_right_column
        )),
        ("normalized_soft_tfidf", SoftTfidfFeaturizer(
            left_col=normalized_left_column, 
            right_col=normalized_right_column
        )),
        ("normalized_token_statistics", TokenStatsFeaturizer(
            left_col=normalized_left_column, 
            right_col=normalized_right_column, 
            stopwords=DEFAULT_STOPWORDS
        )),
        ("normalized_lcs", LCSNormalizedFeaturizer(
            left_col=normalized_left_column, 
            right_col=normalized_right_column
        )),
        ("normalized_rare_token_fraction", SharedRareTokenFracFeaturizer(
            left_col=normalized_left_column, 
            right_col=normalized_right_column, 
            stopwords=DEFAULT_STOPWORDS
        )),
        ("normalized_char_ngram_jaccard", CharNgramJaccardFeaturizer(
            left_col=normalized_left_column, 
            right_col=normalized_right_column
        )),
    ]
    
    # Acronym-specific featurizers
    acronym_featurizers = [
        ("acronym_exact_match", AcronymExactMatchFeaturizer(
            left_col=COLUMN_CONFIG["acronym"]["left"], 
            right_col=COLUMN_CONFIG["acronym"]["right"]
        )),
        ("acronym_length_difference", AcronymLengthDiffFeaturizer(
            left_col=COLUMN_CONFIG["acronym"]["left"], 
            right_col=COLUMN_CONFIG["acronym"]["right"]
        )),
        ("acronym_jaccard", AcronymJaccardFeaturizer(
            left_col=COLUMN_CONFIG["acronym"]["left"], 
            right_col=COLUMN_CONFIG["acronym"]["right"]
        )),
    ]
    
    # Semantic similarity featurizers
    semantic_featurizers = [
        ("semantic_cosine_similarity", SemanticCosineFeaturizer(
            left_col=COLUMN_CONFIG["embedding"]["left"], 
            right_col=COLUMN_CONFIG["embedding"]["right"]
        )),
    ]
    
    # Missing pattern featurizers for different column types
    missing_pattern_featurizers = []
    for column_type in ["acronym", "geographical_area", "medical_condition"]:
        column_mapping = COLUMN_CONFIG[column_type]
        missing_pattern_featurizers.extend([
            (f"{column_type}_both_present", BothPresentFeaturizer(
                left_col=column_mapping["left"], 
                right_col=column_mapping["right"]
            )),
            (f"{column_type}_both_missing", BothMissingFeaturizer(
                left_col=column_mapping["left"], 
                right_col=column_mapping["right"]
            )),
            (f"{column_type}_one_missing", OneMissingFeaturizer(
                left_col=column_mapping["left"], 
                right_col=column_mapping["right"]
            )),
        ])
    
    # Additional domain-specific featurizers
    medical_condition_featurizers = [
        ("medical_condition_exact_match", ExactNormalizedMatchFeaturizer(
            left_col=COLUMN_CONFIG["medical_condition"]["left"], 
            right_col=COLUMN_CONFIG["medical_condition"]["right"]
        )),
        ("medical_condition_jaro_winkler", JaroWinklerFeaturizer(
            left_col=COLUMN_CONFIG["medical_condition"]["left"], 
            right_col=COLUMN_CONFIG["medical_condition"]["right"]
        )),
        ("medical_condition_token_jaccard", TokenJaccardFeaturizer(
            left_col=COLUMN_CONFIG["medical_condition"]["left"], 
            right_col=COLUMN_CONFIG["medical_condition"]["right"], 
            stopwords=DEFAULT_STOPWORDS
        )),
        ("medical_condition_token_stats", TokenStatsFeaturizer(
            left_col=COLUMN_CONFIG["medical_condition"]["left"], 
            right_col=COLUMN_CONFIG["medical_condition"]["right"], 
            stopwords=DEFAULT_STOPWORDS
        )),
        ("medical_condition_char_ngram", CharNgramJaccardFeaturizer(
            left_col=COLUMN_CONFIG["medical_condition"]["left"], 
            right_col=COLUMN_CONFIG["medical_condition"]["right"]
        )),
    ]
    
    geographical_area_featurizers = [
        ("geographical_area_exact_match", ExactNormalizedMatchFeaturizer(
            left_col=COLUMN_CONFIG["geographical_area"]["left"], 
            right_col=COLUMN_CONFIG["geographical_area"]["right"]
        )),
        ("geographical_area_jaro_winkler", JaroWinklerFeaturizer(
            left_col=COLUMN_CONFIG["geographical_area"]["left"], 
            right_col=COLUMN_CONFIG["geographical_area"]["right"]
        )),
        ("geographical_area_token_jaccard", TokenJaccardFeaturizer(
            left_col=COLUMN_CONFIG["geographical_area"]["left"], 
            right_col=COLUMN_CONFIG["geographical_area"]["right"], 
            stopwords=DEFAULT_STOPWORDS
        )),
        ("geographical_area_token_stats", TokenStatsFeaturizer(
            left_col=COLUMN_CONFIG["geographical_area"]["left"], 
            right_col=COLUMN_CONFIG["geographical_area"]["right"], 
            stopwords=DEFAULT_STOPWORDS
        )),
        ("geographical_area_char_ngram", CharNgramJaccardFeaturizer(
            left_col=COLUMN_CONFIG["geographical_area"]["left"], 
            right_col=COLUMN_CONFIG["geographical_area"]["right"]
        )),
    ]
    
    # Combine all featurizers
    all_featurizers = (
        original_name_featurizers + 
        normalized_name_featurizers + 
        acronym_featurizers + 
        semantic_featurizers + 
        missing_pattern_featurizers + 
        medical_condition_featurizers + 
        geographical_area_featurizers
    )
    
    return FeatureUnion(all_featurizers, n_jobs=1)


def validate_dataset_columns(dataset: pd.DataFrame, dataset_name: str) -> None:
    """
    Validate that required columns are present in the dataset.
    
    Args:
        dataset: DataFrame to validate
        dataset_name: Name for error messages
        
    Raises:
        click.ClickException: If required columns are missing
    """
    required_columns = [
        COLUMN_CONFIG["name"]["left"],
        COLUMN_CONFIG["name"]["right"],
        COLUMN_CONFIG["medical_condition"]["left"],
        COLUMN_CONFIG["medical_condition"]["right"],
        COLUMN_CONFIG["geographical_area"]["left"],
        COLUMN_CONFIG["geographical_area"]["right"],
        COLUMN_CONFIG["acronym"]["left"],
        COLUMN_CONFIG["acronym"]["right"],
        COLUMN_CONFIG["embedding"]["left"],
        COLUMN_CONFIG["embedding"]["right"],
        COL_LABEL
    ]

    missing_columns = [col for col in required_columns if col not in dataset.columns]
    if missing_columns:
        raise click.ClickException(f"Dataset '{dataset_name}' missing required columns: {missing_columns}")


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--train_parquet",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to training dataset parquet file"
)
@click.option(
    "--test_parquet",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to test dataset parquet file"
)
@click.option(
    "--label_col", 
    default=COL_LABEL, 
    show_default=True,
    help="Name of the label column in the datasets"
)
@click.option(
    "--output_dir", 
    type=click.Path(file_okay=False, path_type=Path), 
    required=True,
    help="Directory to save feature matrices and artifacts"
)
def main(train_parquet: Path, test_parquet: Path, label_col: str, output_dir: Path):
    """
    Extract comprehensive similarity features for registry deduplication.
    
    This script processes paired registry data to create feature matrices suitable
    for training machine learning models. It applies text normalization and extracts
    multiple types of similarity features including:
    
    - Exact match indicators
    - String similarity measures (Jaro-Winkler, Levenshtein, LCS)
    - Token-based similarities (Jaccard, TF-IDF)
    - Semantic similarities using embeddings
    - Acronym-specific features
    - Medical condition and geographical area features
    - Missing value pattern indicators
    
    Example usage:
        python S02_prepare_features.py \\
            --train_parquet data/train_dataset.parquet \\
            --test_parquet data/test_dataset.parquet \\
            --label_col label \\
            --output_dir data/features/
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets and set pair_id as index
    click.echo("[INFO] Loading training and test datasets...")
    training_dataset = pd.read_parquet(train_parquet).set_index(COL_PAIR_ID)
    test_dataset = pd.read_parquet(test_parquet).set_index(COL_PAIR_ID)
    
    click.echo(f"[INFO] Training dataset: {len(training_dataset):,} pairs")
    click.echo(f"[INFO] Test dataset: {len(test_dataset):,} pairs")

    # Validate required columns
    validate_dataset_columns(training_dataset, "training dataset")
    validate_dataset_columns(test_dataset, "test dataset")

    # Extract labels
    training_labels = training_dataset[label_col].astype(int).values
    test_labels = test_dataset[label_col].astype(int).values

    # Configure and apply text normalization
    click.echo("[INFO] Applying text normalization to registry names...")
    text_normalizer = configure_registry_name_text_normalizer()
    
    training_dataset_normalized = text_normalizer.fit(training_dataset).transform(training_dataset)
    test_dataset_normalized = text_normalizer.transform(test_dataset)

    # Build and fit feature extraction pipeline
    click.echo("[INFO] Building comprehensive feature extraction pipeline...")
    feature_extraction_pipeline = build_comprehensive_feature_pipeline()
    
    click.echo("[INFO] Fitting feature pipeline on training data...")
    feature_extraction_pipeline.fit(training_dataset_normalized, training_labels)
    
    # Extract features
    click.echo("[INFO] Extracting features from training dataset...")
    training_feature_matrix = feature_extraction_pipeline.transform(training_dataset_normalized)
    
    click.echo("[INFO] Extracting features from test dataset...")
    test_feature_matrix = feature_extraction_pipeline.transform(test_dataset_normalized)
    
    extracted_feature_names = feature_extraction_pipeline.get_feature_names_out().tolist()
    
    click.echo(f"[INFO] Extracted {len(extracted_feature_names)} features total")

    # Save feature matrices and labels
    np.save(output_dir / "X_train.npy", training_feature_matrix)
    np.save(output_dir / "X_test.npy", test_feature_matrix)
    np.save(output_dir / "y_train.npy", training_labels)
    np.save(output_dir / "y_test.npy", test_labels)

    # Save feature metadata
    feature_metadata = {
        "feature_extraction": {
            "n_training_samples": int(training_feature_matrix.shape[0]),
            "n_test_samples": int(test_feature_matrix.shape[0]),
            "n_features_extracted": len(extracted_feature_names),
            "feature_names": extracted_feature_names,
        },
        "label_distributions": {
            "training": {str(k): int(v) for k, v in zip(*np.unique(training_labels, return_counts=True))},
            "test": {str(k): int(v) for k, v in zip(*np.unique(test_labels, return_counts=True))},
        },
        "text_normalization": {
            "normalized_suffix": text_normalizer.normalized_suffix,
            "remove_drop_terms": text_normalizer.remove_drop_terms,
            "remove_stopwords": text_normalizer.remove_stopwords,
            "sort_tokens": text_normalizer.sort_tokens,
        }
    }

    (output_dir / "feature_names.json").write_text(json.dumps(extracted_feature_names, indent=2))
    (output_dir / "metadata.json").write_text(json.dumps(feature_metadata, indent=2))

    # Save fitted pipeline components for inference
    with open(output_dir / "feature_pipeline.pkl", "wb") as pipeline_file:
        pickle.dump(feature_extraction_pipeline, pipeline_file)
    with open(output_dir / "text_normalizer.pkl", "wb") as normalizer_file:
        pickle.dump(text_normalizer, normalizer_file)

    # Create comprehensive feature dataframes for analysis
    click.echo("[INFO] Creating comprehensive feature dataframes...")
    training_features_dataframe = create_comprehensive_feature_dataframe(
        training_feature_matrix, extracted_feature_names, training_dataset_normalized, 
        label_col, training_labels, text_normalizer
    )
    test_features_dataframe = create_comprehensive_feature_dataframe(
        test_feature_matrix, extracted_feature_names, test_dataset_normalized, 
        label_col, test_labels, text_normalizer
    )

    training_features_dataframe.to_csv(output_dir / "train_pairs_with_features.csv", index=False)
    test_features_dataframe.to_csv(output_dir / "test_pairs_with_features.csv", index=False)

    # Generate feature distribution plots
    click.echo("[INFO] Creating feature distribution plots...")
    create_feature_distribution_plots(
        training_feature_matrix, test_feature_matrix, extracted_feature_names, output_dir
    )

    click.echo(f"[SUCCESS] Feature extraction completed successfully!")
    click.echo(f"[SUCCESS] Feature matrices saved to: {output_dir}")
    click.echo(f"[SUCCESS] {len(extracted_feature_names)} features extracted")

if __name__ == "__main__":
    main()

