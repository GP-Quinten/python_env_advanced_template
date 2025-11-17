#!/usr/bin/env python3
"""
Production Pipeline Assembly for Registry Deduplication

This script combines the trained components (text normalizer, feature extractor, and classifier)
into a single sklearn Pipeline for production inference. It creates a seamless end-to-end
pipeline that can process raw registry pairs and output deduplication predictions.

Input Requirements:
    - feature_pipeline.pkl: Fitted PairFeatureUnion from feature extraction
    - trained_model.joblib: Trained HistGradientBoostingClassifier
    - text_normalizer.pkl: Fitted RegistryNameNormalizer
    
Output:
    - inference_pipeline.joblib: Complete sklearn Pipeline for production use
    - pipeline_manifest.json: Detailed metadata about the pipeline components
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from datetime import datetime

import click
import joblib
import numpy as np
import pandas as pd
import sklearn

from deduplication_model.normalization.text_normalizer import RegistryNameNormalizer
from deduplication_model.config import COLUMN_CONFIG
from sklearn.pipeline import Pipeline


def create_pipeline_test_data(text_normalizer: RegistryNameNormalizer) -> pd.DataFrame:
    """
    Create synthetic test data to validate the assembled pipeline.
    
    Args:
        text_normalizer: Fitted text normalizer to get expected column names
        
    Returns:
        DataFrame with sample registry pairs for testing
    """
    return pd.DataFrame({
        # Primary name columns (required by normalizer)
        text_normalizer.left_col: ["European Cancer Registry Network", "National Heart Institute"],
        text_normalizer.right_col: ["European Cancer Registry", "National Heart Foundation"],
        
        # Acronym columns
        COLUMN_CONFIG["acronym"]["left"]: ["ECRN", "NHI"],
        COLUMN_CONFIG["acronym"]["right"]: ["ECR", "NHF"],
        
        # Embedding columns (semantic similarity features)
        COLUMN_CONFIG["embedding"]["left"]: [
            [0.1, 0.2, 0.3, 0.4, 0.5], 
            [0.6, 0.7, 0.8, 0.9, 1.0]
        ],
        COLUMN_CONFIG["embedding"]["right"]: [
            [0.11, 0.21, 0.31, 0.41, 0.51], 
            [0.61, 0.71, 0.81, 0.91, 1.01]
        ],
        
        # Medical condition columns
        COLUMN_CONFIG["medical_condition"]["left"]: [
            "Breast Cancer Lung Cancer Colorectal Cancer", 
            "Cardiovascular Disease Coronary Artery Disease"
        ],
        COLUMN_CONFIG["medical_condition"]["right"]: [
            "Breast Cancer Lung Cancer", 
            "Heart Disease Cardiac Conditions"
        ],
        
        # Geographical area columns
        COLUMN_CONFIG["geographical_area"]["left"]: [
            "Europe France Germany Italy Spain", 
            "North America United States Canada"
        ],
        COLUMN_CONFIG["geographical_area"]["right"]: [
            "Europe France Germany", 
            "United States Canada Mexico"
        ],
    })


def extract_pipeline_metadata(
    text_normalizer: RegistryNameNormalizer,
    feature_extractor: object,
    trained_classifier: object
) -> dict:
    """
    Extract comprehensive metadata about all pipeline components.
    
    Args:
        text_normalizer: Fitted text normalizer
        feature_extractor: Fitted feature extraction pipeline
        trained_classifier: Trained classification model
        
    Returns:
        Dictionary with detailed component metadata
    """
    # Extract feature names safely
    try:
        extracted_feature_names = feature_extractor.get_feature_names_out().tolist()
    except Exception:
        extracted_feature_names = []

    return {
        "pipeline_creation": {
            "created_timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "sklearn_version": sklearn.__version__,
        },
        "input_requirements": {
            "expected_format": "pandas.DataFrame",
            "primary_name_columns": {
                "left_column": text_normalizer.left_col,
                "right_column": text_normalizer.right_col,
            },
            "required_columns": [
                text_normalizer.left_col,
                text_normalizer.right_col,
                COLUMN_CONFIG["acronym"]["left"],
                COLUMN_CONFIG["acronym"]["right"],
                COLUMN_CONFIG["embedding"]["left"],
                COLUMN_CONFIG["embedding"]["right"],
                COLUMN_CONFIG["medical_condition"]["left"],
                COLUMN_CONFIG["medical_condition"]["right"],
                COLUMN_CONFIG["geographical_area"]["left"],
                COLUMN_CONFIG["geographical_area"]["right"],
            ],
        },
        "component_details": {
            "text_normalizer": {
                "class_name": type(text_normalizer).__name__,
                "configuration": {
                    "remove_drop_terms": text_normalizer.remove_drop_terms,
                    "remove_stopwords": text_normalizer.remove_stopwords,
                    "sort_tokens": text_normalizer.sort_tokens,
                    "output_mode": text_normalizer.output_mode,
                    "normalized_suffix": text_normalizer.normalized_suffix,
                },
                "preprocessing_rules": {
                    "drop_terms": list(text_normalizer.drop_terms or []),
                    "stopwords": list(text_normalizer.stopwords or []),
                },
            },
            "feature_extractor": {
                "class_name": type(feature_extractor).__name__,
                "extracted_features": {
                    "total_features": len(extracted_feature_names),
                    "feature_names": extracted_feature_names,
                },
            },
            "classifier": {
                "class_name": type(trained_classifier).__name__,
                "model_type": "Gradient Boosting Classifier",
                "supports_probability_output": hasattr(trained_classifier, "predict_proba"),
            },
        },
        "output_format": {
            "prediction_method": "predict",
            "probability_method": "predict_proba" if hasattr(trained_classifier, "predict_proba") else None,
            "output_classes": [0, 1],  # Binary classification
            "interpretation": {
                "0": "No match (different registries)",
                "1": "Match (same registry)",
            },
        },
    }


def perform_pipeline_validation_test(
    assembled_pipeline: Pipeline,
    test_data: pd.DataFrame
) -> dict:
    """
    Perform comprehensive validation testing of the assembled pipeline.
    
    Args:
        assembled_pipeline: Complete inference pipeline
        test_data: Sample data for testing
        
    Returns:
        Dictionary with validation results
        
    Raises:
        Exception: If pipeline validation fails
    """
    validation_results = {"status": "failed", "errors": []}
    
    try:
        # Test basic prediction functionality
        binary_predictions = assembled_pipeline.predict(test_data)
        validation_results["binary_predictions"] = binary_predictions.tolist()
        
        # Test probability prediction if available
        if hasattr(assembled_pipeline, "predict_proba"):
            probability_predictions = assembled_pipeline.predict_proba(test_data)
            validation_results["probability_predictions"] = probability_predictions.tolist()
        
        # Validate prediction output format
        if not isinstance(binary_predictions, np.ndarray):
            validation_results["errors"].append("Predictions are not numpy array")
        
        if binary_predictions.dtype not in [np.int32, np.int64, int]:
            validation_results["errors"].append(f"Unexpected prediction dtype: {binary_predictions.dtype}")
        
        if not all(pred in [0, 1] for pred in binary_predictions):
            validation_results["errors"].append("Predictions contain values other than 0 or 1")
        
        # Mark as successful if no errors
        if not validation_results["errors"]:
            validation_results["status"] = "success"
            
    except Exception as validation_error:
        validation_results["errors"].append(f"Pipeline execution failed: {str(validation_error)}")
        
    return validation_results


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--feature_union",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the fitted PairFeatureUnion pickle file"
)
@click.option(
    "--trained_model",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the trained HistGradientBoostingClassifier joblib file"
)
@click.option(
    "--normalizer",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the fitted RegistryNameNormalizer pickle file"
)
@click.option(
    "--output_dir", 
    type=click.Path(file_okay=False, path_type=Path), 
    required=True,
    help="Directory to save the assembled pipeline and metadata"
)
def main(
    feature_union: Path,
    trained_model: Path,
    normalizer: Path,
    output_dir: Path,
):
    """
    Assemble a complete production inference pipeline for registry deduplication.
    
    This script combines three trained components into a single sklearn Pipeline:
    1. Text normalizer: Preprocesses registry names
    2. Feature extractor: Computes similarity features between pairs
    3. Classifier: Makes deduplication predictions
    
    The resulting pipeline can process raw registry pairs and output predictions
    in a single step, making it suitable for production deployment.
    
    Example usage:
        python S04_make_pipeline.py \\
            --feature_union data/features/feature_pipeline.pkl \\
            --trained_model data/model/trained_model.joblib \\
            --normalizer data/features/text_normalizer.pkl \\
            --output_dir data/production_pipeline/
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the three trained components
    click.echo("[INFO] Loading trained pipeline components...")
    
    with open(feature_union, "rb") as feature_file:
        fitted_feature_extractor = pickle.load(feature_file)
    click.echo(f"[INFO] Loaded feature extractor from: {feature_union}")

    trained_classifier = joblib.load(trained_model)
    click.echo(f"[INFO] Loaded trained classifier from: {trained_model}")

    with open(normalizer, "rb") as normalizer_file:
        fitted_text_normalizer = pickle.load(normalizer_file)
    click.echo(f"[INFO] Loaded text normalizer from: {normalizer}")

    # Assemble the complete inference pipeline
    click.echo("[INFO] Assembling complete inference pipeline...")
    complete_inference_pipeline = Pipeline([
        ("text_normalization", fitted_text_normalizer),
        ("feature_extraction", fitted_feature_extractor),
        ("classification", trained_classifier),
    ])

    # Save the assembled pipeline
    pipeline_output_path = output_dir / "inference_pipeline.joblib"
    joblib.dump(complete_inference_pipeline, pipeline_output_path)
    click.echo(f"[SUCCESS] Saved complete pipeline to: {pipeline_output_path}")

    # Extract and save comprehensive metadata
    click.echo("[INFO] Generating pipeline metadata...")
    pipeline_metadata = extract_pipeline_metadata(
        fitted_text_normalizer, fitted_feature_extractor, trained_classifier
    )
    
    # Add file path information
    pipeline_metadata["artifact_paths"] = {
        "inference_pipeline": str(pipeline_output_path),
        "source_components": {
            "feature_extractor": str(feature_union),
            "trained_classifier": str(trained_model),
            "text_normalizer": str(normalizer),
        }
    }

    metadata_output_path = output_dir / "pipeline_manifest.json"
    metadata_output_path.write_text(json.dumps(pipeline_metadata, indent=2))
    click.echo(f"[SUCCESS] Saved pipeline metadata to: {metadata_output_path}")

    # Perform validation testing with synthetic data
    click.echo("[INFO] Performing pipeline validation test...")
    test_data = create_pipeline_test_data(fitted_text_normalizer)
    validation_results = perform_pipeline_validation_test(complete_inference_pipeline, test_data)

    if validation_results["status"] == "success":
        click.echo("[SUCCESS] Pipeline validation test passed!")
        if "binary_predictions" in validation_results:
            sample_predictions = validation_results["binary_predictions"]
            click.echo(f"[TEST] Sample predictions on synthetic data: {sample_predictions}")
    else:
        click.echo("[ERROR] Pipeline validation test failed!")
        for error in validation_results["errors"]:
            click.echo(f"[ERROR] {error}")
        
        # Still save the pipeline but warn the user
        click.echo("[WARNING] Pipeline saved despite validation errors - manual testing recommended")

    # Add validation results to metadata
    pipeline_metadata["validation_test"] = validation_results
    metadata_output_path.write_text(json.dumps(pipeline_metadata, indent=2))

    click.echo(f"\n[SUCCESS] Pipeline assembly completed!")
    click.echo(f"[SUCCESS] Production pipeline ready at: {pipeline_output_path}")
    click.echo(f"[SUCCESS] Pipeline documentation at: {metadata_output_path}")

if __name__ == "__main__":
    main()
