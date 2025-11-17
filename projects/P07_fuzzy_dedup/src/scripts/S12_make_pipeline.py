#!/usr/bin/env python3
# File: src/scripts/S12_make_pipeline.py
#
# Build a single sklearn Pipeline for production inference:
#   [RegistryNameNormalizer] -> [PairFeatureUnion (fitted)] -> [Trained Model]
# Includes a simple sanity check with a fake sample.

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
from sklearn.pipeline import Pipeline


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--feature_union",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--trained_model",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--normalizer",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)  # Added normalizer option
@click.option(
    "--output_dir", type=click.Path(file_okay=False, path_type=Path), required=True
)
def main(
    feature_union: Path,
    trained_model: Path,
    normalizer: Path,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load fitted feature union (pickle), trained model (joblib), and normalizer (pickle)
    with open(feature_union, "rb") as f:
        features = pickle.load(f)

    model = joblib.load(trained_model)

    with open(normalizer, "rb") as f:
        normalizer_instance = pickle.load(f)

    # --- Build the full inference pipeline
    inference_pipeline = Pipeline(
        [ # -> input 8 cols
            ("normalize", normalizer_instance), # -> output : 16 cols
            ("features", features), # -> output : 30 cols
            ("model", model),  
        ]
    )

    # --- Save pipeline
    pipeline_path = Path(output_dir) / "inference_pipeline.joblib"
    joblib.dump(inference_pipeline, pipeline_path)

    # --- Manifest / metadata
    try:
        feat_names = features.get_feature_names_out().tolist()
    except Exception:
        feat_names = []

    manifest = {
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "sklearn_version": sklearn.__version__,
        "inputs": {
            "left_col": normalizer_instance.left_col,
            "right_col": normalizer_instance.right_col,
            "expects_dataframe": True,
        },
        "normalizer": {
            "class": type(normalizer_instance).__name__,
            "params": {
                "remove_drop_terms": normalizer_instance.remove_drop_terms,
                "remove_stopwords": normalizer_instance.remove_stopwords,
                "sort_tokens": normalizer_instance.sort_tokens,
                "drop_terms": list(normalizer_instance.drop_terms or []),
                "stopwords": list(normalizer_instance.stopwords or []),
                "output_mode": normalizer_instance.output_mode,
                "normalized_suffix": normalizer_instance.normalized_suffix,
            },
        },
        "features": {
            "class": type(features).__name__,
            "feature_names": feat_names,
        },
        "model": {
            "class": type(model).__name__,
        },
        "artifacts": {
            "pipeline_joblib": str(pipeline_path),
        },
    }

    manifest_path = Path(output_dir) / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    click.echo(f"[SUCCESS] Wrote inference pipeline → {pipeline_path}")
    click.echo(f"[SUCCESS] Wrote manifest → {manifest_path}")

    # --- Simple sanity check on fake data
    click.echo("[INFO] Running simple pipeline test with dummy input...")
    df_test = pd.DataFrame(
        {
            normalizer_instance.left_col: ["Foo Cancer Registry", "National Hospital"],
            normalizer_instance.right_col: ["Foo Registry", "Natl Hosp"],
            "left_acronym": ["FCR", "NH"],  # Added dummy acronym values
            "right_acronym": ["FR", "NH"],  # Added dummy acronym values
            "left_embedding": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],  # Dummy embeddings
            "right_embedding": [
                [0.11, 0.21, 0.31],
                [0.41, 0.51, 0.61],
            ],  # Dummy embeddings
            # Add dummy medical condition columns
            "left_medical_conditions": [
                "Hypertension Diabetes Obesity",
                "Cancer Heart Disease Stroke",
            ],
            "right_medical_conditions": ["Hypertension Diabetes", "Oncology Cardiology"],
            # Add dummy geographical area columns (european countries)
            "left_geographical_areas": ["France Germany Italy", "Spain Portugal Greece"],
            "right_geographical_areas": ["France Italy", "Spain Greece"],
        }
    )

    try:
        preds = inference_pipeline.predict(df_test)
        if hasattr(inference_pipeline, "predict_proba"):
            _ = inference_pipeline.predict_proba(df_test)
        click.echo(
            f"[TEST] Pipeline ran successfully. Sample predictions: {preds.tolist()}"
        )
    except Exception as e:
        click.echo(f"[TEST ERROR] Pipeline test failed: {e}")
        click.echo(
            f"[DEBUG] Available columns in test data: {df_test.columns.tolist()}"
        )
        raise

if __name__ == "__main__":
    main()
