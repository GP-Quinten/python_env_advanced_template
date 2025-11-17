#!/usr/bin/env python3
# Run inference using a fitted sklearn pipeline
# Dataset is always JSON, no labels needed

from __future__ import annotations
from pathlib import Path
import click
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
COLUMN_CONFIG = {
    "name": {"left": "left_name", "right": "right_name"},
    "medical_condition": {"left": "left_medical_conditions", "right": "right_medical_conditions"},
    "geographical_area": {"left": "left_geographical_areas", "right": "right_geographical_areas"},
    "acronym": {"left": "left_acronym", "right": "right_acronym"},
    "embedding": {"left": "left_embedding", "right": "right_embedding"}
}

def _clean_additional_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an 's' suffix to medical_condition and geographical_area columns if not present.
    """
    for col_type in ["medical_condition", "geographical_area"]:
        for side in ["left", "right"]:
            # if the singular column exists but not the plural one, rename it
            singular_col = COLUMN_CONFIG[col_type][side].replace("s", "")
            plural_col = COLUMN_CONFIG[col_type][side]
            if singular_col in df.columns and plural_col not in df.columns:
                click.echo(f"[INFO] Renaming column {singular_col} to {plural_col}.")
                df = df.rename(columns={singular_col: plural_col})
            # if neither exists, create an empty plural column
            if singular_col not in df.columns and plural_col not in df.columns:
                click.echo(f"[INFO] Creating empty column {plural_col}.")
                df[plural_col] = ""
    return df


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--pipeline",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to inference pipeline joblib (from R12_make_pipeline).",
)
@click.option(
    "--dataset_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to dataset JSON with left/right columns.",
)
@click.option(
    "--output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory to save predictions CSV.",
)
def main(pipeline: Path, dataset_json: Path, output_dir: Path):
    # hard-coded column names
    id_col = "pair_hash_id"
    left_col = "left_name"
    right_col = "right_name"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load pipeline
    pipe = joblib.load(pipeline)

    # --- Read dataset (always JSON)
    df = pd.read_json(dataset_json)
    # clean column names if needed
    df = _clean_additional_column_names(df)
    # show columns
    click.echo(f"[INFO] Dataset columns: {df.columns.tolist()}")

    # --- Ensure required columns
    for col in (id_col, left_col, right_col):
        if col not in df.columns:
            raise click.ClickException(f"Missing column: {col}")

    # --- Run inference
    try:
        scores = pipe.predict_proba(df)[:, 1]
    except Exception as e:
        click.echo(f"[ERROR] Inference failed: {e}")
        # Try to get more detailed error information
        if hasattr(pipe, "steps"):
            for step_name, step in pipe.steps:
                click.echo(f"[DEBUG] Pipeline step: {step_name}, type: {type(step)}")
                if step_name == "features" and hasattr(step, "featurizers"):
                    for name, featurizer in step.featurizers:
                        if hasattr(featurizer, "left_col") and hasattr(
                            featurizer, "right_col"
                        ):
                            l_col = featurizer.left_col
                            r_col = featurizer.right_col
                            click.echo(
                                f"[DEBUG] Featurizer {name} requires: {l_col}, {r_col}"
                            )
                            if l_col not in df.columns:
                                click.echo(f"[DEBUG] Missing left column: {l_col}")
                            if r_col not in df.columns:
                                click.echo(f"[DEBUG] Missing right column: {r_col}")
        raise

    # --- Plot score distribution
    plt.figure()
    plt.hist(scores, bins=50, edgecolor="k")
    plt.title("Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    dist_path = output_dir / "score_distribution.png"
    plt.savefig(dist_path)
    plt.close()
    click.echo(f"[SUCCESS] Saved score distribution plot → {dist_path}")

    # --- Save predictions
    out_df = df.loc[:, [id_col, left_col, right_col]].copy()
    out_df["score"] = scores
    preds_path = output_dir / "predictions.csv"
    out_df.to_csv(preds_path, index=False)

    click.echo(f"[SUCCESS] Saved predictions → {preds_path}")



if __name__ == "__main__":
    main()
