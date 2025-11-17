#!/usr/bin/env python3
# File: src/scripts/S11_prepare_features.py

import json
import pickle
from pathlib import Path

import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from deduplication_model.normalization.text_normalizer import RegistryNameNormalizer
from deduplication_model.features.aggregator import PairFeatureUnion
from deduplication_model.features.jw_featurizer import JaroWinklerFeaturizer
from deduplication_model.features.levenshtein_featurizer import LevenshteinNormSimFeaturizer
from deduplication_model.features.lcs_featurizer import LCSNormalizedFeaturizer
from deduplication_model.features.jaccard_featurizer import TokenJaccardFeaturizer
from deduplication_model.features.tfidf_token_cosine_featurizer import (
    TfidfTokenCosineFeaturizer,
)
from deduplication_model.features.tfidf_char_cosine_featurizer import (
    TfidfCharCosineFeaturizer,
)
from deduplication_model.features.token_stats_featurizer import TokenStatsFeaturizer
from deduplication_model.features.shared_rare_token_frac_featurizer import (
    SharedRareTokenFracFeaturizer,
)
from deduplication_model.features.char_ngram_jaccard_featurizer import (
    CharNgramJaccardFeaturizer,
)
from deduplication_model.features.exact_match_featurizer import (
    ExactNormalizedMatchFeaturizer,
)
from deduplication_model.features.token_bag_match_featurizer import TokenBagMatchFeaturizer
from deduplication_model.normalization.rules import DEFAULT_STOPWORDS

from deduplication_model.features.acronym_featurizer import (
    AcronymExactMatchFeaturizer,
    AcronymLengthDiffFeaturizer,
    AcronymJaccardFeaturizer,
)
from deduplication_model.features.soft_tfidf_featurizer import SoftTfidfFeaturizer
from deduplication_model.features.semantic_featurizer import SemanticCosineFeaturizer
from deduplication_model.features.missing_pattern_featurizer import (
    BothPresentFeaturizer,
    BothMissingFeaturizer,
    OneMissingFeaturizer,
)

# Configuration
COLUMN_CONFIG = {
    "name": {"left": "left_name", "right": "right_name"},
    "medical_condition": {"left": "left_medical_conditions", "right": "right_medical_conditions"},
    "geographical_area": {"left": "left_geographical_areas", "right": "right_geographical_areas"},
    "acronym": {"left": "left_acronym", "right": "right_acronym"},
    "embedding": {"left": "left_embedding", "right": "right_embedding"}
}

def _read_json_table(path: Path) -> pd.DataFrame:
    return pd.read_json(Path(path))

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

def _prepare_feature_dataframe(X, feature_names, df_norm, label_col, y, normalizer):
    """Helper function to prepare feature dataframe with all columns."""
    feat_df = pd.DataFrame(X, columns=feature_names)
    feat_df[label_col] = y
    
    # Add all columns from configuration
    for col_type, cols in COLUMN_CONFIG.items():
        for side in ['left', 'right']:
            col_name = cols[side]
            if col_name in df_norm.columns:
                feat_df[col_name] = df_norm[col_name].astype(str).to_numpy()
    
    # Handle normalized name columns
    norm_cols = {
        f"{COLUMN_CONFIG['name']['left']}{normalizer.normalized_suffix}": df_norm[f"{COLUMN_CONFIG['name']['left']}{normalizer.normalized_suffix}"].astype(str).to_numpy(),
        f"{COLUMN_CONFIG['name']['right']}{normalizer.normalized_suffix}": df_norm[f"{COLUMN_CONFIG['name']['right']}{normalizer.normalized_suffix}"].astype(str).to_numpy()
    }
    feat_df.update(norm_cols)
    
    return feat_df.replace("", None)

def _plot_feature_distributions(X_train, X_test, feature_names, output_dir: Path):
    plots_dir = output_dir / "feature_distributions"
    plots_dir.mkdir(exist_ok=True, parents=True)
    for i, fname in enumerate(feature_names):
        plt.figure(figsize=(6, 4))
        
        # Extract feature values and handle NaNs
        train_values = X_train[:, i]
        test_values = X_test[:, i]
        
        # Calculate NaN percentages
        train_nan_pct = np.sum(np.isnan(train_values)) / len(train_values) * 100
        test_nan_pct = np.sum(np.isnan(test_values)) / len(test_values) * 100
        
        # Filter out NaNs for plotting
        train_values_clean = train_values[~np.isnan(train_values)]
        test_values_clean = test_values[~np.isnan(test_values)]
        
        # Plot histograms only if we have data after removing NaNs
        if len(train_values_clean) > 0:
            plt.hist(train_values_clean, bins=30, alpha=0.6, label="train", density=True)
        if len(test_values_clean) > 0:
            plt.hist(test_values_clean, bins=30, alpha=0.6, label="test", density=True)
        
        plt.title(f"Distribution of {fname}")
        plt.xlabel(fname)
        plt.ylabel("Density")
        plt.legend()
        
        # Add text showing missing data percentages
        missing_text = f"Missing: Train {train_nan_pct:.1f}%, Test {test_nan_pct:.1f}%"
        plt.figtext(0.5, 0.02, missing_text, ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.savefig(plots_dir / f"{fname}.png", dpi=150)
        plt.close()

@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--train_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--test_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--label_col", default="label", show_default=True)
@click.option(
    "--output_dir", type=click.Path(file_okay=False, path_type=Path), required=True
)
def main(train_json, test_json, label_col, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df_tr = _read_json_table(train_json).set_index("pair_hash_id")
    df_te = _read_json_table(test_json).set_index("pair_hash_id")

    df_tr = _clean_additional_column_names(df_tr)
    df_te = _clean_additional_column_names(df_te)

    # Required columns check
    required_cols = [
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
        label_col
    ]

    for df, name in [(df_tr, "train"), (df_te, "test")]:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise click.ClickException(f"{name} file missing columns: {missing}")

    y_train = df_tr[label_col].astype(int).values
    y_test = df_te[label_col].astype(int).values

    # Text normalization
    normalizer = RegistryNameNormalizer(
        left_col=COLUMN_CONFIG["name"]["left"],
        right_col=COLUMN_CONFIG["name"]["right"],
        output_mode="append",
        remove_drop_terms=True,
        remove_stopwords=True,
        sort_tokens=True,
    )

    df_tr_norm = normalizer.fit(df_tr).transform(df_tr)
    df_te_norm = normalizer.transform(df_te)

    norm_left = f"{COLUMN_CONFIG['name']['left']}{normalizer.normalized_suffix}"
    norm_right = f"{COLUMN_CONFIG['name']['right']}{normalizer.normalized_suffix}"

    # Prepare featurizers
    featurizers = [
        ("raw_exact", ExactNormalizedMatchFeaturizer(left_col=COLUMN_CONFIG["name"]["left"], right_col=COLUMN_CONFIG["name"]["right"])),
        ("norm_exact", ExactNormalizedMatchFeaturizer(left_col=norm_left, right_col=norm_right)),
        ("raw_tok_bag", TokenBagMatchFeaturizer(left_col=COLUMN_CONFIG["name"]["left"], right_col=COLUMN_CONFIG["name"]["right"])),
        ("norm_tok_bag", TokenBagMatchFeaturizer(left_col=norm_left, right_col=norm_right)),
        ("raw_jw", JaroWinklerFeaturizer(left_col=COLUMN_CONFIG["name"]["left"], right_col=COLUMN_CONFIG["name"]["right"])),
        ("norm_jw", JaroWinklerFeaturizer(left_col=norm_left, right_col=norm_right)),
        ("raw_lev", LevenshteinNormSimFeaturizer(left_col=COLUMN_CONFIG["name"]["left"], right_col=COLUMN_CONFIG["name"]["right"])),
        ("norm_lev", LevenshteinNormSimFeaturizer(left_col=norm_left, right_col=norm_right)),
        ("raw_jacc", TokenJaccardFeaturizer(left_col=COLUMN_CONFIG["name"]["left"], right_col=COLUMN_CONFIG["name"]["right"], stopwords=DEFAULT_STOPWORDS)),
        ("norm_jacc", TokenJaccardFeaturizer(left_col=norm_left, right_col=norm_right, stopwords=DEFAULT_STOPWORDS)),
        ("raw_tfidf_tok", TfidfTokenCosineFeaturizer(left_col=COLUMN_CONFIG["name"]["left"], right_col=COLUMN_CONFIG["name"]["right"])),
        ("norm_tfidf_tok", TfidfTokenCosineFeaturizer(left_col=norm_left, right_col=norm_right)),
        ("raw_tfidf_char", TfidfCharCosineFeaturizer(left_col=COLUMN_CONFIG["name"]["left"], right_col=COLUMN_CONFIG["name"]["right"])),
        ("norm_tfidf_char", TfidfCharCosineFeaturizer(left_col=norm_left, right_col=norm_right)),
        ("raw_soft_tfidf", SoftTfidfFeaturizer(left_col=COLUMN_CONFIG["name"]["left"], right_col=COLUMN_CONFIG["name"]["right"])),
        ("norm_soft_tfidf", SoftTfidfFeaturizer(left_col=norm_left, right_col=norm_right)),
        ("raw_tok_stats", TokenStatsFeaturizer(left_col=COLUMN_CONFIG["name"]["left"], right_col=COLUMN_CONFIG["name"]["right"], stopwords=DEFAULT_STOPWORDS)),
        ("norm_tok_stats", TokenStatsFeaturizer(left_col=norm_left, right_col=norm_right, stopwords=DEFAULT_STOPWORDS)),
        ("raw_lcs", LCSNormalizedFeaturizer(left_col=COLUMN_CONFIG["name"]["left"], right_col=COLUMN_CONFIG["name"]["right"])),
        ("norm_lcs", LCSNormalizedFeaturizer(left_col=norm_left, right_col=norm_right)),
        ("raw_rare_tok", SharedRareTokenFracFeaturizer(left_col=COLUMN_CONFIG["name"]["left"], right_col=COLUMN_CONFIG["name"]["right"], stopwords=DEFAULT_STOPWORDS)),
        ("norm_rare_tok", SharedRareTokenFracFeaturizer(left_col=norm_left, right_col=norm_right, stopwords=DEFAULT_STOPWORDS)),
        ("raw_char_ngram", CharNgramJaccardFeaturizer(left_col=COLUMN_CONFIG["name"]["left"], right_col=COLUMN_CONFIG["name"]["right"])),
        ("norm_char_ngram", CharNgramJaccardFeaturizer(left_col=norm_left, right_col=norm_right)),
        ("acr_exact", AcronymExactMatchFeaturizer(left_col=COLUMN_CONFIG["acronym"]["left"], right_col=COLUMN_CONFIG["acronym"]["right"])),
        ("acr_len_diff", AcronymLengthDiffFeaturizer(left_col=COLUMN_CONFIG["acronym"]["left"], right_col=COLUMN_CONFIG["acronym"]["right"])),
        ("acr_jacc", AcronymJaccardFeaturizer(left_col=COLUMN_CONFIG["acronym"]["left"], right_col=COLUMN_CONFIG["acronym"]["right"])),
        # Semantic featurizers
        ("raw_name_cosine_mistral_embed", SemanticCosineFeaturizer(left_col=COLUMN_CONFIG["embedding"]["left"], right_col=COLUMN_CONFIG["embedding"]["right"])),
        # Acronym missing pattern features
        ("acr_both_present", BothPresentFeaturizer(left_col=COLUMN_CONFIG["acronym"]["left"], right_col=COLUMN_CONFIG["acronym"]["right"])),
        ("acr_both_missing", BothMissingFeaturizer(left_col=COLUMN_CONFIG["acronym"]["left"], right_col=COLUMN_CONFIG["acronym"]["right"])),
        ("acr_one_missing", OneMissingFeaturizer(left_col=COLUMN_CONFIG["acronym"]["left"], right_col=COLUMN_CONFIG["acronym"]["right"])),
        # Geographical area missing pattern features
        ("geo_both_present", BothPresentFeaturizer(left_col=COLUMN_CONFIG["geographical_area"]["left"], right_col=COLUMN_CONFIG["geographical_area"]["right"])),
        ("geo_both_missing", BothMissingFeaturizer(left_col=COLUMN_CONFIG["geographical_area"]["left"], right_col=COLUMN_CONFIG["geographical_area"]["right"])),
        ("geo_one_missing", OneMissingFeaturizer(left_col=COLUMN_CONFIG["geographical_area"]["left"], right_col=COLUMN_CONFIG["geographical_area"]["right"])),
        # Medical condition missing pattern features
        ("med_both_present", BothPresentFeaturizer(left_col=COLUMN_CONFIG["medical_condition"]["left"], right_col=COLUMN_CONFIG["medical_condition"]["right"])),
        ("med_both_missing", BothMissingFeaturizer(left_col=COLUMN_CONFIG["medical_condition"]["left"], right_col=COLUMN_CONFIG["medical_condition"]["right"])),
        ("med_one_missing", OneMissingFeaturizer(left_col=COLUMN_CONFIG["medical_condition"]["left"], right_col=COLUMN_CONFIG["medical_condition"]["right"])),
    ]

    # Add medical condition and geographical area featurizers
    med_cond = COLUMN_CONFIG["medical_condition"]
    geo_area = COLUMN_CONFIG["geographical_area"]
    
    additional_featurizers = [
        ("medical_condition_exact", ExactNormalizedMatchFeaturizer(left_col=med_cond["left"], right_col=med_cond["right"])),
        ("medical_condition_jw", JaroWinklerFeaturizer(left_col=med_cond["left"], right_col=med_cond["right"])),
        ("medical_condition_jacc", TokenJaccardFeaturizer(left_col=med_cond["left"], right_col=med_cond["right"], stopwords=DEFAULT_STOPWORDS)),
        ("medical_condition_tok_stats", TokenStatsFeaturizer(left_col=med_cond["left"], right_col=med_cond["right"], stopwords=DEFAULT_STOPWORDS)),
        ("medical_condition_char_ngram", CharNgramJaccardFeaturizer(left_col=med_cond["left"], right_col=med_cond["right"])),
        ("geographical_area_exact", ExactNormalizedMatchFeaturizer(left_col=geo_area["left"], right_col=geo_area["right"])),
        ("geographical_area_jw", JaroWinklerFeaturizer(left_col=geo_area["left"], right_col=geo_area["right"])),
        ("geographical_area_jacc", TokenJaccardFeaturizer(left_col=geo_area["left"], right_col=geo_area["right"], stopwords=DEFAULT_STOPWORDS)),
        ("geographical_area_tok_stats", TokenStatsFeaturizer(left_col=geo_area["left"], right_col=geo_area["right"], stopwords=DEFAULT_STOPWORDS)),
        ("geographical_area_char_ngram", CharNgramJaccardFeaturizer(left_col=geo_area["left"], right_col=geo_area["right"]))
    ]
    
    featurizers.extend(additional_featurizers)

    # Feature processing
    feat_union = PairFeatureUnion(
        featurizers=featurizers,
        batch_size=1000,
    )

    click.echo("[INFO] Processing features")
    feat_union.fit(df_tr_norm, y_train)
    X_train = feat_union.transform(df_tr_norm)
    X_test = feat_union.transform(df_te_norm)
    feature_names = feat_union.get_feature_names_out().tolist()

    # Save results
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_test.npy", y_test)

    # Save metadata and models
    metadata = {
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "label_distribution_train": {str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
        "label_distribution_test": {str(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
    }

    (output_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2))
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    with open(output_dir / "feature_pipeline.pkl", "wb") as f:
        pickle.dump(feat_union, f)
    with open(output_dir / "text_normalizer.pkl", "wb") as f:
        pickle.dump(normalizer, f)

    # Create and save feature dataframes
    feat_df_tr = _prepare_feature_dataframe(X_train, feature_names, df_tr_norm, label_col, y_train, normalizer)
    feat_df_te = _prepare_feature_dataframe(X_test, feature_names, df_te_norm, label_col, y_test, normalizer)

    feat_df_tr.to_csv(output_dir / "train_pairs_with_features.csv", index=False)
    feat_df_te.to_csv(output_dir / "test_pairs_with_features.csv", index=False)

    _plot_feature_distributions(X_train, X_test, feature_names, output_dir)
    click.echo(f"[SUCCESS] Features and pipeline saved to: {output_dir}")

if __name__ == "__main__":
    main()
