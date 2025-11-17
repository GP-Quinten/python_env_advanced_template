#!/usr/bin/env python3
# File: src/scripts/S08_train_histgbm.py
#
# Train a Histogram-based Gradient Boosting classifier with a simple GridSearchCV on the TRAIN set.
# Inputs:  X_train.npy, y_train.npy (from R07_make_features)
# Outputs: best_model.joblib, best_params.json, cv_results.csv, training_summary.json
#
# NOTE: No evaluation on X_test/y_test here (by request).

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import click
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from tqdm_joblib import tqdm_joblib
from sklearn.model_selection import ParameterGrid
from sklearn.utils import check_random_state
import joblib
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance

def load_arrays(features_dir: Path):
    X = np.load(features_dir / "X_train.npy")
    y = np.load(features_dir / "y_train.npy")
    return X, y

def default_param_grid() -> Dict[str, Any]:
    """Parameter grid for HistGradientBoostingClassifier."""
    return {
        "max_iter": [100, 200, 300],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 6, 10, None],
        "min_samples_leaf": [10, 20, 30],
        "l2_regularization": [0.0, 0.1, 0.5],
        "max_bins": [128, 255],
        "early_stopping": [True],
        "validation_fraction": [0.1],
        "n_iter_no_change": [10],
    }

# Add a small grid for quick testing
def default_param_grid_test() -> Dict[str, Any]:
    """Small parameter grid for quick testing."""
    return {
        "max_iter": [50, 100],
        "learning_rate": [0.1],
        "max_depth": [6],
        "min_samples_leaf": [20],
        "l2_regularization": [0.0],
        "max_bins": [255],
        "early_stopping": [True],
        "validation_fraction": [0.1],
        "n_iter_no_change": [10],
    }

def plot_grid_search_results(grid, output_dir: Path):
    import numpy as _np
    # Use max_iter instead of n_estimators for HistGBM
    max_iters = _np.array([int(x) for x in grid.cv_results_['param_max_iter']])
    scores = grid.cv_results_['mean_test_score']
    unique = sorted(set(max_iters))
    avg = [_np.mean([scores[i] for i,v in enumerate(max_iters) if v==u]) for u in unique]
    plt.figure(); plt.plot(unique, avg, marker='o')
    plt.xlabel("max_iter"); plt.ylabel("Mean Test Score")
    plt.title("GridSearchCV: Test Score vs max_iter")
    plt.savefig(output_dir / "grid_search_plot.png"); plt.close()

def plot_hyperparameter_importance(grid, output_dir: Path):
    best = grid.best_params_
    names, vals = zip(*best.items())
    plt.figure(); plt.bar(names, [len(str(v)) for v in vals])  # dummy metric
    plt.xticks(rotation=45); plt.title("Hyperparameter Importance (stub)")
    plt.tight_layout()
    plt.savefig(output_dir / "hyperparam_importance.png"); plt.close()

def plot_loss_curve(estimator, output_dir: Path):
    """Plot training loss curve if available."""
    # HistGradientBoostingClassifier doesn't have train_score_ attribute
    # Instead, we can plot the validation scores if early stopping was used
    if hasattr(estimator, 'validation_scores_') and estimator.validation_scores_ is not None:
        plt.figure()
        plt.plot(estimator.validation_scores_, label="Validation Score")
        plt.xlabel("Iterations")
        plt.ylabel("Score")
        plt.title("Validation Score Curve")
        plt.legend()
        plt.savefig(output_dir / "validation_curve.png")
        plt.close()
    else:
        click.echo("[WARNING] No validation scores available for loss curve plot")

def compute_feature_importance(estimator, X_train, y_train, output_dir: Path, random_state=42):
    """Compute feature importance using available methods."""
    
    # Try native feature importances first
    if hasattr(estimator, "feature_importances_") and estimator.feature_importances_ is not None:
        return estimator.feature_importances_, "Native Feature Importances"
    
    # Fallback to permutation importance
    click.echo("[INFO] Computing permutation feature importance...")
    perm_importance = permutation_importance(
        estimator, X_train, y_train, 
        n_repeats=10, 
        random_state=random_state,
        n_jobs=-1,
        scoring="average_precision"  # Use same scoring as grid search
    )
    
    # Save detailed permutation results
    perm_df = pd.DataFrame({
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std,
        'feature_idx': range(len(perm_importance.importances_mean))
    })
    perm_df.to_csv(output_dir / "permutation_importance_details.csv", index=False)
    
    return perm_importance.importances_mean, "Permutation Feature Importance"

def plot_feature_importance(estimator, X_train, y_train, output_dir: Path, feature_names=None, random_state=42):
    """Plot feature importance with fallback to permutation importance."""
    
    importances, importance_type = compute_feature_importance(
        estimator, X_train, y_train, output_dir, random_state
    )
    
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(importances))]
    
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha="right")
    plt.title(f"{importance_type}")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png")
    plt.close()
    
    return importances, importance_type

def _save_feature_importance_plot(estimator, X_train, y_train, feature_names_path: Path, out_path: Path, random_state=42):
    """Save a bar plot of feature importances with error bars for permutation importance."""
    import json
    import matplotlib.pyplot as plt
    
    # Get feature names
    try:
        with feature_names_path.open("r", encoding="utf-8") as f:
            feature_names = json.load(f)
    except:
        click.echo("[WARNING] Could not load feature names")
        return
    
    # Compute importance
    if hasattr(estimator, "feature_importances_") and estimator.feature_importances_ is not None:
        importances = estimator.feature_importances_
        importance_type = "Native Feature Importances"
        error_bars = None
    else:
        # Use permutation importance with error bars
        click.echo("[INFO] Computing permutation importance for horizontal plot...")
        perm_importance = permutation_importance(
            estimator, X_train, y_train,
            n_repeats=10,
            random_state=random_state,
            n_jobs=-1,
            scoring="average_precision"
        )
        importances = perm_importance.importances_mean
        error_bars = perm_importance.importances_std
        importance_type = "Permutation Feature Importance"
    
    if len(feature_names) != len(importances):
        click.echo("[WARNING] Feature names length doesn't match importances length")
        return
    
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    sorted_errors = error_bars[indices] if error_bars is not None else None
    
    plt.figure(figsize=(14, 8))
    bars = plt.barh(sorted_features, sorted_importances, color="skyblue")
    
    # Add error bars for permutation importance
    if sorted_errors is not None:
        plt.errorbar(sorted_importances, range(len(sorted_features)), 
                    xerr=sorted_errors, fmt='none', color='black', capsize=2)
    
    plt.xlabel("Importance")
    plt.yticks(fontsize=6)
    plt.ylabel("Features")
    plt.title(f"{importance_type}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_param_performance(grid, output_dir: Path):
    import numpy as _np
    results = grid.cv_results_
    mean_scores = results["mean_test_score"]
    params_list = results["params"]
    # support both GridSearchCV (param_grid) and RandomizedSearchCV (param_distributions)
    if hasattr(grid, "param_grid"):
        param_keys = grid.param_grid.keys()
    else:
        param_keys = grid.param_distributions.keys()
    for param in param_keys:
        # collect unique values and drop None to avoid plotting errors
        raw_vals = {p[param] for p in params_list}
        values = sorted(v for v in raw_vals if v is not None)
        avg_scores = [
            _np.mean([mean_scores[i] for i, p in enumerate(params_list) if p[param] == v])
            for v in values
        ]
        plt.figure()
        plt.plot(values, avg_scores, marker="o")
        plt.xlabel(param)
        plt.ylabel("Mean Test Score")
        plt.title(f"Grid Search: {param} vs Score")
        plt.tight_layout()
        plt.savefig(output_dir / f"param_{param}_performance.png")
        plt.close()

# new plotting functions
def plot_hyperparam_heatmap(grid, output_dir: Path, param_x="max_iter", param_y="learning_rate"):
    import numpy as _np
    results = pd.DataFrame(grid.cv_results_)
    pivot = results.pivot_table(index=f"param_{param_y}", columns=f"param_{param_x}", values="mean_test_score")
    plt.figure(figsize=(8,6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
    plt.title(f"Heatmap: {param_y} vs {param_x}")
    plt.tight_layout()
    plt.savefig(output_dir / f"heatmap_{param_y}_vs_{param_x}.png")
    plt.close()

def plot_pairwise_scatter(grid, output_dir: Path, params=None):
    import pandas as _pd
    results = _pd.DataFrame(grid.cv_results_)
    if params is None:
        # support both GridSearchCV (param_grid) and RandomizedSearchCV (param_distributions)
        if hasattr(grid, "param_grid"):
            param_keys = grid.param_grid.keys()
        else:
            param_keys = grid.param_distributions.keys()
        params = list(param_keys)[:3]
    sns.pairplot(
        results,
        vars=[f"param_{p}" for p in params],
        hue="mean_test_score",
        palette="viridis",
        diag_kind="kde"
    )
    plt.suptitle("Pairwise Scatter: params colored by mean_test_score", y=1.02)
    plt.savefig(output_dir / "pairwise_scatter.png")
    plt.close()

def plot_score_distribution(grid, output_dir: Path):
    results = pd.DataFrame(grid.cv_results_)
    if hasattr(grid, "param_grid"):
        param_keys = grid.param_grid.keys()
    else:
        param_keys = grid.param_distributions.keys()
    params = [p for p in param_keys if len(results[f"param_{p}"].unique()) > 1]
    if not params:
        return
    n = len(params)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 4), squeeze=False)
    for i, p in enumerate(params):
        ax = axes[0, i]
        col = f"param_{p}"
        unique_vals = sorted(results[col].dropna().unique())
        try:
            sns.boxplot(x=col, y="mean_test_score", data=results, order=unique_vals, ax=ax)
        except ValueError:
            continue
        ax.set_title(f"Score Distribution for {p}")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "score_distribution_all.png")
    plt.close()

@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--features_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Directory containing X_train.npy and y_train.npy")
@click.option("--output_dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Directory to save model and artifacts")
@click.option("--scoring", default="average_precision", show_default=True,
              help="Scoring metric for GridSearchCV")
@click.option("--cv", type=int, default=5, show_default=True,
              help="Number of CV folds")
@click.option("--n_jobs", type=int, default=-1, show_default=True,
              help="Number of parallel jobs")
@click.option("--random_state", type=int, default=42, show_default=True,
              help="Random seed")
@click.option("--grid_search/--no-grid_search", default=False, show_default=True,
              help="Whether to perform RandomizedSearchCV or use default params")
@click.option("--max_combinations", type=int, default=-1, show_default=True,
              help="Number of hyperparameter combinations to try in RandomizedSearchCV (-1 means all)")
def main(
    features_dir: Path,
    output_dir: Path,
    scoring: str,
    cv: int,
    n_jobs: int,
    random_state: int,
    grid_search: bool,
    max_combinations: int
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load train arrays
    X_train, y_train = load_arrays(features_dir)
    rs = check_random_state(random_state)

    # ---- Estimator (HistGradientBoostingClassifier)
    # Enable feature importances computation
    gbm_kwargs = {
        "random_state": random_state,
        "verbose": 0,
        # Enable early stopping by default
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 10,
        "scoring": "loss",  # Use loss for early stopping
        # Explicitly set to enable importances
        "categorical_features": None,
    }

    estimator = HistGradientBoostingClassifier(**gbm_kwargs)

    if grid_search:
        # ---- RandomizedSearchCV branch
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        param_grid = default_param_grid()
        all_combinations = list(ParameterGrid(param_grid))
        total_candidates = len(all_combinations)
        n_iter = total_candidates if max_combinations == -1 else max_combinations

        click.echo(f"[INFO] Total hyperparameter candidates: {total_candidates}")
        grid = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            n_jobs=n_jobs,
            cv=cv_splitter,
            refit=True,
            return_train_score=False,
            verbose=1,
            random_state=random_state,
        )
        click.echo(f"[INFO] Starting RandomizedSearchCV with HistGradientBoostingClassifier (grid_search=True, n_iter={n_iter})")
        with tqdm_joblib(desc="RandomizedSearchCV", total=n_iter * cv):
            grid.fit(X_train, y_train)

        best_estimator = grid.best_estimator_
        best_params = grid.best_params_
        cv_df = pd.DataFrame(grid.cv_results_)
    else:
        # ---- Default‐params branch
        click.echo(f"[INFO] Skipping GridSearchCV; training HistGradientBoostingClassifier with default parameters (grid_search=False)")
        best_estimator = estimator.fit(X_train, y_train)
        best_params = best_estimator.get_params()
        cv_df = pd.DataFrame([best_params])

    # ---- Save artifacts (both branches)
    best_model_path = output_dir / "best_model.joblib"
    best_params_path = output_dir / "best_params.json"
    cv_results_path = output_dir / "cv_results.csv"
    summary_path = output_dir / "training_summary.json"

    # Save the best estimator directly
    joblib.dump(best_estimator, best_model_path)
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, default=str)  # Add default=str for non-serializable objects

    # Save CV table for transparency/tuning
    cv_df.to_csv(cv_results_path, index=False)

    # ---- Plots & diagnostics
    if grid_search:
        plot_grid_search_results(grid, output_dir)
        click.echo("[INFO] Saved grid search plot")
        plot_hyperparameter_importance(grid, output_dir)
        click.echo("[INFO] Saved hyperparameter importance plot")
        plot_param_performance(grid, output_dir)
        click.echo("[INFO] Saved parameter performance plots")

        # new visualizations
        plot_hyperparam_heatmap(grid, output_dir)
        click.echo("[INFO] Saved heatmap for hyperparameters")
        plot_pairwise_scatter(grid, output_dir)
        click.echo("[INFO] Saved pairwise scatter plot")
        plot_score_distribution(grid, output_dir)
        click.echo("[INFO] Saved score distribution plots")

    # common plots
    plot_loss_curve(best_estimator, output_dir)
    click.echo("[INFO] Saved validation curve")
    plot_feature_importance(best_estimator, X_train, y_train, output_dir, random_state=random_state)
    click.echo("[INFO] Saved feature importance plot")
    _save_feature_importance_plot(
        best_estimator,
        X_train,
        y_train, 
        features_dir / "feature_names.json",
        output_dir / "feature_importance_horizontal.png",
        random_state=random_state
    )
    click.echo("[INFO] Saved horizontal feature importance plot")

    # Minimal metadata (no test metrics)
    class_counts = {str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))}
    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "sklearn_version": sklearn.__version__,
        "estimator_type": "HistGradientBoostingClassifier",
        "features_dir": str(features_dir),
        "output_dir": str(output_dir),
        "X_train_shape": list(X_train.shape),
        "y_train_shape": list(y_train.shape),
        "scoring": scoring,
        "cv_folds": cv,
        "n_jobs": n_jobs,
        "random_state": random_state,
        "early_stopping_enabled": True,
        "param_grid_size": len(all_combinations) if grid_search else int(np.prod([len(v) for v in default_param_grid().values()])),
        "max_combinations": max_combinations,
        "class_distribution_train": class_counts,
        "best_params": best_params,
        "native_missing_value_support": True,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    click.echo(f"[SUCCESS] Saved HistGradientBoostingClassifier model → {best_model_path}")
    click.echo(f"[SUCCESS] Saved best params → {best_params_path}")
    click.echo(f"[SUCCESS] Saved CV results → {cv_results_path}")
    click.echo(f"[SUCCESS] Saved training summary → {summary_path}")


if __name__ == "__main__":
    main()
