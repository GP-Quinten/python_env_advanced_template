from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import click
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, ParameterGrid
from tqdm_joblib import tqdm_joblib
from sklearn.utils import check_random_state
import joblib
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

def load_training_features_and_labels(features_dir: Path):
    """Load training features (X) and labels (y) from numpy arrays."""
    X_train = np.load(features_dir / "X_train.npy")
    y_train = np.load(features_dir / "y_train.npy")
    return X_train, y_train

def create_hyperparameter_search_space() -> Dict[str, Any]:
    """
    Define the hyperparameter search space for HistGradientBoostingClassifier.
    
    Returns:
        Dict containing parameter ranges for RandomizedSearchCV to sample from.
    """
    return {
        "max_iter": [100, 200, 300],                    # Maximum number of boosting iterations
        "learning_rate": [0.05, 0.1, 0.2],             # Step size for gradient descent
        "max_depth": [3, 6, 10, None],                  # Maximum depth of trees (None = unlimited)
        "min_samples_leaf": [10, 20, 30],               # Minimum samples required in leaf nodes
        "l2_regularization": [0.0, 0.1, 0.5],          # L2 regularization strength
        "max_bins": [128, 255],                         # Number of bins for numerical features
        "early_stopping": [True],                       # Enable early stopping
        "validation_fraction": [0.1],                   # Fraction of data for validation
        "n_iter_no_change": [10],                       # Iterations to wait before stopping
    }

def plot_randomized_search_performance_by_max_iter(randomized_search_results, output_dir: Path):
    """
    Plot how model performance varies with max_iter parameter from RandomizedSearchCV.
    
    Args:
        randomized_search_results: Fitted RandomizedSearchCV object
        output_dir: Directory to save the plot
    """
    # Extract max_iter values and corresponding scores from CV results
    max_iter_values = np.array([int(x) for x in randomized_search_results.cv_results_['param_max_iter']])
    cv_scores = randomized_search_results.cv_results_['mean_test_score']
    
    # Calculate average score for each unique max_iter value
    unique_max_iters = sorted(set(max_iter_values))
    average_scores = [
        np.mean([cv_scores[i] for i, val in enumerate(max_iter_values) if val == unique_iter])
        for unique_iter in unique_max_iters
    ]
    
    # Create and save the plot
    plt.figure(figsize=(8, 6))
    plt.plot(unique_max_iters, average_scores, marker='o', linewidth=2, markersize=8)
    plt.xlabel("Maximum Iterations (max_iter)")
    plt.ylabel("Mean Cross-Validation Score")
    plt.title("Model Performance vs Maximum Iterations")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "performance_vs_max_iter.png", dpi=150)
    plt.close()

def plot_individual_hyperparameter_effects(randomized_search_results, output_dir: Path):
    """
    Create individual performance plots for each hyperparameter.
    
    Args:
        randomized_search_results: Fitted RandomizedSearchCV object
        output_dir: Directory to save the plots
    """
    cv_results = randomized_search_results.cv_results_
    mean_scores = cv_results["mean_test_score"]
    parameter_combinations = cv_results["params"]
    hyperparameter_names = randomized_search_results.param_distributions.keys()
    
    for param_name in hyperparameter_names:
        # Get unique values for this parameter
        unique_values = sorted(set(
            combo[param_name] for combo in parameter_combinations 
            if combo[param_name] is not None
        ))
        
        # Calculate average score for each parameter value
        avg_scores_per_value = [
            np.mean([
                mean_scores[i] for i, combo in enumerate(parameter_combinations) 
                if combo[param_name] == value
            ])
            for value in unique_values
        ]
        
        # Create and save individual parameter plot
        plt.figure(figsize=(8, 6))
        plt.plot(unique_values, avg_scores_per_value, marker="o", linewidth=2, markersize=8)
        plt.xlabel(f"Parameter: {param_name}")
        plt.ylabel("Mean Cross-Validation Score")
        plt.title(f"Model Performance vs {param_name}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"performance_vs_{param_name}.png", dpi=150)
        plt.close()

def compute_and_plot_feature_importance(trained_model, X_train, y_train, output_dir: Path, 
                                       feature_names_file: Path = None, random_state=42):
    """
    Compute feature importance using permutation importance and create visualizations.
    
    Args:
        trained_model: The fitted HistGradientBoostingClassifier
        X_train: Training features
        y_train: Training labels
        output_dir: Directory to save plots
        feature_names_file: Optional path to JSON file with feature names
        random_state: Random seed for reproducibility
    """
    click.echo("[INFO] Computing permutation feature importance...")
    
    # Compute permutation importance (robust method for any model type)
    perm_importance_results = permutation_importance(
        trained_model, X_train, y_train, 
        n_repeats=10,                           # Number of times to permute each feature
        random_state=random_state,
        n_jobs=-1,                             # Use all available cores
        scoring="average_precision"            # Same metric as used in hyperparameter search
    )
    
    # Load feature names if available
    try:
        if feature_names_file and feature_names_file.exists():
            with feature_names_file.open("r", encoding="utf-8") as f:
                feature_names = json.load(f)
        else:
            feature_names = [f"Feature_{i}" for i in range(len(perm_importance_results.importances_mean))]
    except Exception:
        feature_names = [f"Feature_{i}" for i in range(len(perm_importance_results.importances_mean))]
    
    # Save detailed importance results to CSV
    importance_details_df = pd.DataFrame({
        'feature_name': feature_names,
        'importance_mean': perm_importance_results.importances_mean,
        'importance_std': perm_importance_results.importances_std,
        'feature_index': range(len(perm_importance_results.importances_mean))
    })
    importance_details_df.to_csv(output_dir / "feature_importance_details.csv", index=False)
    
    # Create horizontal bar plot with error bars
    sorted_indices = np.argsort(perm_importance_results.importances_mean)[::-1]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]
    sorted_importance_means = perm_importance_results.importances_mean[sorted_indices]
    sorted_importance_stds = perm_importance_results.importances_std[sorted_indices]
    
    plt.figure(figsize=(12, max(8, len(feature_names) * 0.3)))
    bars = plt.barh(sorted_feature_names, sorted_importance_means, color="skyblue", alpha=0.8)
    plt.errorbar(sorted_importance_means, range(len(sorted_feature_names)), 
                xerr=sorted_importance_stds, fmt='none', color='red', capsize=3, alpha=0.7)
    
    plt.xlabel("Permutation Importance (Mean ± Std)")
    plt.ylabel("Features")
    plt.title("Feature Importance Analysis\n(Permutation-based with error bars)")
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_horizontal.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return perm_importance_results.importances_mean

def save_training_metadata(output_dir: Path, X_train, y_train, best_hyperparameters: dict, 
                          training_config: dict, hyperparameter_search_enabled: bool):
    """
    Save comprehensive metadata about the training process.
    
    Args:
        output_dir: Directory to save metadata
        X_train: Training features (for shape info)
        y_train: Training labels (for class distribution)
        best_hyperparameters: Best parameters found during search
        training_config: Configuration used for training
        hyperparameter_search_enabled: Whether hyperparameter search was performed
    """
    # Calculate class distribution
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    class_distribution = {str(cls): int(count) for cls, count in zip(unique_classes, class_counts)}
    
    # Create comprehensive metadata
    training_metadata = {
        "training_timestamp": datetime.utcnow().isoformat() + "Z",
        "sklearn_version": sklearn.__version__,
        "model_type": "HistGradientBoostingClassifier",
        
        # Data information
        "training_data": {
            "features_shape": list(X_train.shape),
            "labels_shape": list(y_train.shape),
            "class_distribution": class_distribution,
            "total_samples": int(len(y_train)),
            "num_features": int(X_train.shape[1])
        },
        
        # Training configuration
        "training_configuration": training_config,
        "hyperparameter_search_performed": hyperparameter_search_enabled,
        "best_hyperparameters": best_hyperparameters,
        
        # Model capabilities
        "model_features": {
            "handles_missing_values": True,
            "supports_early_stopping": True,
            "gradient_boosting_type": "histogram-based"
        }
    }
    
    # Save metadata to JSON
    metadata_path = output_dir / "training_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(training_metadata, f, indent=2, default=str)
    
    return metadata_path

@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--features_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Directory containing X_train.npy and y_train.npy files")
@click.option("--output_dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Directory to save trained model and analysis artifacts")
@click.option("--scoring", default="average_precision", show_default=True,
              help="Evaluation metric for hyperparameter optimization")
@click.option("--cv_folds", type=int, default=5, show_default=True,
              help="Number of cross-validation folds for hyperparameter search")
@click.option("--n_jobs", type=int, default=-1, show_default=True,
              help="Number of parallel jobs (-1 uses all available cores)")
@click.option("--random_state", type=int, default=42, show_default=True,
              help="Random seed for reproducible results")
@click.option("--enable_hyperparameter_search/--disable_hyperparameter_search", default=False, show_default=True,
              help="Enable RandomizedSearchCV for hyperparameter optimization")
@click.option("--max_search_iterations", type=int, default=-1, show_default=True,
              help="Maximum hyperparameter combinations to try (-1 tries all possible combinations)")
def main(
    features_dir: Path,
    output_dir: Path,
    scoring: str,
    cv_folds: int,
    n_jobs: int,
    random_state: int,
    enable_hyperparameter_search: bool,
    max_search_iterations: int
):
    """
    Train a HistGradientBoostingClassifier for binary classification with optional hyperparameter optimization.
    
    This script:
    1. Loads training data from numpy arrays
    2. Optionally performs hyperparameter search using RandomizedSearchCV with StratifiedKFold
    3. Trains the final model with best parameters
    4. Generates comprehensive analysis plots and feature importance
    5. Saves model, parameters, and metadata
    """
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ---- STEP 1: Load training data ----
    click.echo("[INFO] Loading training features and labels...")
    X_train, y_train = load_training_features_and_labels(features_dir)
    click.echo(f"[INFO] Training data shape: X={X_train.shape}, y={y_train.shape}")
    
    # Initialize random state for reproducibility
    random_state_generator = check_random_state(random_state)
    
    # ---- STEP 2: Configure base model ----
    base_model_config = {
        "random_state": random_state,
        "verbose": 0,
        "early_stopping": True,          # Enable early stopping to prevent overfitting
        "validation_fraction": 0.1,      # Use 10% of training data for validation
        "n_iter_no_change": 10,         # Stop if no improvement for 10 iterations
        "scoring": "loss",              # Use loss for early stopping decisions
    }
    
    base_classifier = HistGradientBoostingClassifier(**base_model_config)
    
    if enable_hyperparameter_search:
        # ---- STEP 3A: Hyperparameter optimization branch ----
        click.echo("[INFO] Hyperparameter search ENABLED - using RandomizedSearchCV")
        
        # Create stratified cross-validation splitter (maintains class balance in each fold)
        cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        # Define hyperparameter search space
        hyperparameter_space = create_hyperparameter_search_space()
        
        # Calculate total possible combinations
        all_possible_combinations = list(ParameterGrid(hyperparameter_space))
        total_combinations = len(all_possible_combinations)
        
        # Determine number of iterations to perform
        search_iterations = total_combinations if max_search_iterations == -1 else max_search_iterations
        
        click.echo(f"[INFO] Total possible hyperparameter combinations: {total_combinations}")
        click.echo(f"[INFO] Will evaluate {search_iterations} combinations")
        
        # Configure RandomizedSearchCV
        # This performs:
        # 1. Random sampling from hyperparameter space
        # 2. Cross-validation for each sampled combination
        # 3. Selection of best parameters based on scoring metric
        hyperparameter_optimizer = RandomizedSearchCV(
            estimator=base_classifier,
            param_distributions=hyperparameter_space,    # Space to sample from
            n_iter=search_iterations,                    # Number of combinations to try
            scoring=scoring,                             # Metric to optimize (average_precision)
            n_jobs=n_jobs,                              # Parallel processing
            cv=cv_splitter,                             # StratifiedKFold cross-validation
            refit=True,                                 # Refit on full dataset with best params
            return_train_score=False,                   # Don't compute train scores (saves time)
            verbose=1,                                  # Show progress
            random_state=random_state,
        )
        
        # Perform hyperparameter search with progress tracking
        click.echo(f"[INFO] Starting RandomizedSearchCV (this may take a while)...")
        with tqdm_joblib(desc="Hyperparameter Search", total=search_iterations * cv_folds):
            hyperparameter_optimizer.fit(X_train, y_train)
        
        # Extract results
        best_trained_model = hyperparameter_optimizer.best_estimator_
        best_hyperparameters = hyperparameter_optimizer.best_params_
        cv_results_dataframe = pd.DataFrame(hyperparameter_optimizer.cv_results_)
        
        click.echo(f"[INFO] Best cross-validation score: {hyperparameter_optimizer.best_score_:.4f}")
        click.echo(f"[INFO] Best hyperparameters: {best_hyperparameters}")
        
    else:
        # ---- STEP 3B: Default parameters branch ----
        click.echo("[INFO] Hyperparameter search DISABLED - using default parameters")
        
        # Train with default parameters (no optimization)
        best_trained_model = base_classifier.fit(X_train, y_train)
        best_hyperparameters = best_trained_model.get_params()
        cv_results_dataframe = pd.DataFrame([best_hyperparameters])  # Single row for consistency
    
    # ---- STEP 4: Save trained model and parameters ----
    model_save_path = output_dir / "trained_model.joblib"
    parameters_save_path = output_dir / "best_hyperparameters.json"
    cv_results_save_path = output_dir / "cross_validation_results.csv"
    
    # Save the trained model
    joblib.dump(best_trained_model, model_save_path)
    click.echo(f"[SUCCESS] Trained model saved to: {model_save_path}")
    
    # Save best hyperparameters
    with open(parameters_save_path, "w", encoding="utf-8") as f:
        json.dump(best_hyperparameters, f, indent=2, default=str)
    click.echo(f"[SUCCESS] Best parameters saved to: {parameters_save_path}")
    
    # Save cross-validation results
    cv_results_dataframe.to_csv(cv_results_save_path, index=False)
    click.echo(f"[SUCCESS] CV results saved to: {cv_results_save_path}")
    
    # ---- STEP 5: Generate analysis plots ----
    if enable_hyperparameter_search:
        click.echo("[INFO] Generating hyperparameter analysis plots...")
        
        plot_randomized_search_performance_by_max_iter(hyperparameter_optimizer, output_dir)
        click.echo("[SUCCESS] Saved performance vs max_iter plot")
        
        plot_individual_hyperparameter_effects(hyperparameter_optimizer, output_dir)
        click.echo("[SUCCESS] Saved individual hyperparameter effect plots")
    
    # ---- STEP 6: Feature importance analysis ----
    click.echo("[INFO] Computing feature importance...")
    feature_names_file = features_dir / "feature_names.json"
    compute_and_plot_feature_importance(
        best_trained_model, X_train, y_train, output_dir, 
        feature_names_file, random_state
    )
    click.echo("[SUCCESS] Feature importance analysis completed")
    
    # ---- STEP 7: Save comprehensive metadata ----
    training_config = {
        "features_directory": str(features_dir),
        "output_directory": str(output_dir),
        "scoring_metric": scoring,
        "cv_folds": cv_folds,
        "n_jobs": n_jobs,
        "random_state": random_state,
        "max_search_iterations": max_search_iterations,
    }
    
    metadata_path = save_training_metadata(
        output_dir, X_train, y_train, best_hyperparameters, 
        training_config, enable_hyperparameter_search
    )
    click.echo(f"[SUCCESS] Training metadata saved to: {metadata_path}")
    
    click.echo("\n" + "="*60)
    click.echo("TRAINING COMPLETED SUCCESSFULLY!")
    click.echo("="*60)
    click.echo(f"Model type: HistGradientBoostingClassifier")
    click.echo(f"Hyperparameter search: {'ENABLED' if enable_hyperparameter_search else 'DISABLED'}")
    click.echo(f"Training samples: {len(y_train):,}")
    click.echo(f"Number of features: {X_train.shape[1]:,}")
    click.echo(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

