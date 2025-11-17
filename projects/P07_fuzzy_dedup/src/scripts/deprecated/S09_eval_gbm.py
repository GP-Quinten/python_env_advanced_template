#!/usr/bin/env python3
# File: src/scripts/S09_eval_gbm.py
#
# Evaluate a trained model on the TEST split, produce standard metrics/plots,
# highlight PR region at precision >= 0.95 (with threshold), and write a Markdown report.

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import click
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    log_loss,
)
from sklearn.calibration import calibration_curve


def _load_arrays(features_dir: Path):
    X_test = np.load(features_dir / "X_test.npy")
    y_test = np.load(features_dir / "y_test.npy")
    return X_test, y_test


def _load_model(model_path: Path):
    return joblib.load(model_path)


def _get_proba(estimator, X: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.ravel()
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    raise ValueError("Estimator has neither predict_proba nor decision_function.")


def _choose_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    strategy: Optional[str],
    fixed_threshold: Optional[float],
) -> float:
    if fixed_threshold is not None:
        return float(fixed_threshold)
    if strategy is None or strategy.lower() == "none":
        return 0.5
    strategy = strategy.lower()
    if strategy == "f1":
        precision, recall, thr = precision_recall_curve(y_true, proba)
        precision, recall = precision[:-1], recall[:-1]
        f1 = (2 * precision * recall) / np.maximum(precision + recall, 1e-12)
        return float(thr[np.argmax(f1)]) if len(thr) else 0.5
    if strategy == "youden":
        fpr, tpr, thr = roc_curve(y_true, proba)
        j = tpr - fpr
        return float(thr[np.argmax(j)]) if len(thr) else 0.5
    return 0.5


def _save_individual_plots(recall, precision, ap, fpr, tpr, roc_auc, y_true, proba, out_dir: Path):
    # PR
    plt.figure()
    plt.step(recall, precision, where="post")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision–Recall (AP={ap:.4f})")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout(); plt.savefig(out_dir / "pr_curve.png", dpi=150); plt.close()

    # ROC
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC (AUC={roc_auc:.4f})")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout(); plt.savefig(out_dir / "roc_curve.png", dpi=150); plt.close()

    # Calibration
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="uniform")
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction of positives")
    plt.title("Calibration curve")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout(); plt.savefig(out_dir / "calibration.png", dpi=150); plt.close()

    # Histogram
    plt.figure()
    plt.hist(proba[y_true == 0], bins=30, alpha=0.5, label="Negative class", color="red")
    plt.hist(proba[y_true == 1], bins=30, alpha=0.5, label="Positive class", color="blue")
    plt.xlabel("Predicted probability (positive)"); plt.ylabel("Count")
    plt.title("Score distribution")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout(); plt.savefig(out_dir / "proba_hist.png", dpi=150); plt.close()


def _save_combined_figure(
    recall,
    precision,
    ap,
    fpr,
    tpr,
    roc_auc,
    y_true,
    proba,
    out_path: Path,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # PR (top-left)
    ax = axes[0, 0]
    ax.step(recall, precision, where="post", label="PR curve")
    # Highlight region precision >= 0.95
    mask = precision[:-1] >= 0.95
    ax.fill_between(recall[:-1], precision[:-1], 0.95, where=mask, step="post", alpha=0.3, label="Precision ≥ 0.95")
    ax.axhline(0.95, linestyle="--", linewidth=1, color="gray")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall (AP={ap:.4f})")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="best")

    # ROC (top-right)
    ax = axes[0, 1]
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC (AUC={roc_auc:.4f})")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # Calibration (bottom-left)
    ax = axes[1, 0]
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="uniform")
    ax.plot(mean_pred, frac_pos, marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # Histogram with classes (bottom-right)
    ax = axes[1, 1]
    ax.hist(proba[y_true == 0], bins=30, alpha=0.5, label="Negative class", color="red")
    ax.hist(proba[y_true == 1], bins=30, alpha=0.5, label="Positive class", color="blue")
    ax.set_xlabel("Predicted probability (positive)"); ax.set_ylabel("Count")
    ax.set_title("Score distribution")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _pr_highlight_plot(
    y_true: np.ndarray,
    proba: np.ndarray,
    target_precision: float,
    out_path: Path
) -> Tuple[Optional[float], Optional[float]]:
    """
    Save a PR plot highlighting the region where precision >= target_precision.
    Returns (threshold_for_target_precision, recall_at_that_threshold) if achievable, else (None, None).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # thresholds has length N-1; align to first N-1 precision/recall points
    p = precision[:-1]
    r = recall[:-1]
    t = thresholds

    # Find indices where precision >= target
    idxs = np.where(p >= target_precision)[0]
    thr_target = None
    rec_target = None
    if len(idxs) > 0:
        # choose the point with max recall among those meeting the precision target
        # (equivalently, the last index since thresholds are decreasing)
        best_idx = idxs[-1]
        thr_target = float(t[best_idx])
        rec_target = float(r[best_idx])

    # Plot
    plt.figure(figsize=(7, 5))
    plt.step(recall, precision, where="post", label="PR curve")

    # Highlight region precision >= target
    if len(idxs) > 0:
        # To shade, we need contiguous segments. Build a mask over r (N-1 long).
        mask = p >= target_precision
        # draw shaded bars between (r[i], r[i+1]) approx using scatter/line where mask True
        plt.fill_between(
            r,
            p,
            target_precision,
            where=mask,
            step="pre",
            alpha=0.3,
            label=f"Precision ≥ {target_precision:.2f}",
        )
        # mark chosen point if available
        if thr_target is not None:
            plt.scatter([rec_target], [p[best_idx]], s=40, zorder=3)
            plt.annotate(
                f"thr≈{thr_target:.4f}\nP≥{target_precision:.2f}, R≈{rec_target:.3f}",
                (rec_target, p[best_idx]),
                textcoords="offset points",
                xytext=(10, -20),
                fontsize=9,
            )
    else:
        plt.text(0.5, 0.5, f"No point with precision ≥ {target_precision:.2f}",
                 ha="center", va="center", transform=plt.gca().transAxes)

    plt.axhline(target_precision, linestyle="--", linewidth=1)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR curve with high-precision region")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return thr_target, rec_target


def _markdown_report(
    out_dir: Path,
    model_path: Path,
    metrics: dict,
    pr_highlight_png: str,
    combined_png: str,
    top_fp: Optional[str] = None,
    top_fn: Optional[str] = None,
):
    """Write a Markdown report that summarizes metrics and embeds plots."""
    cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    tn, fp_val = cm[0]
    fn, tp = cm[1]

    md_lines = []
    md_lines.append("# Model Evaluation Report")
    md_lines.append("")
    md_lines.append(f"- **Model:** `{model_path}`")
    md_lines.append(f"- **Samples:** {metrics.get('n_samples')}")
    md_lines.append(f"- **Prevalence (test):** {metrics.get('prevalence'):.4f}")
    md_lines.append("")
    md_lines.append("## Global Metrics")
    md_lines.append("")
    md_lines.append(f"- AUROC: **{metrics.get('auroc'):.4f}**")
    md_lines.append(f"- Average Precision (AUPRC): **{metrics.get('average_precision'):.4f}**")
    md_lines.append(f"- Brier score: **{metrics.get('brier_score'):.5f}**")
    if metrics.get("log_loss") is not None:
        md_lines.append(f"- Log loss: **{metrics.get('log_loss'):.5f}**")
    md_lines.append("")
    md_lines.append("## Thresholded Metrics")
    md_lines.append("")
    md_lines.append(f"- Decision threshold: **{metrics.get('threshold'):.4f}** (optimize: `{metrics.get('optimize')}`)")
    md_lines.append(f"- Accuracy: **{metrics.get('accuracy_at_threshold'):.4f}**")
    md_lines.append(f"- Precision: **{metrics.get('precision_at_threshold'):.4f}**")
    md_lines.append(f"- Recall: **{metrics.get('recall_at_threshold'):.4f}**")
    md_lines.append(f"- F1: **{metrics.get('f1_at_threshold'):.4f}**")
    md_lines.append("")
    md_lines.append("## Confusion Matrix")
    md_lines.append("")
    md_lines.append("|        | Pred 0 | Pred 1 |")
    md_lines.append("|--------|--------|--------|")
    md_lines.append(f"| True 0 | {tn:6d} | {fp_val:6d} |")
    md_lines.append(f"| True 1 | {fn:6d} | {tp:6d} |")
    md_lines.append("")
    # High precision operating point (0.95)
    thr095 = metrics.get("threshold_for_precision_0_95")
    rec095 = metrics.get("recall_at_precision_0_95")
    md_lines.append("## High-Precision Operating Point (Precision ≥ 0.95)")
    if thr095 is not None:
        md_lines.append(f"- Threshold achieving P≥0.95 (max recall): **{thr095:.4f}**")
        md_lines.append(f"- Recall at that threshold: **{rec095:.4f}**")
    else:
        md_lines.append("- No operating point reaches precision ≥ 0.95.")
    md_lines.append("")
    # Add top false positives/negatives if available
    if top_fp:
        md_lines.append("## Top False Positives")
        md_lines.append("")
        md_lines.append(top_fp)
        md_lines.append("")
    if top_fn:
        md_lines.append("## Top False Negatives")
        md_lines.append("")
        md_lines.append(top_fn)
        md_lines.append("")
    md_lines.append("## Key Plots")
    md_lines.append("")
    md_lines.append(f"![PR curve (highlighted P≥0.95)]({pr_highlight_png})")
    md_lines.append("")
    md_lines.append(f"![Evaluation plots (PR/ROC/Calibration/Histogram)]({combined_png})")
    md_lines.append("")
    (out_dir / "report.md").write_text("\n".join(md_lines), encoding="utf-8")


def _false_negatives_positives_report(
    predictions_df: pd.DataFrame,
    output_dir: Path,
):
    """
    Generate a Markdown report for False Negatives and False Positives,
    including pair names if available.
    """
    df = predictions_df.copy()
    # Create a 'pair' column using descriptive name columns if present
    if "registry_name" in df.columns and "alias" in df.columns:
        df["pair"] = df["registry_name"].astype(str) + " | " + df["alias"].astype(str)
    else:
        df["pair"] = df["pair_id"].astype(str)
    # Mark error types based on y_true and y_pred
    df["type"] = np.where(
        (df["y_true"] == 1) & (df["y_pred"] == 0), "False Negative",
        np.where((df["y_true"] == 0) & (df["y_pred"] == 1), "False Positive", "Correct")
    )
    false_negatives = df[df["type"] == "False Negative"].sort_values(by="proba", ascending=False)
    false_positives = df[df["type"] == "False Positive"].sort_values(by="proba", ascending=False)

    md_lines = []
    md_lines.append("# False Negatives and False Positives Report")
    md_lines.append("")
    md_lines.append("## False Negatives")
    md_lines.append("")
    if not false_negatives.empty:
        md_lines.append(false_negatives.to_markdown(index=False, tablefmt="pipe"))
    else:
        md_lines.append("No False Negatives found.")
    md_lines.append("")
    md_lines.append("## False Positives")
    md_lines.append("")
    if not false_positives.empty:
        md_lines.append(false_positives.to_markdown(index=False, tablefmt="pipe"))
    else:
        md_lines.append("No False Positives found.")
    md_lines.append("")
    (output_dir / "false_negatives_positives_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    click.echo(f"[INFO] False Negatives and False Positives report saved to {output_dir / 'false_negatives_positives_report.md'}")


def _save_feature_importance_plot(estimator, feature_names_path: Path, out_path: Path):
    """
    Save a bar plot of feature importances, specifying the type of importance used.
    """
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        importance_type = "Feature Importances (Gini Importance)"
    elif hasattr(estimator, "coef_") and estimator.coef_.ndim == 1:
        importances = np.abs(estimator.coef_)
        importance_type = "Feature Importances (Absolute Coefficients)"
    else:
        click.echo("[WARNING] Model does not support feature importances.")
        return

    with feature_names_path.open("r", encoding="utf-8") as f:
        feature_names = json.load(f)
    if len(feature_names) != len(importances):
        raise ValueError("Mismatch between number of features and feature importances.")
    
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_importances, color="skyblue")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title(f"{importance_type}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--features_dir",
              type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
              required=True,
              help="Directory with X_test.npy and y_test.npy.")
@click.option("--model_path",
              type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
              required=True,
              help="Path to trained model (e.g., best_model.joblib).")
@click.option("--output_dir",
              type=click.Path(dir_okay=True, file_okay=False, path_type=Path),
              required=True,
              help="Where to write metrics and plots.")
@click.option("--threshold", type=float, default=None,
              help="Fixed decision threshold (0..1). If not set, uses 0.5 unless --optimize is provided.")
@click.option("--optimize", type=click.Choice(["none", "f1", "youden"], case_sensitive=False),
              default="none", show_default=True,
              help="Choose threshold to optimize F1 or Youden's J (on TEST; for exploration only).")
@click.option("--target_precision", type=float, default=0.95, show_default=True,
              help="Target precision for highlighting PR curve and threshold extraction.")
@click.option("--feature_names_path",
              type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
              default="data/R07_make_features/feature_names.json",
              show_default=True,
              help="Path to JSON file containing feature names.")
# New options for joining with metadata:
@click.option("--pairs_meta",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None,
              help="Path to pairs metadata table (e.g. pairs_meta.parquet).")
@click.option("--test_idx",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None,
              help="Path to test index file (test_idx.npy).")
@click.option("--name_cols",
              default="registry_name,alias",  # updated default to match pairs_meta columns
              help="Comma-separated column names for pair names in pairs_meta.")
def main(
    features_dir: Path,
    model_path: Path,
    output_dir: Path,
    threshold: Optional[float],
    optimize: str,
    target_precision: float,
    feature_names_path: Path,
    pairs_meta: Optional[Path],
    test_idx: Optional[Path],
    name_cols: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data/model
    X_test, y_test = _load_arrays(features_dir)
    estimator = _load_model(model_path)

    # ---- Load test index if provided (or default)
    if test_idx is None:
        test_idx = features_dir / "test_idx.npy"
    test_ids = np.load(test_idx)

    # ---- Load pairs metadata if provided
    if pairs_meta is not None:
        meta = pd.read_parquet(pairs_meta)
        # Set pair_id as index for lookups
        meta = meta.set_index("pair_id")
        # Retrieve descriptive columns for test split (preserving order)
        test_meta = meta.loc[test_ids]
    else:
        test_meta = None

    # ---- Probabilities & curves
    proba = _get_proba(estimator, X_test)
    roc_auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)
    precision, recall, pr_thr = precision_recall_curve(y_test, proba)
    fpr, tpr, roc_thr = roc_curve(y_test, proba)

    # ---- Choose threshold for standard thresholded metrics
    thr = _choose_threshold(y_test, proba, optimize, threshold)
    y_pred = (proba >= thr).astype(int)

    # ---- Metrics JSON (global + thresholded)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec_score = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    brier = brier_score_loss(y_test, y_pred)
    try:
        ll = log_loss(y_test, np.vstack([1 - proba, proba]).T, labels=[0, 1])
    except Exception:
        ll = None

    metrics = {
        "n_samples": int(len(y_test)),
        "prevalence": float(np.mean(y_test)),
        "threshold": float(thr),
        "optimize": optimize,
        "auroc": float(roc_auc),
        "average_precision": float(ap),
        "accuracy_at_threshold": float(acc),
        "precision_at_threshold": float(prec),
        "recall_at_threshold": float(rec_score),
        "f1_at_threshold": float(f1),
        "brier_score": float(brier),
        "log_loss": (float(ll) if ll is not None else None),
        "confusion_matrix": cm,
    }

    # ---- Save curves & predictions CSV (existing code)
    pd.DataFrame({
        "recall": recall,
        "precision": precision,
        "threshold": np.r_[pr_thr, np.nan]
    }).to_csv(output_dir / "pr_curve.csv", index=False)

    pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "threshold": roc_thr
    }).to_csv(output_dir / "roc_curve.csv", index=False)

    pd.DataFrame({
        "y_true": y_test,
        "proba": proba,
        "y_pred": y_pred
    }).to_csv(output_dir / "predictions.csv", index=False)

    # ---- Plots (individual)
    _save_individual_plots(recall, precision, ap, fpr, tpr, roc_auc, y_test, proba, output_dir)

    # ---- Combined 2x2 figure
    combined_png = "evaluation_plots.png"
    _save_combined_figure(
        recall, precision, ap, fpr, tpr, roc_auc, y_test, proba, output_dir / combined_png
    )

    # ---- PR curve highlight at target precision
    pr_highlight_png = "pr_curve_highlight.png"
    thr095, rec095 = _pr_highlight_plot(
        y_true=y_test,
        proba=proba,
        target_precision=target_precision,
        out_path=output_dir / pr_highlight_png
    )
    metrics["threshold_for_precision_0_95"] = thr095
    metrics["recall_at_precision_0_95"] = rec095

    # ---- Save metrics JSON
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- Text summary
    report_txt = classification_report(y_test, (proba >= thr).astype(int), digits=4, zero_division=0)
    (output_dir / "classification_report.txt").write_text(report_txt)

    # ---- Join predictions with meta data (if available)
    predictions_df = pd.DataFrame({
        "pair_id": test_ids,
        "y_true": y_test,
        "proba": proba,
        "y_pred": y_pred
    })
    if test_meta is not None:
        # Separate name columns from the provided comma‐delimited list
        cols = [s.strip() for s in name_cols.split(",")]
        # Join the descriptive columns (and label if available)
        predictions_df = predictions_df.join(test_meta[cols + [test_meta.columns[-1]]], on="pair_id")
        # Save a CSV with full predictions info
        predictions_df.to_csv(output_dir / "predictions_with_names.csv", index=False)
    else:
        predictions_df.to_csv(output_dir / "predictions_with_names.csv", index=False)

    # ---- Save FP & FN reports as CSVs
    false_positives = predictions_df[(predictions_df["y_true"] == 0) & (predictions_df["y_pred"] == 1)]
    false_negatives = predictions_df[(predictions_df["y_true"] == 1) & (predictions_df["y_pred"] == 0)]
    false_positives.to_csv(output_dir / "false_positives.csv", index=False)
    false_negatives.to_csv(output_dir / "false_negatives.csv", index=False)

    # ---- Generate top errors markdown (top 3 each)
    top_fp_md = ""
    top_fn_md = ""
    if not false_positives.empty:
        top_fp_md = false_positives.sort_values(by="proba", ascending=False).head(3).to_markdown(index=False)
    if not false_negatives.empty:
        top_fn_md = false_negatives.sort_values(by="proba", ascending=False).head(3).to_markdown(index=False)

    # ---- Markdown report (modified to include top error tables)
    _markdown_report(
        out_dir=output_dir,
        model_path=model_path,
        metrics=metrics,
        pr_highlight_png=pr_highlight_png,
        combined_png=combined_png,
        top_fp=top_fp_md,
        top_fn=top_fn_md
    )

    # ---- Existing reports
    _false_negatives_positives_report(predictions_df, output_dir)
    _save_feature_importance_plot(estimator, feature_names_path, output_dir / "feature_importance.png")

    click.echo(f"[SUCCESS] Wrote evaluation artifacts to {output_dir}")
    click.echo("  - metrics.json")
    click.echo("  - pr_curve.csv / roc_curve.csv / predictions.csv")
    click.echo("  - pr_curve.png / roc_curve.png / calibration.png / proba_hist.png")
    click.echo("  - pr_curve_highlight.png (P≥target shaded, threshold annotated)")
    click.echo("  - evaluation_plots.png (combined 2x2 subplot)")
    click.echo("  - classification_report.txt")
    click.echo("  - report.md")
    click.echo("  - feature_importance.png")


if __name__ == "__main__":
    main()
