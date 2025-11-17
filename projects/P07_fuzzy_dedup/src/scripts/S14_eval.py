#!/usr/bin/env python3
# Evaluate predictions (CSV) against labels from the original dataset (JSON).
# Saves metrics, plots, and joined CSV.

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import click
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

COL_PAIR_ID = "pair_hash_id"


def _align_by_key(preds: pd.DataFrame, data: pd.DataFrame, left_col: str, right_col: str) -> pd.DataFrame:
    """Return DataFrame with y_true joined to preds."""
    preds = preds.copy()
    data = data.copy()

    # 1) Preferred: join on pair_id if present in both
    if COL_PAIR_ID in preds.columns and COL_PAIR_ID in data.columns:
        return preds.merge(data[[COL_PAIR_ID]], on=COL_PAIR_ID, how="left").merge(
            data, on=COL_PAIR_ID, how="left", suffixes=("", "_data")
        )

    # 2) Next: join on (left_col, right_col) if present in both
    if {left_col, right_col}.issubset(preds.columns) and {left_col, right_col}.issubset(data.columns):
        return preds.merge(
            data[[left_col, right_col]],
            on=[left_col, right_col],
            how="left"
        ).merge(
            data,
            on=[left_col, right_col],
            how="left",
            suffixes=("", "_data")
        )

    # 3) Fallback: align by order if lengths match
    if len(preds) == len(data):
        preds = preds.reset_index(drop=True)
        data = data.reset_index(drop=True)
        for col in data.columns:
            preds[col] = data[col]
        return preds

    raise click.ClickException(
        "Could not align predictions with dataset. "
        "Provide pair_id in both, or left/right name columns, or ensure same row order/length."
    )


def _choose_threshold(y_true: np.ndarray, proba: np.ndarray,
                      strategy: str = "none", fixed_threshold: Optional[float] = None) -> float:
    if fixed_threshold is not None:
        return float(fixed_threshold)
    strategy = (strategy or "none").lower()
    if strategy == "f1":
        precision, recall, thr = precision_recall_curve(y_true, proba)
        precision, recall = precision[:-1], recall[:-1]
        if len(precision) == 0:
            return 0.5
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
    # highlight precision ≥ 0.95
    mask = precision[:-1] >= 0.95
    plt.fill_between(
        recall[:-1], precision[:-1], 0.95, where=mask,
        step="post", alpha=0.3, label="Precision ≥ 0.95"
    )
    plt.axhline(0.95, linestyle="--", linewidth=1, color="gray")
    plt.legend()
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision–Recall (AP={ap:.4f})")
    plt.tight_layout(); plt.savefig(out_dir / "pr_curve.png", dpi=150); plt.close()

    # ROC
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC (AUC={roc_auc:.4f})")
    plt.tight_layout(); plt.savefig(out_dir / "roc_curve.png", dpi=150); plt.close()

    # Calibration
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="uniform")
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction of positives")
    plt.title("Calibration")
    plt.tight_layout(); plt.savefig(out_dir / "calibration.png", dpi=150); plt.close()

    # Histogram
    plt.figure()
    plt.hist(proba[y_true == 0], bins=30, alpha=0.5, label="neg")
    plt.hist(proba[y_true == 1], bins=30, alpha=0.5, label="pos")
    plt.xlabel("Predicted probability"); plt.ylabel("Count")
    plt.title("Score distribution")
    plt.legend()
    plt.tight_layout(); plt.savefig(out_dir / "proba_hist.png", dpi=150); plt.close()


def _save_combined_figure(recall, precision, ap, fpr, tpr, roc_auc, y_true, proba, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # PR (top-left) with shaded region precision ≥ 0.95
    ax = axes[0, 0]
    ax.step(recall, precision, where="post", label="PR curve")
    mask = precision[:-1] >= 0.95
    ax.fill_between(
        recall[:-1], precision[:-1], 0.95, where=mask,
        step="post", alpha=0.3, label="Precision ≥ 0.95"
    )
    ax.axhline(0.95, linestyle="--", linewidth=1, color="gray")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall (AP={ap:.4f})")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="best")

    # ROC (top-right)
    ax = axes[0, 1]
    ax.plot(fpr, tpr); ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC (AUC={roc_auc:.4f})")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # Calibration (bottom-left)
    ax = axes[1, 0]
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="uniform")
    ax.plot(mean_pred, frac_pos, marker="o"); ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # Histogram (bottom-right)
    ax = axes[1, 1]
    ax.hist(proba[y_true == 0], bins=30, alpha=0.5, label="neg")
    ax.hist(proba[y_true == 1], bins=30, alpha=0.5, label="pos")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Count")
    ax.set_title("Score distribution")
    ax.legend(); ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _pr_highlight_plot(y_true: np.ndarray, proba: np.ndarray,
                       target_precision: float, out_path: Path) -> Tuple[Optional[float], Optional[float]]:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    p, r, t = precision[:-1], recall[:-1], thresholds
    thr_target = rec_target = None
    if len(p):
        idxs = np.where(p >= target_precision)[0]
        if len(idxs):
            best_idx = idxs[-1]
            thr_target = float(t[best_idx])
            rec_target = float(r[best_idx])

    plt.figure(figsize=(7, 5))
    plt.step(recall, precision, where="post", label="PR")
    if len(p):
        mask = p >= target_precision
        plt.fill_between(r, p, target_precision, where=mask, step="pre", alpha=0.3, label=f"P≥{target_precision:.2f}")
        if thr_target is not None and rec_target is not None:
            plt.scatter([rec_target], [precision[np.where(recall == rec_target)[0][0]]], s=30)
    plt.axhline(target_precision, linestyle="--", linewidth=1)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("PR (high-precision region)")
    plt.legend(); plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    return thr_target, rec_target


def _false_negatives_positives_report(predictions_df: pd.DataFrame, output_dir: Path):
    df = predictions_df.copy()
    # extract acronyms from registry names
    if "left_name" in df.columns:
        df["left_acronym"] = df["left_name"].str.extract(r"\(([^)]+)\)", expand=False).fillna("")
    if "right_name" in df.columns:
        df["right_acronym"] = df["right_name"].str.extract(r"\(([^)]+)\)", expand=False).fillna("")

    # compute FN/FP flags
    df["type"] = np.where(
        (df["label"] == 1) & (df["y_pred"] == 0), "False Negative",
        np.where((df["label"] == 0) & (df["y_pred"] == 1), "False Positive", "Correct")
    )

    false_negatives = df[df["type"] == "False Negative"].sort_values(by="proba", ascending=True)
    false_positives = df[df["type"] == "False Positive"].sort_values(by="proba", ascending=False)

    md_lines = ["# False Negatives and False Positives Report\n", "## False Negatives\n"]
    cols = ["left_name", "left_acronym", "right_name", "right_acronym", "label", "proba", "y_pred"]
    if not false_negatives.empty:
        md_lines.append(false_negatives[cols].to_markdown(index=False, tablefmt="pipe"))
    else:
        md_lines.append("No False Negatives found.")
    md_lines.append("\n## False Positives\n")
    if not false_positives.empty:
        md_lines.append(false_positives[cols].to_markdown(index=False, tablefmt="pipe"))
    else:
        md_lines.append("No False Positives found.")
    (output_dir / "false_negatives_positives_report.md").write_text("\n".join(md_lines), encoding="utf-8")


def _markdown_report(out_dir: Path, metrics: dict):
    cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    tn, fp = cm[0]
    fn, tp = cm[1]
    md = []
    md.append("# Prediction Evaluation Report\n")
    md.append(f"- Samples: **{metrics['n_samples']}**, Prevalence: **{metrics['prevalence']:.4f}**\n")
    md.append(f"- AUROC: **{metrics['auroc']:.4f}**, AUPRC: **{metrics['average_precision']:.4f}**\n")
    md.append(f"- Threshold: **{metrics['threshold']:.4f}** (strategy: `{metrics['threshold_strategy']}`)\n")
    md.append(f"- Acc: **{metrics['accuracy_at_threshold']:.4f}**, "
              f"P: **{metrics['precision_at_threshold']:.4f}**, "
              f"R: **{metrics['recall_at_threshold']:.4f}**, "
              f"F1: **{metrics['f1_at_threshold']:.4f}**\n")
    md.append("\n## Confusion Matrix\n\n")
    md.append("|        | Pred 0 | Pred 1 |\n|--------|--------|--------|\n")
    md.append(f"| True 0 | {tn:6d} | {fp:6d} |\n")
    md.append(f"| True 1 | {fn:6d} | {tp:6d} |\n")
    md.append("\n## Key Plots\n\n")
    md.append("![PR curve (highlighted P≥0.95)](pr_curve_highlight.png)\n\n")
    md.append("![Evaluation plots (PR/ROC/Calibration/Histogram)](evaluation_plots.png)\n\n")
    md.append("![Feature Importance](feature_importance.png)\n\n")
    md.append("![PR](pr_curve.png)\n\n")
    md.append("![ROC](roc_curve.png)\n\n")
    md.append("![Calibration](calibration.png)\n\n")
    md.append("![Histogram](proba_hist.png)\n")
    # insert precision→threshold/recall table
    md.append("\n## Thresholds for target precisions\n\n")
    md.append("| Precision | Threshold | Recall |\n|-----------|-----------|--------|\n")
    for p in sorted(metrics.get("pr_thresholds", {})):
        thr = metrics["pr_thresholds"][p]["threshold"]
        rec = metrics["pr_thresholds"][p]["recall"]
        md.append(f"| {p:.2f} | {thr if thr is not None else 'N/A'} | "
                  f"{rec if rec is not None else 'N/A'} |\n")
    (out_dir / "report.md").write_text("".join(md), encoding="utf-8")


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--dataset_json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Original dataset JSON (contains labels).")
@click.option("--predictions_csv", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Predictions CSV (from R13). Should include 'proba' column (or 'score').")
@click.option("--output_dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--label_col", default="label", show_default=True)
@click.option("--left_col", default="left_name", show_default=True)
@click.option("--right_col", default="right_name", show_default=True)
@click.option("--optimize", type=click.Choice(["none", "f1", "youden"], case_sensitive=False),
              default="none", show_default=True)
@click.option("--threshold", type=float, default=None, help="Fixed threshold (0..1). Overrides --optimize if set.")
@click.option("--target_precision", type=float, default=0.95, show_default=True)
def main(dataset_json, predictions_csv, output_dir, label_col, left_col, right_col, optimize, threshold, target_precision):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load
    data = pd.read_json(dataset_json)
    preds = pd.read_csv(predictions_csv)

    # Normalize proba column name
    if "proba" not in preds.columns:
        if "score" in preds.columns:
            preds = preds.rename(columns={"score": "proba"})
        else:
            raise click.ClickException("Predictions CSV must contain 'proba' (or 'score').")

    # Align
    joined = _align_by_key(preds, data, left_col, right_col)
    if label_col not in joined.columns:
        raise click.ClickException(f"Label column '{label_col}' not found after join.")

    # y_true / proba
    y_true = joined[label_col].astype(int).to_numpy()
    proba = joined["proba"].astype(float).to_numpy()

    # Curves & metrics
    roc_auc = roc_auc_score(y_true, proba)
    ap = average_precision_score(y_true, proba)
    precision, recall, pr_thr = precision_recall_curve(y_true, proba)
    fpr, tpr, roc_thr = roc_curve(y_true, proba)

    thr = _choose_threshold(y_true, proba, optimize, threshold)
    y_pred = (proba >= thr).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec_score = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    brier = brier_score_loss(y_true, y_pred)
    try:
        ll = log_loss(y_true, np.vstack([1 - proba, proba]).T, labels=[0, 1])
    except Exception:
        ll = None

    # Save curves / joined preds
    pd.DataFrame({"recall": recall, "precision": precision, "threshold": np.r_[pr_thr, np.nan]}).to_csv(out / "pr_curve.csv", index=False)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thr}).to_csv(out / "roc_curve.csv", index=False)

    # Keep name columns if present
    keep_cols = [COL_PAIR_ID, left_col, right_col] if left_col in joined.columns and right_col in joined.columns else [COL_PAIR_ID]
    keep_cols = [c for c in keep_cols if c in joined.columns]
    joined_out = joined[keep_cols + [label_col, "proba"]].copy() if keep_cols else joined[[label_col, "proba"]].copy()
    joined_out["y_pred"] = y_pred
    joined_out.to_csv(out / "predictions_with_labels.csv", index=False)

    # Plots
    _save_individual_plots(recall, precision, ap, fpr, tpr, roc_auc, y_true, proba, out)
    _save_combined_figure(recall, precision, ap, fpr, tpr, roc_auc, y_true, proba, out / "evaluation_plots.png")
    _pr_highlight_plot(y_true, proba, target_precision, out / "pr_curve_highlight.png")

    # compute thresholds & recall for precisions 0.85→0.96
    pr_thresholds = {}
    precision_arr, recall_arr, thr_arr = precision_recall_curve(y_true, proba)
    # align so thr_arr[i] ↔ precision_arr[i+1], recall_arr[i+1]
    p_arr, r_arr = precision_arr[1:], recall_arr[1:]
    for p in np.arange(0.85, 0.97, 0.01):
        idxs = np.where(p_arr >= p)[0]
        if len(idxs):
            best = idxs[0]  # minimal threshold for fixed precision → maximal recall
            pr_thresholds[round(p, 2)] = {
                "threshold": float(thr_arr[best]),
                "recall":    float(r_arr[best])
            }
        else:
            pr_thresholds[round(p, 2)] = {"threshold": None, "recall": None}
    # save that lookup
    (out / "precision_recall_thresholds.json").write_text(
        json.dumps(pr_thresholds, indent=2), encoding="utf-8"
    )

    # Save metrics JSON
    metrics = {
        "n_samples": int(len(y_true)),
        "prevalence": float(np.mean(y_true)),
        "auroc": float(roc_auc),
        "average_precision": float(ap),
        "threshold_strategy": optimize,
        "threshold": float(thr),
        "accuracy_at_threshold": float(acc),
        "precision_at_threshold": float(prec),
        "recall_at_threshold": float(rec_score),
        "f1_at_threshold": float(f1),
        "brier_score": float(brier),
        "log_loss": (float(ll) if ll is not None else None),
        "confusion_matrix": cm,
    }
    metrics["pr_thresholds"] = pr_thresholds
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Text report + FP/FN CSVs
    report_txt = classification_report(y_true, y_pred, digits=4, zero_division=0)
    (out / "classification_report.txt").write_text(report_txt)

    df_join = joined_out.copy()
    false_pos = df_join[(df_join[label_col] == 0) & (df_join["y_pred"] == 1)]
    false_neg = df_join[(df_join[label_col] == 1) & (df_join["y_pred"] == 0)]
    false_pos.to_csv(out / "false_positives.csv", index=False)
    false_neg.to_csv(out / "false_negatives.csv", index=False)

    # Markdown report for FP/FN
    _false_negatives_positives_report(joined_out, out)

    # Markdown report with all plots
    _markdown_report(out, metrics)

    click.echo(f"[SUCCESS] Wrote evaluation artifacts to {out}")


if __name__ == "__main__":
    main()
