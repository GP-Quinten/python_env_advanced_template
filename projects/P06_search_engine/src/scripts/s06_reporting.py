import os
import json
import click
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple, Dict, List

from p06_search_engine import config, io  # unchanged import


# =========================
# MLflow setup & data load
# =========================

def setup_mlflow() -> None:
    """Configure MLflow tracking and experiment."""
    os.chdir(config.PATH_PROJECT)
    mlflow.set_tracking_uri(config.MLFLOW_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)


def load_mlflow_runs() -> pd.DataFrame:
    """Fetch finished MLflow runs and keep params/metrics/tags of interest."""
    df = mlflow.search_runs(filter_string="status = 'FINISHED'")
    if df.empty:
        raise RuntimeError("❌ No MLflow runs found.")
    print(f"✅ Found {len(df)} finished MLflow runs.")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Example row:\n{df.iloc[0].to_dict()}")

    keep_cols = ["tags.mlflow.runName"]
    keep_cols += df.columns[df.columns.str.startswith("params.")].tolist()
    keep_cols += df.columns[df.columns.str.startswith("metrics.")].tolist()

    df = (
        df[keep_cols]
        .where(lambda d: d.ne("None"))
        .astype(float, errors="ignore")
    )

    # Normalize selected param columns if present
    for col in [
        "params.searching_distance",
        "params.searching_tokenization",
        "params.searching_fusion_type",
    ]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.split(".").str[-1]
                .str.replace("_", "-", regex=False)
                .str.lower()
            )

    # Sort by MAP descending when present
    if "metrics.searching_map" in df.columns:
        df = df.sort_values("metrics.searching_map", ascending=False)

    return df


# =========================
# Evaluation dataframe
# =========================

def build_eval_df(
    df_searches: pd.DataFrame,
    df_annotations: pd.DataFrame,
    df_registries: pd.DataFrame,
    df_queries: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge inputs and compute k, y, precision@k, recall@k, and f1@k
    with safe handling of division by zero and clipping to [0, 1].
    """
    df_eval = (
        df_searches.merge(df_annotations, on=["query_id", "registry_id"], how="outer")
        .merge(df_registries, on="registry_id", how="left")
        .merge(df_queries, on="query_id", how="left")
        .sort_values(["query_id", "registry_id"])
        .assign(
            search_rank=lambda d: d["search_rank"].fillna(np.inf),
            annotation_label=lambda d: pd.Categorical(d["annotation_label"], categories=["YES", "NO"], ordered=True),
        )
    )

    df_eval = df_eval.assign(
        k=lambda d: d["search_rank"].fillna(np.inf),
        y=lambda d: d["annotation_label"] == "YES",
    )

    cum_pos = df_eval.groupby("query_id")["y"].transform("cumsum")
    total_pos = df_eval.groupby("query_id")["y"].transform("sum")

    df_eval["precision_k"] = np.where(df_eval["k"] > 0, cum_pos / df_eval["k"], 0.0)
    df_eval["recall_k"]   = np.where(total_pos > 0, cum_pos / total_pos, 0.0)

    df_eval["precision_k"] = np.clip(df_eval["precision_k"], 0.0, 1.0)
    df_eval["recall_k"]   = np.clip(df_eval["recall_k"], 0.0, 1.0)

    denom = df_eval["precision_k"] + df_eval["recall_k"]
    df_eval["f1_k"] = np.where(denom > 0, 2.0 * (df_eval["precision_k"] * df_eval["recall_k"]) / denom, 0.0)

    return df_eval


# =========================
# Plot helpers
# =========================

def _auto_left_margin(fig: plt.Figure, ax: plt.Axes, pad: float = 0.03) -> None:
    """
    Increase the left margin so y-tick labels (experiment names) are fully visible.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bboxes = [t.get_window_extent(renderer) for t in ax.get_yticklabels() if t.get_text()]
    if not bboxes:
        return
    max_width_px = max(b.width for b in bboxes)
    fig_width_px = fig.get_figwidth() * fig.dpi
    left_frac = min(max((max_width_px / fig_width_px) + pad, 0.15), 0.85)
    fig.subplots_adjust(left=left_frac)


def plot_map_by_experiment(df_runs: pd.DataFrame, output_dir: str, fname: str = "map_by_experiment.png") -> str:
    """
    Horizontal bar chart of MAP per experiment with auto left-margin for full names.
    """
    labels = df_runs["tags.mlflow.runName"].astype(str).tolist()
    values = df_runs["metrics.searching_map"].to_numpy()

    fig, ax = plt.subplots(figsize=(14, max(8, 0.8 * len(labels))))
    y = np.arange(len(labels))

    ax.barh(y, values, color="skyblue")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12, ha="right")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean Average Precision")
    ax.set_ylabel("Experiment Name", labelpad=60)
    ax.set_title("MAP performance by experiment")

    # Ensure labels fit:
    _auto_left_margin(fig, ax)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_map_by_query(df_eval: pd.DataFrame, output_dir: str, title: str, fname: str) -> str:
    s_plot = (
        df_eval.loc[df_eval["annotation_label"].eq("YES")]
        .groupby(["query_id", "query_text"])["precision_k"]
        .mean()
        .reset_index()
        .sort_values("precision_k")
    )
    if s_plot.empty:
        return ""

    plt.figure(figsize=(10, 6))
    plt.bar(s_plot["query_id"], s_plot["precision_k"], color="skyblue")
    plt.axhline(s_plot["precision_k"].mean(), color="orange", label="mean")
    plt.axhline(s_plot["precision_k"].median(), linestyle="--", color="green", label="median")
    plt.xticks(rotation=75)
    plt.ylabel("Mean Average Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_precision_recall_at_k(df_eval: pd.DataFrame, output_dir: str, title: str, fname: str) -> str:
    df_k = df_eval.loc[np.isfinite(df_eval["k"]), ["k", "precision_k", "recall_k"]].copy()
    if df_k.empty:
        return ""
    df_k = df_k.groupby("k")[["precision_k", "recall_k"]].mean().reset_index().sort_values("k").head(100)

    plt.figure(figsize=(10, 6))
    plt.plot(df_k["k"], df_k["precision_k"], label="Precision@k", color="tab:blue")
    plt.plot(df_k["k"], df_k["recall_k"], label="Recall@k", color="tab:orange")
    plt.xlabel("k")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_distance_distribution(df_eval: pd.DataFrame, output_dir: str, title: str, fname: str) -> str:
    if "search_distance" not in df_eval.columns:
        return ""
    dpos = df_eval.loc[df_eval["y"], "search_distance"].dropna()
    dneg = df_eval.loc[~df_eval["y"], "search_distance"].dropna()
    if dpos.empty and dneg.empty:
        return ""

    plt.figure(figsize=(10, 6))
    plt.hist([dpos, dneg], bins=50, density=True, alpha=0.6, label=["positive", "negative"])
    plt.xlabel("search_distance")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_rank_distribution_positive(df_eval: pd.DataFrame, output_dir: str, title: str, fname: str) -> str:
    ranks_pos = df_eval.loc[df_eval["y"], "search_rank"].replace(np.inf, np.nan).dropna()
    if ranks_pos.empty:
        return ""
    plt.figure(figsize=(10, 6))
    plt.hist(ranks_pos, bins=50, color="skyblue")
    plt.xlabel("search_rank")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path)
    plt.close()
    return out_path


# =========================
# Analytics helpers
# =========================

def summarize_map_by_query(df_eval: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    s_plot = (
        df_eval.loc[df_eval["annotation_label"].eq("YES")]
        .groupby(["query_id", "query_text"])["precision_k"]
        .mean()
        .reset_index()
        .sort_values("precision_k")
        .rename(columns={"precision_k": "map"})
    )

    n_queries = int(s_plot.shape[0])
    mean_map = float(s_plot["map"].mean()) if n_queries > 0 else 0.0
    median_map = float(s_plot["map"].median()) if n_queries > 0 else 0.0

    top5 = s_plot.sort_values("map", ascending=False).head(5)[["query_id", "query_text", "map"]].to_dict(orient="records")
    worst5 = s_plot.sort_values("map", ascending=True).head(5)[["query_id", "query_text", "map"]].to_dict(orient="records")

    summary = {
        "n_queries": n_queries,
        "mean_map": mean_map,
        "median_map": median_map,
        "top5_queries": top5,
        "worst5_queries": worst5,
    }
    return s_plot, summary


def f1_by_rank_threshold(df_eval: pd.DataFrame, output_dir: str, fname: str = "f1_by_rank.png") -> str:
    results = []
    for rank_threshold in range(1, 51):
        stats = (
            df_eval.loc[df_eval["search_rank"] <= rank_threshold]
            .drop_duplicates("query_id", keep="last")["f1_k"]
            .agg(["mean", "median"])
        )
        results.append({"rank_threshold": rank_threshold, **stats.to_dict()})
    df_results = pd.DataFrame(results)

    plt.figure(figsize=(10, 6))
    plt.plot(df_results["rank_threshold"], df_results["mean"], label="mean")
    plt.plot(df_results["rank_threshold"], df_results["median"], linestyle="--", label="median")
    plt.xlabel("Rank threshold")
    plt.ylabel("F1 score")
    plt.title("F1 by threshold on rank")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path)
    plt.close()
    return out_path


def f1_by_distance_threshold(df_eval: pd.DataFrame, output_dir: str, fname: str = "f1_by_distance.png") -> str:
    try:
        if "search_distance" not in df_eval.columns:
            return ""
        sd = df_eval["search_distance"].dropna()
        if sd.empty:
            return ""

        min_d, max_d = float(sd.min()), float(sd.max())
        thresholds = np.linspace(min_d, max_d, 50)
        dist_results = []
        for th in thresholds:
            subset = df_eval.loc[df_eval["search_distance"] <= th]
            if subset.empty:
                dist_results.append({"distance_threshold": float(th), "mean": np.nan, "median": np.nan})
                continue
            stats = subset.drop_duplicates("query_id", keep="last")["f1_k"].agg(["mean", "median"])
            dist_results.append({"distance_threshold": float(th), **stats.to_dict()})

        df_dist_results = pd.DataFrame(dist_results)
        plt.figure(figsize=(10, 6))
        plt.plot(df_dist_results["distance_threshold"], df_dist_results["mean"], label="mean")
        plt.plot(df_dist_results["distance_threshold"], df_dist_results["median"], linestyle="--", label="median")
        plt.xlabel("Distance threshold")
        plt.ylabel("F1 score")
        plt.title("F1 by threshold on distance")
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(output_dir, fname)
        plt.savefig(out_path)
        plt.close()
        return out_path
    except Exception as e:
        print(f"⚠️ Warning while computing/saving f1_by_distance plot: {e}")
        return ""


# =========================
# Per-run reporting
# =========================

def per_run_reports(
    df_runs: pd.DataFrame,
    df_queries: pd.DataFrame,
    df_registries: pd.DataFrame,
    df_annotations: pd.DataFrame,
    searches_dir: str,
    graphs_dir: str,
    reports_dir: str,
) -> None:
    for _, run_row in df_runs.iterrows():
        run_name = run_row["tags.mlflow.runName"]
        run_search_file = os.path.join(searches_dir, f"{run_name}.json")
        run_out_dir = os.path.join(graphs_dir, run_name)
        os.makedirs(run_out_dir, exist_ok=True)

        try:
            df_search_run = pd.read_json(run_search_file)
        except Exception as e:
            print(f"⚠️ Skipping run '{run_name}': cannot load {run_search_file}: {e}")
            continue

        try:
            df_eval_run = build_eval_df(df_search_run, df_annotations, df_registries, df_queries)

            # Plots
            try:
                plot_map_by_query(df_eval_run, run_out_dir, f"MAP by query ({run_name})", "map_by_query.png")
            except Exception:
                pass

            plot_precision_recall_at_k(df_eval_run, run_out_dir, f"Precision@k and Recall@k ({run_name})", "precision_recall_at_k.png")
            plot_distance_distribution(df_eval_run, run_out_dir, f"Search distance distribution ({run_name})", "distance_distribution.png")
            plot_rank_distribution_positive(df_eval_run, run_out_dir, f"Rank distribution for positive matches ({run_name})", "rank_distribution_positive.png")

            # Per-run text report
            try:
                map_by_query_run, run_summary = summarize_map_by_query(df_eval_run)
                rpt_path = os.path.join(reports_dir, f"{run_name}.txt")
                with open(rpt_path, "w") as fh:
                    fh.write(f"Run: {run_name}\n")
                    fh.write(f"MAP (mean over queries): {run_summary['mean_map']:.4f}\n")
                    fh.write(f"MAP (median over queries): {run_summary['median_map']:.4f}\n")
                    fh.write(f"Per-run graphs: {os.path.relpath(run_out_dir, os.path.dirname(reports_dir))}\n")
            except Exception:
                pass

        except Exception as e:
            print(f"⚠️ Error while processing run '{run_name}': {e}")
            continue


# =========================
# CLI
# =========================

@click.command()
@click.option("--benchmarking", type=click.Path(exists=True), required=True, help="Benchmark result JSON (dependency only).")
@click.option("--registries", "path_registries", type=click.Path(exists=True), required=True, help="Prepared registries JSON.")
@click.option("--queries", "path_queries", type=click.Path(exists=True), required=True, help="2nd-round queries JSON.")
@click.option("--annotations", "path_annotations", type=click.Path(exists=True), required=True, help="2nd-round annotations JSON.")
@click.option("--searches_dir", "path_searches", type=click.Path(exists=True), required=True, help="Dir with per-run search JSONs.")
@click.option("--output_dir", type=click.Path(), required=True, help="Directory where report + plots will be stored.")
@click.option("--report_txt", type=click.Path(), required=True, help="Formatted text report.")
@click.option("--exps_reports/--no-exps_reports", default=False, help="Save per-experiment graphs and reports.")
def main(benchmarking, path_registries, path_queries, path_annotations, path_searches, output_dir, report_txt, exps_reports):
    os.makedirs(output_dir, exist_ok=True)

    # Dedicated subfolders for per-run artifacts
    reports_dir = os.path.join(output_dir, "reports")
    graphs_dir = os.path.join(output_dir, "graphs")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    # --- Setup & load ---
    setup_mlflow()
    df_runs = load_mlflow_runs().copy()

    # Load once for reuse
    df_queries = pd.read_json(path_queries)
    df_registries = pd.read_json(path_registries)
    df_annotations = pd.read_json(path_annotations)

    # --- Optional per-run reporting ---
    if exps_reports:
        per_run_reports(
            df_runs=df_runs,
            df_queries=df_queries,
            df_registries=df_registries,
            df_annotations=df_annotations,
            searches_dir=path_searches,
            graphs_dir=graphs_dir,
            reports_dir=reports_dir,
        )

    # --- Aggregate plot: MAP by experiment (with full names) ---
    plot_map_by_experiment(df_runs, output_dir)

    # --- Threshold analysis on BEST run ---
    best_experiment = df_runs.iloc[0]["tags.mlflow.runName"]
    df_searches_best = pd.read_json(os.path.join(path_searches, f"{best_experiment}.json"))
    df_eval = build_eval_df(df_searches_best, df_annotations, df_registries, df_queries)

    # Plots on best run
    plot_map_by_query(df_eval, output_dir, "Mean Average Precision by query", "map_by_query.png")
    plot_precision_recall_at_k(df_eval, output_dir, "Precision@k and Recall@k (averaged over queries)", "precision_recall_at_k.png")
    plot_distance_distribution(df_eval, output_dir, "Search distance distribution: positive vs negative", "distance_distribution.png")
    plot_rank_distribution_positive(df_eval, output_dir, "Rank distribution for positive matches", "rank_distribution_positive.png")

    # CSV + JSON summary
    try:
        map_by_query_df, q_summary = summarize_map_by_query(df_eval)
        map_csv_path = os.path.join(output_dir, "map_by_query.csv")
        map_by_query_df.to_csv(map_csv_path, index=False)

        summary = {
            "n_experiments": len(df_runs),
            "best_experiment": best_experiment,
            **q_summary,
        }
        summary_path = os.path.join(output_dir, "report_summary.json")
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2)
    except Exception as e:
        print("⚠️ Warning while saving per-query MAP or summary:", e)
        map_csv_path, summary_path = "", ""

    # Rank & distance threshold analyses
    f1_by_rank_threshold(df_eval, output_dir, fname="f1_by_rank.png")
    f1_by_distance_threshold(df_eval, output_dir, fname="f1_by_distance.png")

    # --- Textual report ---
    best_row = df_runs.iloc[0]
    with open(report_txt, "w") as f:
        f.write("# Benchmark Report\n\n")
        f.write(f"Total experiments: {len(df_runs)}\n\n")
        f.write("## Best experiment:\n")
        f.write(f"- Name: {best_row['tags.mlflow.runName']}\n")
        f.write(f"- MAP: {best_row['metrics.searching_map']:.3f}\n\n")
        f.write("## Top 5 runs:\n")
        f.write(df_runs[["tags.mlflow.runName", "metrics.searching_map"]].head(5).to_string(index=False))
        f.write("\n\n")

        # Extended query-level section (if computed)
        try:
            f.write("## Query-level MAP summary\n")
            f.write(f"- Number of queries with annotations: {q_summary['n_queries']}\n")
            f.write(f"- Mean MAP (over queries): {q_summary['mean_map']:.3f}\n")
            f.write(f"- Median MAP (over queries): {q_summary['median_map']:.3f}\n\n")
            f.write("### Top 5 queries (query_id, MAP):\n")
            for q in q_summary["top5_queries"]:
                txt = (q["query_text"][:140] + "...") if q.get("query_text") and len(q["query_text"]) > 140 else q.get("query_text")
                f.write(f"- {q['query_id']}: map={q['map']:.3f} - {txt}\n")
            f.write("\n### Worst 5 queries (query_id, MAP):\n")
            for q in q_summary["worst5_queries"]:
                txt = (q["query_text"][:140] + "...") if q.get("query_text") and len(q["query_text"]) > 140 else q.get("query_text")
                f.write(f"- {q['query_id']}: map={q['map']:.3f} - {txt}\n")
            f.write("\n")
            if map_csv_path:
                f.write(f"Per-query MAP CSV: {os.path.basename(map_csv_path)}\n")
            if summary_path:
                f.write(f"Summary JSON: {os.path.basename(summary_path)}\n\n")
        except Exception:
            f.write("No per-query MAP summary available.\n\n")

        f.write("See plots in folder:\n")
        for file in os.listdir(output_dir):
            if file.endswith(".html") or file.endswith(".png"):
                f.write(f"- {file}\n")

    print(f"✅ Report saved to {report_txt}")
    print(f"✅ Plots and CSV/JSON saved to {output_dir}")


if __name__ == "__main__":
    main()
