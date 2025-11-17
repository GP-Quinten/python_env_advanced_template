import os
import click
import mlflow
import weaviate
import json
import time
from tqdm import tqdm
from joblib import Parallel, delayed
from weaviate.exceptions import WeaviateConnectionError

from p06_search_engine import (
    config, io, preparing, indexing, searching, assessing, thresholding
)


# -------------------------------------------------
# Utilities
# -------------------------------------------------
def generate_setting_id(preparing_setting_id: str, searching_setting_id: str) -> str:
    """Generate a unique setting ID for a run by combining preparing and searching IDs."""
    return f"{preparing_setting_id}___{searching_setting_id}"

def clean_mlflow_experiment(experiment_name: str):
    """Delete all runs in the given MLflow experiment (if it exists)."""
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        click.echo(f"ℹ️ MLflow experiment '{experiment_name}' does not exist — nothing to clean")
        return
    exp_id = exp.experiment_id
    try:
        runs = client.search_runs([exp_id], filter_string="")
    except Exception as e:
        click.echo(f"⚠️ Failed to list runs for experiment '{experiment_name}': {e}")
        return
    deleted = 0
    for r in runs:
        try:
            client.delete_run(r.info.run_id)
            deleted += 1
        except Exception as e:
            click.echo(f"⚠️ Failed to delete run {getattr(r, 'info', {}).run_id if r else 'unknown'}: {e}")
    click.echo(f"🧹 Cleaned {deleted} runs from MLflow experiment '{experiment_name}'")


# NEW: small helper to make objects JSON / MLflow friendly
def _to_json_serializable(obj):
    """Convert numpy/pandas scalar types and common containers to native Python types for JSON/MLflow."""
    # basic native types
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    # numpy / pandas scalars have .item()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    # dict / mapping
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    # list/tuple -> list
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    # fallback: try cast to float, else string
    try:
        return float(obj)
    except Exception:
        return str(obj)


# NEW: probe for local Weaviate availability with retries
def check_weaviate_available(retries: int = 3, delay: float = 1.0) -> bool:
    """Return True if a local Weaviate instance is reachable and ready.

    Tries to connect up to `retries` times, sleeping `delay` seconds between attempts.
    Returns False if connection or readiness checks fail.
    """
    for attempt in range(1, retries + 1):
        try:
            with weaviate.connect_to_local() as client:
                if client.is_ready():
                    return True
        except Exception:
            # swallow and retry
            if attempt < retries:
                time.sleep(delay)
    return False


# -------------------------------------------------
# Single Run
# -------------------------------------------------
def main_single_run(
    queries: list[dict],
    registries: list[dict],
    annotations: list[dict],
    preparing_setting: dict,
    searching_setting: dict,
    setting_id: str,
    output_dir: str,
    thresholding_n_ranks: int,
    thresholding_n_relevances: int,
    reuse: bool,
):
    """Run the full pipeline once: preparing → indexing → searching → assessing → thresholding."""

    # check for reuse of existing search results
    searches_dir = os.path.join(output_dir, "searches")
    search_file = os.path.join(searches_dir, f"{setting_id}.json")
    searches = None
    click.echo(f"Running setting: {setting_id}")
    click.echo(f"Checking for existing search results at: {search_file}")
    if reuse and os.path.exists(search_file):
        click.echo(f"↪️ Reusing existing search results for {setting_id}, loading from {search_file}")
        # load searches from disk instead of executing Weaviate
        searches = io.read_searches(path_data_searches=searches_dir, setting_id=setting_id)
    else:
        # ----------------------
        # PREPARING
        # ----------------------
        click.echo("🔹 PREPARING registries")
        registries = preparing.prepare_registries(registries=registries, setting=preparing_setting)

        # ----------------------
        # WEAVIATE (INDEXING + SEARCHING)
        # ----------------------
        click.echo("🔹 Connecting to Weaviate")
        with weaviate.connect_to_local() as weaviate_client:
            # configure Mistral integration
            weaviate_client.integrations.configure(
                weaviate.classes.config.Integrations.mistral(api_key=os.environ["MISTRAL_API_KEY"])
            )

            if not weaviate_client.is_ready():
                raise RuntimeError("❌ Weaviate client is not ready")

            # INDEXING
            click.echo("🔹 INDEXING registries")
            collection_name = indexing.generate_registries_collection_name(setting_id)
            registries_collection = indexing.get_or_create_registries_collection(
                client=weaviate_client,
                collection_name=collection_name,
                searching_setting=searching_setting,
            )
            indexing.upload_registries(collection=registries_collection, registries=registries)

            # SEARCHING
            click.echo("🔹 SEARCHING registries")
            searches = searching.search_registries_for_queries(
                queries=queries,
                collection=registries_collection, 
                searching_setting=searching_setting, 
                thresholding_setting={"rank": 500},  # FIXME: make param if needed
            )


            # Ensure output directory exists before writing searches
            os.makedirs(searches_dir, exist_ok=True)

            io.write_searches(
                searches=searches,
                path_data_searches=searches_dir,
                setting_id=setting_id,
            )

    # ----------------------
    # ASSESSING
    # ----------------------
    click.echo("🔹 ASSESSING results")
    df_assessments = (
        assessing.build_df_results(searches=searches, annotations=annotations)
        .pipe(assessing.clean_df_results)
        .pipe(assessing.compute_precision_k)
        .pipe(assessing.compute_recall_k)
        .pipe(assessing.compute_f1_k)
    )
    searching_map = df_assessments.pipe(assessing.compute_global_mean_average_precision)

    # ----------------------
    # THRESHOLDING
    # ----------------------
    click.echo("🔹 THRESHOLDING")

    # prepare cache paths
    thresholds_dir = os.path.join(output_dir, "thresholding")
    threshold_file = os.path.join(thresholds_dir, f"{setting_id}.json")

    best_setting, best_metrics = {}, {"f1": 0}

    # If requested, try to reuse cached thresholding results
    if reuse and os.path.exists(threshold_file):
        click.echo(f"↪️ Reusing existing thresholding results for {setting_id}, loading from {threshold_file}")
        with open(threshold_file, "r") as fh:
            cached = json.load(fh)
        best_setting = cached.get("best_setting", {})
        best_metrics = cached.get("best_metrics", {"f1": 0})
    else:
        threshold_grid = thresholding.generate_settings_grid(
            n_ranks=thresholding_n_ranks,
            n_relevances=thresholding_n_relevances,
            relevances_values=df_assessments["search_relevance"].dropna().tolist(),
        )

        if not threshold_grid:
            click.echo("⚠️ Threshold grid is empty, skipping threshold search")
        else:
            for t_setting in tqdm(threshold_grid, desc="Thresholding grid"):
                t_metrics = (
                    df_assessments
                    .pipe(thresholding.apply_thresholding, setting=t_setting)
                    .pipe(thresholding.compute_partial_metrics)
                    .reindex([q["query_id"] for q in queries], axis=0, fill_value=0)
                    .mean()
                )
                # convert to plain dict for JSON-compatibility and comparisons
                try:
                    t_metrics = t_metrics.to_dict()
                except Exception:
                    # already a dict/serializable
                    pass

                if t_metrics.get("f1", 0) > best_metrics["f1"]:
                    best_metrics, best_setting = t_metrics, t_setting

        # persist best thresholding result
        os.makedirs(thresholds_dir, exist_ok=True)
        # ensure values are JSON-serializable (handles numpy/pandas scalars)
        payload = {"best_setting": _to_json_serializable(best_setting), "best_metrics": _to_json_serializable(best_metrics)}
        with open(threshold_file, "w") as fh:
            json.dump(payload, fh)

    # ----------------------
    # TRACKING
    # ----------------------
    click.echo("🔹 Logging to MLflow")
    mlflow.log_params({f"preparing_{k}": v for k, v in preparing_setting.items()})
    mlflow.log_params({f"searching_{k}": v for k, v in searching_setting.items()})
    mlflow.log_metric("searching_map", searching_map)
    mlflow.log_params({f"thresholding_{k}": v for k, v in best_setting.items()})
    # make sure metrics are native Python types before logging
    mlflow.log_metrics({f"thresholding_{k}": _to_json_serializable(v) for k, v in best_metrics.items()})


# NEW: top-level wrapper so joblib can start MLflow run & call main_single_run in worker
def _run_single_job(preparing_setting, searching_setting, setting_id, queries, registries, annotations, output_dir, thresholding_n_ranks, thresholding_n_relevances, reuse):
    """Wrapper run executed in parallel workers: start an MLflow run and call main_single_run."""
    # start MLflow run inside the worker so logs/metrics go to the correct run
    try:
        with mlflow.start_run(run_name=setting_id):
            main_single_run(
                queries=queries,
                registries=registries,
                annotations=annotations,
                preparing_setting=preparing_setting,
                searching_setting=searching_setting,
                setting_id=setting_id,
                output_dir=output_dir,
                thresholding_n_ranks=thresholding_n_ranks,
                thresholding_n_relevances=thresholding_n_relevances,
                reuse=reuse,
            )
    except Exception as e:
        # Log the failure to MLflow and stdout but DO NOT re-raise: prevents joblib from bubbling the exception.
        try:
            mlflow.log_param("failed_reason", str(e))
        except Exception:
            pass
        click.echo(f"❌ Job {setting_id} failed with error: {e}")
        return


# -------------------------------------------------
# Multiple Runs
# -------------------------------------------------
def main_multiple_runs(
    queries: list[dict],
    registries: list[dict],
    annotations: list[dict],
    preparing_settings_space: dict,
    searching_settings_space: dict,
    thresholding_n_ranks: int,
    thresholding_n_relevances: int,
    n_combinations: int,
    output_dir: str,
    reuse: bool,
    n_jobs: int = -1,
    weaviate_available: bool = True,  # NEW: pass availability flag
):
    """Run experiments across all combinations of preparing and searching settings."""
    preparing_settings_space_grid = preparing.generate_settings_grid(preparing_settings_space)
    searching_settings_space_grid = searching.generate_settings_grid(searching_settings_space)
    click.echo(f"    Preparing settings combinations: {len(preparing_settings_space_grid)}")
    click.echo(f"    Searching settings combinations: {len(searching_settings_space_grid)}")
    total_combinations = len(preparing_settings_space_grid) * len(searching_settings_space_grid)
    click.echo(f"    Total parameter combinations: {total_combinations}")
    click.echo(f"    n_combinations = {n_combinations if n_combinations != -1 else 'all'}")

    # Build list of tasks to run (skip finished runs) and respect n_combinations limit
    tasks = []
    count = 0
    searches_dir = os.path.join(output_dir, "searches")
    for preparing_setting in preparing_settings_space_grid:
        preparing_id = preparing.compose_setting_id(setting=preparing_setting)

        for searching_setting in searching_settings_space_grid:
            searching_id = searching.compose_setting_id(setting=searching_setting)
            setting_id = generate_setting_id(preparing_id, searching_id)

            # Skip if already finished
            if not mlflow.search_runs(filter_string=f"tags.mlflow.runName = '{setting_id}' and status = 'FINISHED'").empty:
                continue

            # If Weaviate is not available, skip runs that require live search (i.e., no cached search file and reuse is False)
            search_file = os.path.join(searches_dir, f"{setting_id}.json")
            needs_weaviate = not (reuse and os.path.exists(search_file))
            if needs_weaviate and not weaviate_available:
                click.echo(f"↪️ Skipping {setting_id} because Weaviate is not reachable and no cached searches (reuse={reuse}).")
                continue

            # limit number of runs if requested
            if n_combinations != -1 and count >= n_combinations:
                break

            tasks.append((preparing_setting, searching_setting, setting_id))
            count += 1

        if n_combinations != -1 and count >= n_combinations:
            break

    if not tasks:
        click.echo("No new parameter combinations to run (all finished, skipped due to Weaviate unavailability, or none scheduled).")
        return

    # Log how many threads will be used
    if n_jobs == -1:
        import multiprocessing
        n_threads = multiprocessing.cpu_count()
    else:
        n_threads = n_jobs
    click.echo(f"Scheduling {len(tasks)} jobs in parallel (n_jobs = {n_jobs}, threads = {n_threads})")

    if n_jobs == 1:
        # Run sequentially, no joblib
        for t in tqdm(tasks, desc="Experiment runs"):
            _run_single_job(
                t[0], t[1], t[2],
                queries, registries, annotations,
                output_dir, thresholding_n_ranks, thresholding_n_relevances, reuse
            )
    else:
        # Run in parallel with joblib
        list(tqdm(
            Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_run_single_job)(
                    t[0], t[1], t[2],
                    queries, registries, annotations,
                    output_dir, thresholding_n_ranks, thresholding_n_relevances, reuse
                ) for t in tasks
            ),
            desc="Experiment runs",
            total=len(tasks)
        ))


# -------------------------------------------------
# CLI Entry Point
# -------------------------------------------------
@click.command()
@click.option("--path_queries", type=click.Path())
@click.option("--path_registries", type=click.Path())
@click.option("--path_annotations", type=click.Path())
@click.option("--thresholding_n_ranks", type=int, help="Number of top-k ranks for thresholding")
@click.option("--thresholding_n_relevances", type=int, help="Number of relevances for thresholding")
@click.option("--output_dir", type=click.Path(), help="Output dir")
@click.option("--n_combinations", type=int, default=-1, help="Number of combinations to run; -1 for all")
@click.option("--reuse/--no-reuse", default=False, help="Skip runs if search results already exist")
@click.option("--n_jobs", type=int, default=-1, help="Number of parallel jobs for benchmarking (-1 uses all cores)")  # NEW
def main(path_queries, path_registries, path_annotations, thresholding_n_ranks, thresholding_n_relevances, output_dir, n_combinations, reuse, n_jobs):
    """CLI entry point for benchmarking search engines."""

    # ----------------------
    # TRACEABILITY
    # ----------------------
    click.echo("📂 INPUT FILES")
    click.echo(f"  Queries: {path_queries}")
    click.echo(f"  Registries: {path_registries}")
    click.echo(f"  Annotations: {path_annotations}")
    click.echo("\n⚙️ PARAMETERS")
    click.echo(f"  thresholding_n_ranks = {thresholding_n_ranks}")
    click.echo(f"  thresholding_n_relevances = {thresholding_n_relevances}")

    # ----------------------
    # BUILD SETTINGS SPACES
    # ----------------------
    preparing_settings_space = config.PREPARING_SETTINGS_SPACE
    searching_settings_space = config.SEARCHING_SETTINGS_SPACE

    # print settings space summary
    click.echo("\n🧩 SETTINGS SPACES")
    click.echo("  Preparing settings space:")
    for k, v in preparing_settings_space.items():
        click.echo(f"    {k}: {v}")
    click.echo("  Searching settings space:")
    for m, s in searching_settings_space.items():
        click.echo(f"    {m}:")
        for k, v in s.items():
            click.echo(f"      {k}: {v}")

    # ----------------------
    # LOAD DATA
    # ----------------------
    click.echo("\n📥 LOADING DATA")
    queries = io.read_queries(path_data_queries=path_queries)
    click.echo(f"  Loaded {len(queries)} queries")

    registries = io.read_registries(path_data_registries=path_registries)
    click.echo(f"  Registries (before filtering): {len(registries)}")

    registries = [r for r in registries if r.get("registry_occurrences") is None or r.get("registry_occurrences", 0) >= 3]
    click.echo(f"  Registries (after filtering): {len(registries)} (EMA or ≥3 occurrences)")

    annotations = io.read_annotations(path_data_annotations=path_annotations)
    click.echo(f"  Loaded {len(annotations)} annotations")

    # ----------------------
    # CHECK WEAVIATE AVAILABILITY
    # ----------------------
    click.echo("\n🔎 Checking Weaviate availability")
    weaviate_available = check_weaviate_available()
    if weaviate_available:
        click.echo("  ✅ Weaviate appears reachable")
    else:
        click.echo("  ⚠️ Weaviate is not reachable; runs requiring live indexing/searching will be skipped (cached searches still used if --reuse)")

    # ----------------------
    # RUN EXPERIMENTS
    # ----------------------
    click.echo("\n🚀 STARTING EXPERIMENTS")
    main_multiple_runs(
        queries=queries,
        registries=registries,
        annotations=annotations,
        preparing_settings_space=preparing_settings_space,
        searching_settings_space=searching_settings_space,
        thresholding_n_ranks=thresholding_n_ranks,
        thresholding_n_relevances=thresholding_n_relevances,
        n_combinations=n_combinations,
        output_dir=output_dir,
        reuse=reuse,
        n_jobs=n_jobs,  # NEW: pass n_jobs to main_multiple_runs
        weaviate_available=weaviate_available,  # NEW
    )
 
    # ----------------------
    # RETRIEVE BEST PARAMS
    # ----------------------
    click.echo("\n🏆 RETRIEVING BEST PARAMS")

    df_experiments = mlflow.search_runs(filter_string="status = 'FINISHED'")
    df_experiments = df_experiments[[
        "tags.mlflow.runName", 
        *df_experiments.columns[df_experiments.columns.str.startswith("params.")], 
        *df_experiments.columns[df_experiments.columns.str.startswith("metrics.")], 
    ]]
    df_experiments = df_experiments.sort_values(by=f"metrics.{config.METRICS_FOR_BEST_PARAMS}", ascending=False)

    best_experiment = df_experiments.iloc[0].dropna().to_dict()

    best_params = {
        "id": best_experiment["tags.mlflow.runName"], 
        "preparing": {
            "_".join(k.split("_")[1:]):v for k, v in best_experiment.items() if k.startswith("params.preparing_")
        },
        "searching": {
            "_".join(k.split("_")[1:]):v for k, v in best_experiment.items() if k.startswith("params.searching_")
        },
        "thresholding": {
            "_".join(k.split("_")[1:]):v for k, v in best_experiment.items() if k.startswith("params.thresholding_")
        }
    }

    click.echo(f"  Best params (by {config.METRICS_FOR_BEST_PARAMS}):")
    click.echo(json.dumps(best_params, indent=4))

    # ----------------------
    # SAVE best_params.json
    # ----------------------
    os.makedirs(output_dir, exist_ok=True)
    best_params_path = os.path.join(output_dir, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=4)
    click.echo(f"Saved best_params.json to {best_params_path}")


if __name__ == "__main__":
    # ----------------------
    # INIT MLFLOW
    # ----------------------
    mlflow.set_tracking_uri(config.MLFLOW_URI)
    # Clean existing runs in the experiment before starting new benchmarking runs
    clean_mlflow_experiment(config.MLFLOW_EXPERIMENT_NAME)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
 
    # ----------------------
    # RUN CLI
    # ----------------------
    main()
