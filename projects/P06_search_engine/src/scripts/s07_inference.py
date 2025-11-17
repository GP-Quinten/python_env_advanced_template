import os
import json
import click
import weaviate

from p06_search_engine import config, io, indexing, searching
from weaviate.classes.config import VectorDistances, Tokenization
from weaviate.classes.query import HybridFusion

def load_best_parameters(path_params: str) -> dict:
    with open(path_params, "r") as f:
        best_params = json.load(f)

    print(HybridFusion("FUSION_TYPE_RANKED"))

    def remove_none(d):
        if isinstance(d, dict):
            return {k: remove_none(v) for k, v in d.items() if v is not None}
        return d

    best_experiment_id = best_params["id"]
    best_params = {
        "preparing": {
            "name": bool(best_params["preparing"].get("name", False)), 
            "acronym": bool(best_params["preparing"].get("acronym", False)), 
            "locations": int(best_params["preparing"].get("locations", 0)), 
            "conditions": int(best_params["preparing"].get("conditions", 0)), 
        }, 
        "searching": {
            "method": best_params["searching"].get("method", "vector"), 
            "distance": VectorDistances(best_params["searching"].get("distance").split(".")[-1].lower()) if best_params["searching"].get("distance") is not None else None, 
            "tokenization": Tokenization(best_params["searching"].get("tokenization").split(".")[-1].lower()) if best_params["searching"].get("tokenization") is not None else None, 
            "alpha": float(best_params["searching"].get("alpha")) if best_params["searching"].get("alpha") is not None else None, 
            "fusion_type": HybridFusion(f"FUSION_TYPE_{best_params['searching'].get('fusion_type').split('.')[-1].upper()}") if best_params["searching"].get("fusion_type") is not None else None, 
        }, 
        "thresholding": {
            "rank": int(best_params["thresholding"].get("rank")) if best_params["thresholding"].get("rank") is not None else None, 
            "relevance": float(best_params["thresholding"].get("relevance")) if best_params["thresholding"].get("relevance") is not None else None, 
        }, 
    }
    best_params = remove_none(best_params)
    return best_experiment_id, best_params

# -------------------------------------------------
# CLI Entry Point
# -------------------------------------------------
@click.command()
@click.option("--path_queries", type=click.Path(), default=config.PATH_DATA_2ND_QUERIES)
@click.option("--path_registries", type=click.Path(), default=config.PATH_DATA_REGISTRIES)
@click.option("--path_params", type=click.Path())
@click.option("--results_json", type=click.Path())
def main(path_queries: str, path_registries: str, path_params: str, results_json: str):

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

    # ----------------------
    # LOAD PARAMS
    # ----------------------
    click.echo("\n⚙️  LOADING PARAMS")
    best_experiment_id, best_params = load_best_parameters(path_params=path_params)
    click.echo(f"  Loaded params: {best_params}")

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
        collection_name = indexing.generate_registries_collection_name(best_experiment_id)
        registries_collection = indexing.get_registries_collection(
            client=weaviate_client,
            collection_name=collection_name,
        )

        # SEARCHING
        click.echo("🔹 SEARCHING registries")
        searches = searching.search_registries_for_queries(
            queries=queries,
            collection=registries_collection,
            searching_setting=best_params["searching"], 
            thresholding_setting=best_params["thresholding"],
        )

    # ----------------------
    # SAVE RESULTS
    # ----------------------
    click.echo("\n💾 SAVING RESULTS")
    with open(results_json, "w") as f:
        json.dump(searches, f)



if __name__ == "__main__":
    main()
