import json
import click
import requests
from bs4 import BeautifulSoup
import os

BASE = "https://catalogues.ema.europa.eu"

def list_registries(query: str):
    """Search EMA registry catalogue and extract results for a query."""
    url = f"{BASE}/search"
    params = {
        "search_api_fulltext": query,
        "conjunction": "OR",
        "f[0]": "content_type:darwin_data_source",
    }
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/122.0.0.0 Safari/537.36"),
        "Accept-Language": "en-US,en;q=0.9",
    }

    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    results = []
    for art in soup.select("article.node.darwin-data-source"):
        a = art.select_one("a.article-title")
        if not a:
            continue

        title = a.get_text(strip=True)
        href = a.get("href", "")
        full_url = BASE + href if href.startswith("/") else href

        fp = art.select_one(".darwin-data-source__field-darwin-first-published time")
        first_published = fp.get("datetime") if fp else None

        last_upd = art.select_one(".form-item-update-date-field .field-content")
        last_updated = last_upd.get_text(strip=True) if last_upd else None

        countries = [c.get_text(strip=True)
                     for c in art.select(".darwin-country__field-darwin-term-short-name")]

        domain_el = art.select_one(".darwin-data-source__field-darwin-source-domain")
        domain = domain_el.get_text(strip=True) if domain_el else None

        types = [t.get_text(strip=True)
                 for t in art.select(".darwin-data-source--type .label")]

        results.append({
            "title": title,
            "url": full_url,
            "countries": countries,
            "domain": domain,
            "types": types,
            "first_published": first_published,
            "last_updated": last_updated,
        })

    return results


@click.command()
@click.option("--queries_json", required=True, type=click.Path(exists=True),
              help="Input JSON file containing queries.")
@click.option("--output_results_json", default="ema_search_results.json", show_default=True,
              help="Output JSON file to save results.")
@click.option("--output_dir", type=click.Path(), required=True,
              help="Directory to store output files and metadata.")
def main(queries_json, output_results_json, output_dir):
    """
    CLI tool to query EMA registries.
    """
    with open(queries_json, "r", encoding="utf-8") as f:
        queries = json.load(f)

    results = {}
    for entry in queries:
        query_id = entry.get("query_id")
        query_text = entry.get("query_text")
        click.echo(f"Running query: {query_id} {query_text}")
        res = list_registries(query_text)
        results[query_id] = {
            "query_text": query_text,
            "num_results": len(res),
            "results": res,
            # Optionally, include other fields from entry if needed
            "disease": entry.get("disease"),
            "demographics": entry.get("demographics"),
            "geography": entry.get("geography"),
            "treatment": entry.get("treatment"),
            "outcome": entry.get("outcome"),
            "Select Query": entry.get("Select Query"),
            "Note_GH": entry.get("Note_GH"),
        }
        click.echo(f"  Found {len(res)} results")

    os.makedirs(output_dir, exist_ok=True)
    output_results_path = os.path.join(output_dir, os.path.basename(output_results_json))
    with open(output_results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Generate metadata.json with stats
    metadata = {
        "output_file": output_results_path,
        "num_queries": len(results),
        "total_num_results": sum(r["num_results"] for r in results.values()),
        "per_query": {
            key: {
                "query_text": r["query_text"],
                "num_results": r["num_results"]
            }
            for key, r in results.items()
        }
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, indent=2, ensure_ascii=False)

    click.echo(f"\n✅ Results saved to {output_results_path}")
    click.echo(f"📊 Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
