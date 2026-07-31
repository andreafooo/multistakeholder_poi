import json
import os

from civic_reranker import load_coordinates
from evaluation_metrics import max_pairwise_haversine
from globals import BASE_DIR, available_datasets


def compute_and_save_max_geo_distance(dataset):
    """
    Precompute the maximum pairwise haversine distance (km) across a dataset's
    full item catalog, and save it next to the dataset's other metadata. Used
    as a fixed external bound to normalize GeoILD (see evaluation_metrics.py),
    computed once from the catalog itself rather than re-derived every time an
    evaluation notebook runs.
    """
    poi_df = load_coordinates(dataset)
    item_coords = dict(zip(poi_df["item_id:token"], zip(poi_df["lat:float"], poi_df["lon:float"])))

    max_distance_km = max_pairwise_haversine(item_coords)
    assert max_distance_km > 0, (
        f"{dataset}: max pairwise catalog distance is {max_distance_km} (<= 0) -- "
        "degenerate catalog (single point, or duplicate coordinates only)? "
        "GeoILD normalization divides by this value."
    )

    out_path = os.path.join(BASE_DIR, f"{dataset}_dataset", f"{dataset}_max_geo_distance.json")
    with open(out_path, "w") as f:
        json.dump({"max_geo_distance_km": max_distance_km}, f, indent=4)

    print(f"{dataset}: max pairwise catalog distance = {max_distance_km:.2f} km -> saved to {out_path}")
    return max_distance_km


def load_max_geo_distance(dataset):
    """Load a dataset's precomputed max pairwise catalog distance (km)."""
    path = os.path.join(BASE_DIR, f"{dataset}_dataset", f"{dataset}_max_geo_distance.json")
    with open(path) as f:
        return json.load(f)["max_geo_distance_km"]


def main():
    for dataset in available_datasets:
        compute_and_save_max_geo_distance(dataset)


if __name__ == "__main__":
    main()
