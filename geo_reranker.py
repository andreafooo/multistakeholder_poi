import pandas as pd
import numpy as np
import os
import json
import traceback
from math import radians, sin, cos, sqrt, atan2
from tqdm import tqdm
from reranker import (
    create_base_recommendations,
    dataset_metadata,
    recommender_dir_combiner,
    save_top_k,
)
from globals import (
    BASE_DIR,
    available_datasets,
    recommendation_dirpart,
    top_k_eval,
    top_k_resample,
)


class GeoReranker:
    def __init__(self, coordinates_df):
        """
        Parameters
        ----------
        coordinates_df : pd.DataFrame
            Must contain columns: item_id:token, lat:float, lon:float
        """
        # Build lookup dict: item_id str -> (lat, lon)
        self.coords = dict(
            zip(
                coordinates_df["item_id:token"],
                zip(coordinates_df["lat:float"], coordinates_df["lon:float"]),
            )
        )

    # --- haversine distance in km ---
    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2):
        """
        Returns the great-circle distance in km between two points
        given their latitude and longitude in decimal degrees.
        """
        R = 6371.0  # Earth radius in km

        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))

    def _distance(self, item_a, item_b):
        """
        Haversine distance between two item string IDs.
        Returns inf if either item has no coordinates (cold-start safe).
        """
        coords_a = self.coords.get(item_a)
        coords_b = self.coords.get(item_b)
        if coords_a is None or coords_b is None:
            return float("inf")
        return self._haversine(coords_a[0], coords_a[1], coords_b[0], coords_b[1])

    # --- core greedy nearest-neighbour traversal ---
    def _geo_select(self, candidates, relevance_scores, top_k):
        """
        Greedy geographic traversal.

        1. Anchor on the highest BPR relevance score item.
        2. From the last selected item, pick the geographically closest
           remaining candidate.
        3. Repeat until top_k items are selected.

        Parameters
        ----------
        candidates       : list of item_id strings
        relevance_scores : dict {item_id: float}
        top_k            : int

        Returns
        -------
        list of item_id strings in geo-greedy order
        """
        selected = []
        remaining = set(candidates)

        # Step 1: anchor on highest relevance score
        anchor = max(remaining, key=lambda i: relevance_scores[i])
        selected.append(anchor)
        remaining.discard(anchor)

        # Step 2+: greedy nearest neighbour from last selected
        while remaining and len(selected) < top_k:
            last = selected[-1]
            nearest = min(remaining, key=lambda i: self._distance(last, i))
            selected.append(nearest)
            remaining.discard(nearest)

        return selected

    # --- OFFLINE: full recommendation frame ---
    def rerank_all(self, recommendations_df, top_k):
        """
        Re-rank a full recommendations DataFrame geographically.

        Parameters
        ----------
        recommendations_df : pd.DataFrame
            Columns: user_id:token, item_id:token, score
        top_k : int

        Returns
        -------
        pd.DataFrame  columns: user_id:token, item_id:token, rank, score
                      sorted by user_id:token, rank
        """
        results = []

        for user_id, group in tqdm(
            recommendations_df.groupby("user_id:token"),
            desc="Geo re-ranking users",
        ):
            if group.duplicated("item_id:token").any():
                group = group.drop_duplicates("item_id:token")

            candidates = group["item_id:token"].tolist()
            scores = dict(zip(group["item_id:token"], group["score"]))

            reranked = self._geo_select(candidates, scores, top_k)

            for rank, item_id in enumerate(reranked):
                results.append(
                    {
                        "user_id:token": user_id,
                        "item_id:token": item_id,
                        "rank": rank,
                        "score": scores[item_id],
                    }
                )

        out_df = pd.DataFrame(results)
        out_df = out_df.sort_values(["user_id:token", "rank"]).reset_index(drop=True)
        return out_df

    # --- ONLINE: single user at inference time ---
    def rerank_user(self, user_candidates, relevance_scores, top_k):
        """
        Re-rank candidates for a single user on the fly.

        Parameters
        ----------
        user_candidates  : list of item_id strings
        relevance_scores : dict {item_id: float}
        top_k            : int

        Returns
        -------
        list of item_id strings in geo-greedy order
        """
        return self._geo_select(user_candidates, relevance_scores, top_k)


def load_coordinates(dataset):
    """Load POI coordinates for a dataset."""
    path = os.path.join(
        BASE_DIR,
        f"{dataset}_dataset", "processed_data_capri", "poiCoos.txt",
    )

    coords_df = pd.read_csv(path, header=None, sep="\t", names=["item_id:token", "lat:float", "lon:float"])
    coords_df["item_id:token"] = coords_df["item_id:token"].astype(str)+"_x"
    print(f"Loaded {len(coords_df)} POI coordinates for dataset '{dataset}'")
    return coords_df


def main(available_datasets):
    for dataset in tqdm(available_datasets, desc="Processing datasets"):
        data = dataset_metadata(dataset)

        # Build reranker once per dataset, reuse across all models
        coords_df = load_coordinates(dataset)
        reranker = GeoReranker(coords_df)

        for result in tqdm(
            data, desc=f"Processing models for {dataset}", leave=False
        ):
            try:
                print(
                    f"Processing model {result['model']} on dataset {result['dataset']}"
                )

                baseline_topk_dir, basedir = recommender_dir_combiner(
                    dataset, result["directory"]
                )

                base_resample, base_eval = create_base_recommendations(
                    baseline_topk_dir,
                    top_k_resample=top_k_resample,
                    top_k_eval=top_k_eval,
                )

                out_df = reranker.rerank_all(
                    recommendations_df=base_resample,
                    top_k=top_k_resample,
                )
                save_top_k(out_df, basedir, "geo")

            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)


if __name__ == "__main__":
    main(available_datasets)