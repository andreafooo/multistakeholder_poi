import json
import os

import pandas as pd

from civic_reranker import load_coordinates
from evaluation_metrics import behavioral_ild_per_user, geographic_ild_per_user
from globals import (
    BASE_DIR,
    available_datasets,
    fairness_agents,
    valid_popularity,
)
from provider_reranker import build_item_similarity, open_training_data


def compute_user_profile_popularity(train_df, valid_popularity=valid_popularity):
    """
    Mean item popularity of each user's own training profile -- same computation
    as `upts` in platform_reranker.rerank_upd, but kept independent of any
    particular re-ranking run.
    """
    value_counts = train_df["item_id:token"].value_counts().reset_index()
    value_counts.columns = ["item_id:token", "count"]
    value_counts[valid_popularity] = value_counts["count"] / len(value_counts)

    df = train_df.merge(
        value_counts[["item_id:token", valid_popularity]], on="item_id:token", how="left"
    )
    return df.groupby("user_id:token")[valid_popularity].mean().to_dict()


def percentile_rank_normalize(raw_scores):
    """
    Rescale a {user_id: raw_value} dict to [0, 1] via percentile rank. Outlier-robust
    and guarantees a uniform distribution (directly comparable across agents), but
    discards magnitude -- a 2km gap and a 2000km gap can move the result by the same
    amount if they occupy the same rank position.
    """
    ranks = pd.Series(raw_scores).rank(method="average", pct=True)
    return ranks.to_dict()


def compute_and_save_user_compatibility(dataset):
    """
    Precompute each user's profile-based compatibility c_i with the fairness
    agents (see dynamic_allocation.py), from their own training check-in
    history -- independent of any specific model run or delivered list:

    - "mmr":       mean behavioral ILD of the user's own profile items (same
                    item-item cosine similarity matrix as provider_reranker's
                    MMR agent). Higher = user's own taste is already diverse,
                    so pushing diversity for them is compatible.
    - "geo":       1 - mean geographic ILD of the user's own profile items
                    (same haversine approach as civic_reranker's geo agent).
                    The geo agent's push is to minimize distance travelled, so
                    a user whose own profile is already geographically compact
                    is compatible; a user who already ranges widely is not.
    - "cp_min_js": 1 - mean profile popularity. Users whose profile already
                    skews niche/long-tail are compatible with cp_min_js's
                    anti-popularity-bias calibration; mainstream-only users
                    are not.

    Each raw stat is normalized to [0, 1] across the dataset's users via
    percentile rank before being turned into a compatibility score. Saved once
    per dataset so dynamic_allocation.py can look it up per user instead of
    recomputing a proxy on every run.
    """
    train_df = open_training_data(dataset)

    item_sim, item_id_to_idx = build_item_similarity(train_df)
    ild_raw = behavioral_ild_per_user(train_df, item_sim, item_id_to_idx)

    coords_df = load_coordinates(dataset)
    item_coords = dict(
        zip(coords_df["item_id:token"], zip(coords_df["lat:float"], coords_df["lon:float"]))
    )
    geo_ild_raw = geographic_ild_per_user(train_df, item_coords)

    popularity_raw = compute_user_profile_popularity(train_df)

    ild_norm = percentile_rank_normalize(ild_raw)
    geo_ild_norm = percentile_rank_normalize(geo_ild_raw)
    popularity_norm = percentile_rank_normalize(popularity_raw)

    all_users = set(ild_raw) | set(geo_ild_raw) | set(popularity_raw)
    compatibility = {
        user_id: {
            "mmr": ild_norm.get(user_id, 0.0),
            "geo": 1.0 - geo_ild_norm.get(user_id, 0.0),
            "cp_min_js": 1.0 - popularity_norm.get(user_id, 0.0),
        }
        for user_id in all_users
    }

    assert set(fairness_agents) <= {"mmr", "geo", "cp_min_js"}, (
        f"compute_user_compatibility only knows how to score {{'mmr', 'geo', 'cp_min_js'}}, "
        f"but globals.fairness_agents is {fairness_agents}"
    )

    out_path = os.path.join(BASE_DIR, f"{dataset}_dataset", f"{dataset}_user_compatibility.json")
    with open(out_path, "w") as f:
        json.dump(compatibility, f, indent=4)

    print(f"{dataset}: saved per-user compatibility scores (percentile_rank) for {len(compatibility)} users -> {out_path}")
    return compatibility


def load_user_compatibility(dataset):
    """Load a dataset's precomputed per-user, per-agent compatibility scores."""
    path = os.path.join(BASE_DIR, f"{dataset}_dataset", f"{dataset}_user_compatibility.json")
    with open(path) as f:
        return json.load(f)


def main():
    for dataset in available_datasets:
        compute_and_save_user_compatibility(dataset)


if __name__ == "__main__":
    main()
