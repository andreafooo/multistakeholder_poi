"""
Core relevance/diversity metrics (nDCG, behavioral ILD, geographic ILD) for
baseline/platform/provider/civic, given one model's recommendation directory.

This is a lean subset of the analysis in offline_evaluation.ipynb (cell 14) --
overall per-user scores only, no HighPop/MedPop/LowPop group breakdown, no
JSD/gini/poplift. It exists to support sweep_top_k_resample.py; extend from
the notebook's fuller logic later if the group/JSD breakdown turns out to
matter for picking top_k_resample.
"""

import json
import os

import pandas as pd

from civic_reranker import load_coordinates
from evaluation_metrics import behavioral_ild_per_user, geographic_ild_per_user, ndcg
from globals import top_k_eval
from platform_reranker import open_ground_truth_user_group
from provider_reranker import build_item_similarity

METHODS = ["baseline", "platform", "provider", "civic"]


def _top_k_to_df(path):
    with open(path) as f:
        data = json.load(f)
    rows = []
    for user, items in data.items():
        if items and isinstance(items[0], list):
            items = items[0]
        for item in items:
            rows.append({"user_id:token": user, "item_id:token": item})
    return pd.DataFrame(rows)


def compute_metrics_for_run(dataset, model_name, model_dir):
    """
    model_dir: path to a timestamped recommendation directory (containing
    baseline/platform/provider/civic subdirs, each holding a
    top_k_recommendations.json -- see platform_reranker.save_top_k).

    Returns a tidy long-format DataFrame: one row per (method, metric, user).
    """
    train_data, test_data, _ = open_ground_truth_user_group(dataset)
    poi_df = load_coordinates(dataset)
    item_coords = dict(zip(poi_df["item_id:token"], zip(poi_df["lat:float"], poi_df["lon:float"])))
    item_sim_matrix, item_idx = build_item_similarity(train_data)

    rows = []
    for method in METHODS:
        json_path = (
            os.path.join(model_dir, "top_k_recommendations.json")
            if method == "baseline"
            else os.path.join(model_dir, method, "top_k_recommendations.json")
        )
        if not os.path.exists(json_path):
            print(f"  [skip] {method}: {json_path} not found")
            continue

        df = _top_k_to_df(json_path)
        metric_scores = {
            "ndcg": ndcg(test_data, df, top_k_eval),
            "ild": behavioral_ild_per_user(df, item_sim_matrix, item_idx),
            "geo_ild": geographic_ild_per_user(df, item_coords),
        }

        for metric_name, scores in metric_scores.items():
            for user_id, value in scores.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model_name,
                        "method": method,
                        "metric": metric_name,
                        "user_id": user_id,
                        "value": value,
                    }
                )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True, help="e.g. LightGCN")
    parser.add_argument("--model-dir", required=True,
                         help="path to the timestamped recommendation directory")
    args = parser.parse_args()

    df = compute_metrics_for_run(args.dataset, args.model, args.model_dir)
    print(df.groupby(["method", "metric"])["value"].agg(["mean", "count"]))
