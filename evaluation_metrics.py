import numpy as np
import math
from math import radians, sin, cos, sqrt, atan2, log2
from collections import Counter


def ndcg(test_data, df, top_k_eval=10):
    test_data = test_data.copy()  # Prevent modifying the original data
    ndcg_scores = {}

    for user_id in df["user_id:token"].unique():
        user_recommendations = df[df["user_id:token"] == user_id]
        recommended_items = user_recommendations["item_id:token"].tolist()[:top_k_eval]

        true_items = test_data[test_data["user_id:token"] == user_id][
            "item_id:token"
        ].values
        true_relevance = [1 if item in true_items else 0 for item in recommended_items]

        # Compute DCG@k
        dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(true_relevance))

        # Compute iDCG@k
        idcg = sum(
            1 / np.log2(idx + 2) for idx in range(min(len(true_items), top_k_eval))
        )

        ndcg_scores[user_id] = dcg / idcg if idcg > 0 else 0

    return ndcg_scores


def calculate_arp_poplift(
    df, item_popularity, user_profile_popularity, valid_popularity
):
    # df = df.merge(item_popularity, on="item_id:token", how="left")
    df = df.merge(user_profile_popularity, on="user_id:token", how="left")

    arp_scores = df.groupby("user_id:token")[valid_popularity].mean().to_dict()
    upts_scores = df.groupby("user_id:token")["upts"].mean().to_dict()

    # Calculate poplift as the percentage deviation (ARP - UPP) / UPP for each user
    poplift_scores = {
        user_id: ((arp_scores[user_id] - upts_scores[user_id]) / upts_scores[user_id])
        for user_id in arp_scores
        if upts_scores[user_id] != 0
    }

    return arp_scores, poplift_scores


def calculate_deltas(
    test_data,
    base_df,
    calibrated_df,
    item_popularity,
    user_profile_popularity,
    valid_popularity,
    top_k_eval,
):
    base_ndcg_scores = ndcg(test_data, base_df, top_k_eval)
    calibrated_ndcg_scores = ndcg(test_data, calibrated_df, top_k_eval)

    base_arp_scores, base_poplift_scores = calculate_arp_poplift(
        base_df, item_popularity, user_profile_popularity, valid_popularity
    )
    calibrated_arp_scores, calibrated_poplift_scores = calculate_arp_poplift(
        calibrated_df, item_popularity, user_profile_popularity, valid_popularity
    )

    return (
        base_arp_scores,
        base_poplift_scores,
        calibrated_arp_scores,
        calibrated_poplift_scores,
        base_ndcg_scores,
        calibrated_ndcg_scores,
    )


def jensen_shannon(profile_ratios, recommended_ratios):
    """
    Computes the Jensen-Shannon divergence for the given recommendations and user profile.
    """
    epsilon = 1e-8  # Small non-zero value

    # Compute JS divergence
    A = 0
    B = 0
    for c in ["h_ratio", "m_ratio", "t_ratio"]:
        profile_ratio = profile_ratios[c]
        recommended_ratio = recommended_ratios[c]

        if profile_ratio == 0:
            profile_ratio += epsilon

        if recommended_ratio == 0:
            recommended_ratio += epsilon

        A += profile_ratio * log2(
            (2 * profile_ratio) / (profile_ratio + recommended_ratio)
        )
        B += recommended_ratio * log2(
            (2 * recommended_ratio) / (profile_ratio + recommended_ratio)
        )

    js = (A + B) / 2

    return js



def gini_index(item_ids, num_items):
    """
    Computes the Gini index over item exposure distribution.
    0 = perfectly equal exposure, 1 = one item gets all exposure.
    
    Args:
        item_ids: flat list of recommended item ids (with repetition)
        num_items: total number of unique items in the catalog
    """
    counts = list(Counter(item_ids).values())
    counts += [0] * (num_items - len(counts))  # unobserved items get 0
    counts.sort()                               # ascending order required

    M = num_items
    total = sum(counts)

    if total == 0 or M == 0:
        return 0.0

    gini = (M + 1 - 2 * sum((M - k) * c / total for k, c in enumerate(counts))) / M
    return gini

def haversine(lat1, lon1, lat2, lon2):
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


def behavioral_ild_per_user(df, item_similarity_matrix, item_index):
    """
    Computes behavioral ILD per user.
    Returns dict {user_id: ild_score}
    """
    scores = {}
    for user_id, group in df.groupby("user_id:token"):
        items = group["item_id:token"].tolist()
        indices = [item_index[i] for i in items if i in item_index]
        if len(indices) < 2:
            scores[user_id] = 0.0
            continue
        distances = [
            1 - item_similarity_matrix[indices[a], indices[b]]
            for a in range(len(indices))
            for b in range(a + 1, len(indices))
        ]
        scores[user_id] = float(np.mean(distances))
    return scores


def geographic_ild_per_user(df, item_coords, warn_threshold_m=1.0):
    """
    Computes geographic ILD (all-pairs haversine) per user.
    Prints pairs with distance below warn_threshold_m (in meters).
    """
    scores = {}
    for user_id, group in df.groupby("user_id:token"):
        items = group["item_id:token"].tolist()
        coords = [(i, item_coords[i]) for i in items if i in item_coords]
        
        if len(coords) < 2:
            scores[user_id] = 0.0
            continue

        distances = []
        for a in range(len(coords)):
            for b in range(a + 1, len(coords)):
                item_a, (lat1, lon1) = coords[a]
                item_b, (lat2, lon2) = coords[b]
                dist_km = haversine(lat1, lon1, lat2, lon2)
                dist_m  = dist_km * 1000

                if dist_m < warn_threshold_m:
                    print(
                        f"[GEO-ILD WARNING] user={user_id} | "
                        f"items=({item_a}, {item_b}) | "
                        f"dist={dist_m:.4f}m | "
                        f"coords=({lat1},{lon1}) vs ({lat2},{lon2})"
                    )

                distances.append(dist_km)

        scores[user_id] = float(np.mean(distances))
    return scores


def distance_traveled_per_user(df, item_coords):
    """
    Computes sequential distance traveled per user (sum of consecutive haversine distances).
    Returns dict {user_id: dist_traveled}
    """
    scores = {}
    for user_id, group in df.groupby("user_id:token"):
        items = group["item_id:token"].tolist()
        coords = [item_coords[i] for i in items if i in item_coords]
        if len(coords) < 2:
            scores[user_id] = 0.0
            continue
        scores[user_id] = float(sum(
            haversine(*coords[k], *coords[k + 1])
            for k in range(len(coords) - 1)
        ))
    return scores


def evaluation_user_group_means(
    per_user,         # dict {metric_name: {user_id: score}}
    user_groups,
    top_k_df,
    total_catalog_size=None,
):
    """
    per_user: {
        "ndcg":           {user_id: score},
        "arp":            {user_id: score},
        "poplift":        {user_id: score},
        "behavioral_ild": {user_id: score},
        "geo_ild":        {user_id: score},
        "dist_traveled":  {user_id: score},
    }
    """
    group_means = {}
    per_user_by_group = {metric: {} for metric in per_user}

    for group_name, user_ids in user_groups.items():

        # filter each metric to this group's users
        for metric, scores in per_user.items():
            per_user_by_group[metric][group_name] = {
                u: scores[u] for u in user_ids if u in scores
            }

        # group-level aggregates
        def _group_mean(metric):
            vals = list(per_user_by_group[metric][group_name].values())
            return float(np.mean(vals)) if vals else None

        group_top_k_df = top_k_df[top_k_df["user_id:token"].isin(user_ids)]
        flattened_item_ids = group_top_k_df["item_id:token"].values.tolist()
        num_items = total_catalog_size or group_top_k_df["item_id:token"].nunique()

        group_means[group_name] = {
            metric: _group_mean(metric) for metric in per_user
        }
        group_means[group_name]["gini"] = gini_index(flattened_item_ids, num_items)

    return group_means, per_user_by_group