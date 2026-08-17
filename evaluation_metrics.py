import numpy as np
from math import radians, sin, cos, sqrt, atan2, log2
from collections import Counter
from rbo import RankingSimilarity


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


def jensen_shannon_per_user(user_profiles, recs_df, item_pop_col="item_pop_group"):
    """
    Per-user Jensen-Shannon divergence between a user's own popularity profile
    (h/m/t ratio of their training check-ins) and their own delivered top-k list,
    as opposed to the group-level `jensen_shannon` above which compares a whole
    group's aggregate ratios. `user_profiles` is the per-user ratio table from
    platform_reranker.calculate_user_popularity_distributions (has h_ratio/
    m_ratio/t_ratio); `recs_df` is the delivered top-k, already merged with
    item_popularity so it carries `item_pop_col`.

    Returns {user_id: jsd_score}.
    """
    profiles = user_profiles.set_index("user_id:token")[["h_ratio", "m_ratio", "t_ratio"]]

    scores = {}
    for user_id, group in recs_df.groupby("user_id:token"):
        if user_id not in profiles.index:
            continue
        profile_ratios = profiles.loc[user_id].to_dict()
        rec_counts = group[item_pop_col].value_counts(normalize=True)
        recommended_ratios = {f"{g}_ratio": rec_counts.get(g, 0.0) for g in ("h", "m", "t")}
        scores[user_id] = jensen_shannon(profile_ratios, recommended_ratios)
    return scores


def per_user_distribution_stats(per_user_by_group):
    """
    Median/std/min/max per metric per group, complementing the group means already
    produced by evaluation_user_group_means -- gives the shape of the per-user
    distribution (e.g. a long worst-case tail hidden behind a fine-looking mean),
    not just its center.

    per_user_by_group: {metric: {group_name: {user_id: score}}}, as returned by
    evaluation_user_group_means.
    Returns {group_name: {metric: {"median", "std", "min", "max"}}}.
    """
    stats = {}
    for metric, by_group in per_user_by_group.items():
        for group_name, scores in by_group.items():
            vals = list(scores.values())
            if not vals:
                continue
            stats.setdefault(group_name, {})[metric] = {
                "median": float(np.median(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
    return stats


def rank_biased_overlap(reference_list, candidate_list, k=None):
    """
    Rank-biased overlap (Webber et al.) between a candidate list and a
    reference list -- top-weighted, used for m_i.
    """
    return RankingSimilarity(list(reference_list), list(candidate_list)).rbo(k=k)


def nash_social_welfare(scores):
    """
    Geometric mean of per-stakeholder scores (e.g. RBO agreement scores).
    Unlike an arithmetic mean, this collapses to 0 the moment any single
    stakeholder is fully unserved, and among non-zero allocations it still
    rewards balance over a high score for one stakeholder bought at another's
    expense -- the standard Nash Social Welfare property, applied here to
    each aggregation mechanism's per-stakeholder RBO agreement instead of
    utilities.
    """
    vals = [float(v) for v in scores]
    if not vals or any(v <= 0 for v in vals):
        return 0.0
    return float(np.exp(np.mean(np.log(vals))))


def fairness_l2(scores, weights=None):
    """
    Compromise-programming L2 distance from the ideal point (every agent
    fully served, s_i=1) over per-agent achievement scores `scores`
    (each in [0,1], e.g. RBO agreement with a fairness agent's own list --
    see unified_fairness_scores). With weights summing to 1 (uniform by
    default), each (1-s_i) deviation is itself in [0,1], so the result is
    bounded to [0,1] by construction -- no separate normalization step.
    Lower is better (0 = every agent fully served); unlike NSW this is
    compensatory, so a mechanism can offset a weak agent with strong ones.
    """
    vals = [float(v) for v in scores]
    if not vals:
        return 0.0
    if weights is None:
        weights = [1.0 / len(vals)] * len(vals)
    return float(np.sqrt(sum(w * (1 - v) ** 2 for w, v in zip(weights, vals))))


def fairness_chebyshev(scores, weights=None):
    """
    Compromise-programming L-infinity (Rawlsian worst-case) distance from
    the ideal point -- same inputs and [0,1] bound as fairness_l2, but
    driven entirely by whichever single agent is worst served, ignoring how
    well the others do. Lower is better (0 = every agent fully served).
    """
    vals = [float(v) for v in scores]
    if not vals:
        return 0.0
    if weights is None:
        weights = [1.0 / len(vals)] * len(vals)
    return float(max(w * (1 - v) for w, v in zip(weights, vals)))


def unified_fairness_nsw(delivered_list, stakeholder_lists, k=None):
    """
    Single-user unified fairness score for one aggregation mechanism's
    delivered list, treating each fairness stakeholder's own re-ranker
    output as that stakeholder's ground truth (same convention as
    FairnessTracker.update in dynamic_allocation.py / social_choice_aggregation.py's
    agreement_scores: RBO(stakeholder_list, delivered_list)).

    `stakeholder_lists` is {agent_name: item_list}, one entry per fairness
    agent (e.g. globals.fairness_agents == ["platform", "civic", "provider"]).

    Returns (rbo_scores, nsw) where rbo_scores is {agent: rbo} and nsw is
    their Nash Social Welfare -- a single number that can't be gamed by
    satisfying two stakeholders at a third's expense.
    """
    rbo_scores = {
        agent: rank_biased_overlap(agent_list, delivered_list, k=k)
        for agent, agent_list in stakeholder_lists.items()
        if agent_list and delivered_list
    }
    return rbo_scores, nash_social_welfare(rbo_scores.values())


def unified_fairness_metric_scores(ild, geo_ild, jsd, geo_ild_ideal, weights=None):
    """
    Achievement scores s_i in [0,1] (1=ideal) for the three raw per-mechanism
    metrics that anchor the three fairness agents -- ILD (Provider), GeoILD
    (Civic), JSD (Platform) -- collapsed via compromise programming into
    Fairness_L2 (compensatory) and Fairness_Chebyshev (Rawlsian worst-case).
    Unlike unified_fairness_nsw (built on RBO rank-agreement with each
    stakeholder's own list), this measures how close each mechanism's own
    metric VALUE comes to that metric's achievable ideal.

    s_ild:      ILD itself -- already bounded [0,1] (cosine similarity over
                non-negative vectors), higher=better, ideal=1, no anchor needed.
    s_jsd:      1 - JSD -- bounded [0,1] since jensen_shannon() is log2-based
                (max divergence = 1), lower=better raw, ideal=0, no anchor needed.
    s_geo_ild:  min(1, geo_ild_ideal / geo_ild) -- GeoILD has no theoretical
                bound (raw km), so this is a RATIO against `geo_ild_ideal`
                (the Civic specialist's own empirically-best/lowest GeoILD)
                rather than a min-max normalization: no separate worst-case
                anchor is needed, at the cost of an asymptotic (not linear)
                penalty as geo_ild grows past the ideal. geo_ild <= 0 maps to
                s=1 (degenerate single-item lists, guarded to avoid a division
                by zero).

    Returns {"s_ild", "s_geo_ild", "s_jsd", "fairness_l2", "fairness_chebyshev"}.
    """
    s_ild = float(np.clip(ild, 0.0, 1.0))
    s_jsd = float(np.clip(1.0 - jsd, 0.0, 1.0))
    s_geo_ild = 1.0 if geo_ild <= 0 else float(min(1.0, geo_ild_ideal / geo_ild))
    vals = [s_ild, s_geo_ild, s_jsd]
    return {
        "s_ild": s_ild,
        "s_geo_ild": s_geo_ild,
        "s_jsd": s_jsd,
        "fairness_l2": fairness_l2(vals, weights=weights),
        "fairness_chebyshev": fairness_chebyshev(vals, weights=weights),
    }


def max_pairwise_haversine(item_coords):
    """
    Maximum pairwise great-circle distance (km) among all items in item_coords.
    Used as a fixed, condition-independent normalization ceiling for GeoILD --
    unlike JSD (bounded [0,1] by construction, log2-based) or ILD (bounded [0,1]
    when built on cosine similarity over non-negative vectors), geographic
    distance has no universal theoretical bound; the ceiling depends on this
    dataset's own geographic extent, computed once from the catalog itself
    rather than from whichever conditions happen to be in a comparison table.
    """
    coords = np.array(list(item_coords.values()), dtype=float)
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])

    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    R = 6371.0  # Earth radius in km, matches haversine()
    return float((R * c).max())


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
                # dist_m  = dist_km * 1000

                # if dist_m < warn_threshold_m:
                    # print(
                    #     f"[GEO-ILD WARNING] user={user_id} | "
                    #     f"items=({item_a}, {item_b}) | "
                    #     f"dist={dist_m:.4f}m | "
                    #     f"coords=({lat1},{lon1}) vs ({lat2},{lon2})"
                    # )

                distances.append(dist_km)

        scores[user_id] = float(np.mean(distances))
    return scores


def geographic_ild_per_user_network(df, item_coords, osrm_client):
    """
    Same as geographic_ild_per_user, but real travel distance (OSRM) instead
    of haversine. One /table request per user (all-pairs in a single round
    trip, via osrm_client.table_km -- the same batching civic_reranker.py's
    GeoReranker._prewarm uses) rather than a /route call per pair: a user's
    top-k is well under OSRM_TABLE_MAX_COORDS, so this is one HTTP call
    instead of up to C(k,2), and the disk cache means repeat item pairs
    across users/conditions cost nothing after the first lookup.
    """
    scores = {}
    for user_id, group in df.groupby("user_id:token"):
        items = group["item_id:token"].tolist()
        items_with_coords = [(i, *item_coords[i]) for i in items if i in item_coords]

        if len(items_with_coords) < 2:
            scores[user_id] = 0.0
            continue

        table = osrm_client.table_km(items_with_coords, haversine)
        distances = [
            table[(items_with_coords[a][0], items_with_coords[b][0])]
            for a in range(len(items_with_coords))
            for b in range(a + 1, len(items_with_coords))
        ]
        scores[user_id] = float(np.mean(distances))
    return scores


def geo_distance_std_per_user(df, item_coords):
    """
    Computes the standard deviation of all-pairs haversine distances (km)
    between a user's reviewed businesses. Returns dict {user_id: std_km}.
    """
    scores = {}
    for user_id, group in df.groupby("user_id:token"):
        items = group["item_id:token"].tolist()
        coords = [item_coords[i] for i in items if i in item_coords]

        if len(coords) < 2:
            scores[user_id] = 0.0
            continue

        distances = [
            haversine(*coords[a], *coords[b])
            for a in range(len(coords))
            for b in range(a + 1, len(coords))
        ]
        scores[user_id] = float(np.std(distances))
    return scores


def geo_distance_max_per_user(df, item_coords):
    """
    Computes the maximum all-pairs haversine distance (km) between a user's
    reviewed businesses. Returns dict {user_id: max_km}.
    """
    scores = {}
    for user_id, group in df.groupby("user_id:token"):
        items = group["item_id:token"].tolist()
        coords = [item_coords[i] for i in items if i in item_coords]

        if len(coords) < 2:
            scores[user_id] = 0.0
            continue

        distances = [
            haversine(*coords[a], *coords[b])
            for a in range(len(coords))
            for b in range(a + 1, len(coords))
        ]
        scores[user_id] = float(np.max(distances))
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


def distance_traveled_per_user_network(df, item_coords, osrm_client):
    """
    Same as distance_traveled_per_user, but real travel distance (OSRM).
    Uses table_km per user like geographic_ild_per_user_network -- if that
    was already called for the same df, this is a pure cache hit (no
    network calls), since table_km warms every pair in a user's list, not
    just the consecutive ones this function reads back out.
    """
    scores = {}
    for user_id, group in df.groupby("user_id:token"):
        items = group["item_id:token"].tolist()
        items_with_coords = [(i, *item_coords[i]) for i in items if i in item_coords]
        if len(items_with_coords) < 2:
            scores[user_id] = 0.0
            continue
        table = osrm_client.table_km(items_with_coords, haversine)
        scores[user_id] = float(sum(
            table[(items_with_coords[k][0], items_with_coords[k + 1][0])]
            for k in range(len(items_with_coords) - 1)
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