"""
Empirical nadir/utopia experiment via a 4x4 weighted-Borda payoff table.

Combines the cached baseline/platform/provider/civic per-agent candidate
lists (BASE_DIR/<dataset>_dataset/recommendations/<model_dir>/<agent>/
top_k_recommendations.json) with

    combined_score[item] = 1.0 * borda_baseline[user][item]
                          + w_platform * borda_platform[user][item]
                          + w_provider * borda_provider[user][item]
                          + w_civic    * borda_civic[user][item]

where borda_agent[user][item] = N - rank(item) + 1, using only the top-N
truncation of that agent's cached list (N = N_RESAMPLE below, a smaller
depth than globals.top_k_resample=150 for this experiment -- the shared
150-length cached files are left untouched).

For each of platform/provider/civic, a single-objective Bayesian
optimization (skopt.gp_minimize) searches (w_platform, w_provider, w_civic)
-- reparameterized as w = 3 * softmax(z) over unconstrained z in R^3, so the
sum-to-3 simplex constraint holds by construction with no rejection
sampling and no floor on individual weights -- to maximize that
stakeholder's own metric. The resulting 4x4 payoff table (rows =
optimization target, columns = ndcg/platform/provider/civic) gives each
stakeholder's utopia (its diagonal entry) and empirical nadir (worst value
in its column across the other rows, baseline-only included).
"""
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.special import softmax
from skopt import gp_minimize
from skopt.space import Real
from tqdm import tqdm

from civic_reranker import load_coordinates
from evaluation_metrics import (
    behavioral_ild_per_user,
    calculate_arp_poplift,
    fairness_chebyshev,
    fairness_l2,
    geographic_ild_per_user,
    gini_index,
    jensen_shannon_per_user,
    max_pairwise_haversine,
    ndcg,
)
from globals import BASE_DIR, methods_to_aggregate, recommendation_dirpart, top_k_eval, valid_popularity
from platform_reranker import calculate_user_popularity_distributions
from postprocess_baseline_top_k import dataset_metadata
from provider_reranker import build_item_similarity
from social_choice_aggregation import (
    flatten_winners,
    get_paths_for_sc_input,
    get_sc_output_dir,
    get_user_recommendations,
    run_social_choice_for_user,
)

DATASET = "yelpphl"
MODEL = "BPR"
N_RESAMPLE = 75  # per-agent truncation depth for this experiment only
N_CALLS = 60  # BO iterations per stakeholder run (spec: ~50-100)
N_INITIAL_POINTS = 15
RANDOM_STATE = 42

OUT_DIR = os.path.join(BASE_DIR, f"{DATASET}_dataset", "payoff_experiment")


# ---------------------------------------------------------------------------
# Data loading (mirrors offline_evaluation.ipynb's setup cell, minus the
# OSRM/group-breakdown machinery this experiment doesn't need)
# ---------------------------------------------------------------------------

def find_model_dir(dataset, model, recommendation_dirpart=recommendation_dirpart):
    data = dataset_metadata(dataset, recommendation_dirpart)
    matches = [d for d in data if d["model"] == model and d.get("model_type") == "general"]
    if not matches:
        raise ValueError(f"no run found for dataset={dataset!r} model={model!r}")
    if len(matches) > 1:
        dirs = [m["directory"] for m in matches]
        raise ValueError(f"multiple runs found for dataset={dataset!r} model={model!r}: {dirs}")
    return os.path.join(BASE_DIR, f"{dataset}_dataset", recommendation_dirpart, matches[0]["directory"])


def load_agent_list(json_path):
    """user_id -> ordered item list, flattening the baseline file's [[...]] nesting."""
    with open(json_path) as f:
        data = json.load(f)
    out = {}
    for user, items in data.items():
        if isinstance(items, list) and len(items) > 0 and isinstance(items[0], list):
            items = items[0]
        out[user] = items
    return out


def borda_scores(agent_lists, n):
    """user -> {item: score}, score = n - rank + 1 over that agent's own top-n."""
    out = {}
    for user, items in agent_lists.items():
        top = items[:n]
        out[user] = {item: n - idx for idx, item in enumerate(top)}
    return out


def load_ground_truth(dataset):
    train_data = pd.read_csv(
        os.path.join(BASE_DIR, f"{dataset}_dataset", "processed_data_recbole", f"{dataset}_sample.train.inter"),
        sep="\t",
    )
    test_data = pd.read_csv(
        os.path.join(BASE_DIR, f"{dataset}_dataset", "processed_data_recbole", f"{dataset}_sample.test.inter"),
        sep="\t",
    )
    valid_data = pd.read_csv(
        os.path.join(BASE_DIR, f"{dataset}_dataset", "processed_data_recbole", f"{dataset}_sample.valid.inter"),
        sep="\t",
    )
    train_data = pd.concat([train_data, valid_data])

    value_counts = train_data["item_id:token"].value_counts().reset_index()
    value_counts.columns = ["item_id:token", "count"]
    value_counts[valid_popularity] = value_counts["count"] / len(value_counts)
    checkin_df = train_data.merge(
        value_counts[["item_id:token", valid_popularity]], on="item_id:token", how="left"
    )
    checkin_df = checkin_df.sort_values(by=valid_popularity, ascending=False)
    item_popularity = checkin_df.drop_duplicates(subset="item_id:token", keep="first")[
        ["item_id:token", valid_popularity]
    ]

    h_group = item_popularity.head(int(len(item_popularity) * 0.2)).copy()
    h_group["item_pop_group"] = "h"
    t_group = item_popularity.tail(int(len(item_popularity) * 0.2)).copy()
    t_group["item_pop_group"] = "t"
    m_group = item_popularity[
        ~item_popularity["item_id:token"].isin(h_group["item_id:token"])
        & ~item_popularity["item_id:token"].isin(t_group["item_id:token"])
    ].copy()
    m_group["item_pop_group"] = "m"
    item_popularity = pd.concat([h_group, m_group, t_group]).sort_values(by=valid_popularity, ascending=False)

    # Per-user historical popularity tendency (mean item_pop of their own
    # check-ins), the UPP denominator in poplift = (ARP - UPP) / UPP -- same
    # as `upts` in offline_evaluation.ipynb's open_ground_truth_user_group.
    upts = checkin_df.groupby("user_id:token")[valid_popularity].mean().reset_index()
    upts.columns = ["user_id:token", "upts"]

    return train_data, test_data, item_popularity, upts


def build_env(dataset=DATASET, model=MODEL, n_resample=N_RESAMPLE):
    model_dir = find_model_dir(dataset, model)

    agent_paths = {
        "baseline": os.path.join(model_dir, "baseline", "top_k_recommendations.json"),
        "platform": os.path.join(model_dir, "platform", "top_k_recommendations.json"),
        "provider": os.path.join(model_dir, "provider", "top_k_recommendations.json"),
        "civic": os.path.join(model_dir, "civic", "top_k_recommendations.json"),
    }
    agent_lists = {agent: load_agent_list(p) for agent, p in agent_paths.items()}
    agent_borda = {agent: borda_scores(lists, n_resample) for agent, lists in agent_lists.items()}

    train_data, test_data, item_popularity, upts = load_ground_truth(dataset)
    total_catalog_size = train_data["item_id:token"].nunique()

    poi_df = load_coordinates(dataset)
    item_coords = dict(zip(poi_df["item_id:token"], zip(poi_df["lat:float"], poi_df["lon:float"])))
    geo_ild_theoretical_max = max_pairwise_haversine(item_coords)

    item_sim_matrix, item_idx = build_item_similarity(train_data)

    train_data_with_pop = train_data.merge(item_popularity, on="item_id:token", how="left")
    user_profiles = calculate_user_popularity_distributions(train_data_with_pop, item_popularity)

    # Civic specialist's own per-user GeoILD at delivered (top_k_eval) length --
    # the achievement-score anchor for s_geo_ild below, same convention as
    # unified_fairness_metric_scores / the offline_evaluation.ipynb loop's
    # civic_geo_ild_ideal (built from civic's own cached list, independent of
    # this experiment's N_RESAMPLE candidate-pool truncation).
    civic_own_rows = [
        {"user_id:token": user, "item_id:token": item}
        for user, items in agent_lists["civic"].items()
        for item in items[:top_k_eval]
    ]
    civic_own_df = pd.DataFrame(civic_own_rows)
    geo_ild_ideal_scores = geographic_ild_per_user(civic_own_df, item_coords)
    geo_ild_ideal_fallback = (
        float(np.mean(list(geo_ild_ideal_scores.values()))) if geo_ild_ideal_scores else 0.0
    )

    return {
        "model_dir": model_dir,
        "agent_borda": agent_borda,
        "test_data": test_data,
        "item_popularity": item_popularity,
        "item_coords": item_coords,
        "item_sim_matrix": item_sim_matrix,
        "item_idx": item_idx,
        "user_profiles": user_profiles,
        "geo_ild_ideal_scores": geo_ild_ideal_scores,
        "geo_ild_ideal_fallback": geo_ild_ideal_fallback,
        "geo_ild_theoretical_max": geo_ild_theoretical_max,
        "user_profile_popularity": upts,
        "total_catalog_size": total_catalog_size,
    }


# ---------------------------------------------------------------------------
# Core evaluate()
# ---------------------------------------------------------------------------

def combined_top_k(w_platform, w_provider, w_civic, agent_borda, k=top_k_eval):
    weights = {"baseline": 1.0, "platform": w_platform, "provider": w_provider, "civic": w_civic}
    rows = []
    for user, base_scores in agent_borda["baseline"].items():
        candidate_items = set(base_scores.keys())
        for agent in ("platform", "provider", "civic"):
            candidate_items |= set(agent_borda[agent].get(user, {}).keys())

        scored = []
        for item in candidate_items:
            score = sum(
                weights[agent] * agent_borda[agent].get(user, {}).get(item, 0)
                for agent in weights
            )
            scored.append((item, score))
        scored.sort(key=lambda x: -x[1])

        for item, _ in scored[:k]:
            rows.append({"user_id:token": user, "item_id:token": item})
    return pd.DataFrame(rows)


def score_df(df, env):
    """Shared scoring tail for any {user_id:token, item_id:token} top-k
    dataframe -- used by evaluate() on this experiment's own recombination,
    and reused as-is (no parallel reimplementation) to score real
    social_choice_aggregation.py output and the existing static/equal-weight
    cached run for comparison."""
    ndcg_scores = ndcg(test_data=env["test_data"], df=df, top_k_eval=top_k_eval)

    df_with_pop = df.merge(env["item_popularity"], on="item_id:token", how="left")
    jsd_scores = jensen_shannon_per_user(env["user_profiles"], df_with_pop)

    ild_scores = behavioral_ild_per_user(df, env["item_sim_matrix"], env["item_idx"])
    geo_ild_scores = geographic_ild_per_user(df, env["item_coords"])

    # s_geo_ild = min(1, ideal/geo_ild): bounded achievement score anchored to
    # the civic specialist's own GeoILD (env["geo_ild_ideal_scores"]), same
    # convention as evaluation_metrics.unified_fairness_metric_scores -- caps
    # at 1 rather than rewarding unbounded geographic spread past that ideal.
    ideal_fallback = env["geo_ild_ideal_fallback"]
    ideal_scores = env["geo_ild_ideal_scores"]
    civic_scores = {
        user: 1.0 if geo_ild <= 0 else min(1.0, ideal_scores.get(user, ideal_fallback) / geo_ild)
        for user, geo_ild in geo_ild_scores.items()
    }

    return {
        "ndcg": float(np.mean(list(ndcg_scores.values()))),
        # 1 - JSD: higher = better calibration, same achievement-score
        # convention as s_jsd in evaluation_metrics.unified_fairness_metric_scores
        "platform": float(1.0 - np.mean(list(jsd_scores.values()))),
        "provider": float(np.mean(list(ild_scores.values()))),
        "civic": float(np.mean(list(civic_scores.values()))),
    }


def score_df_raw(df, env):
    """Same per-user metric calls as score_df, but reports them untransformed:
    JSD as raw divergence (lower=better, no 1-x flip) and GeoILD as raw mean
    km (lower=better, no ideal-ratio normalization) -- the "original scores"
    convention used elsewhere in this codebase (e.g. offline_evaluation.ipynb),
    as opposed to score_df's bounded/inverted achievement-score convention
    used internally for the BO objectives."""
    ndcg_scores = ndcg(test_data=env["test_data"], df=df, top_k_eval=top_k_eval)

    df_with_pop = df.merge(env["item_popularity"], on="item_id:token", how="left")
    jsd_scores = jensen_shannon_per_user(env["user_profiles"], df_with_pop)

    ild_scores = behavioral_ild_per_user(df, env["item_sim_matrix"], env["item_idx"])
    geo_ild_scores = geographic_ild_per_user(df, env["item_coords"])

    # poplift: (ARP - UPP) / UPP per user, then averaged -- 0 = recommends at
    # the same popularity level as the user's own history, >0 = towards more
    # popular items, <0 = towards the long tail.
    _, poplift_scores = calculate_arp_poplift(
        df_with_pop, env["item_popularity"], env["user_profile_popularity"], valid_popularity
    )
    # gini: catalog-level exposure inequality across all delivered
    # recommendations (not per-user) -- 0 = perfectly equal exposure, 1 = one
    # item gets everything.
    gini = gini_index(df["item_id:token"].tolist(), env["total_catalog_size"])

    return {
        "ndcg": float(np.mean(list(ndcg_scores.values()))),
        "jsd (lower better)": float(np.mean(list(jsd_scores.values()))),
        "ild (higher better)": float(np.mean(list(ild_scores.values()))),
        "geo_ild_km (lower better)": float(np.mean(list(geo_ild_scores.values()))),
        "poplift (0=neutral)": float(np.mean(list(poplift_scores.values()))),
        "gini (lower better)": float(gini),
    }


def score_df_theoretical(df, env):
    """Achievement scores against each metric's THEORETICAL [0,1] bound,
    rather than score_df's empirical/mechanism-specific achievement anchor:

    s_platform = 1 - JSD -- already bounded [0,1] by construction (JSD is
    log2-based, max divergence = 1), no anchor needed.
    s_provider = ILD itself -- already bounded [0,1] by construction (cosine
    similarity over non-negative vectors), no anchor needed.
    s_civic = 1 - geo_ild / max_pairwise_haversine -- GeoILD has no universal
    theoretical bound in raw km, so the closest thing to one is this
    catalog's own maximum achievable pairwise distance (env
    ["geo_ild_theoretical_max"]): no recommendation list drawn from this
    catalog can exceed it, so it's a fixed theoretical ceiling, unlike
    score_df's civic_geo_ild_ideal (an empirical, mechanism-specific anchor).

    1 = theoretically ideal in all three (JSD=0 / ILD=1 / GeoILD=0, i.e.
    maximally tight geographic clustering), matching the "lower raw GeoILD
    is better" convention confirmed for this codebase's civic specialist.
    """
    ndcg_scores = ndcg(test_data=env["test_data"], df=df, top_k_eval=top_k_eval)

    df_with_pop = df.merge(env["item_popularity"], on="item_id:token", how="left")
    jsd_scores = jensen_shannon_per_user(env["user_profiles"], df_with_pop)

    ild_scores = behavioral_ild_per_user(df, env["item_sim_matrix"], env["item_idx"])
    geo_ild_scores = geographic_ild_per_user(df, env["item_coords"])

    theoretical_max = env["geo_ild_theoretical_max"]
    s_geo_ild_scores = {
        user: float(np.clip(1.0 - geo_ild / theoretical_max, 0.0, 1.0))
        for user, geo_ild in geo_ild_scores.items()
    }

    s_platform = float(1.0 - np.mean(list(jsd_scores.values())))
    s_provider = float(np.mean(list(ild_scores.values())))
    s_civic = float(np.mean(list(s_geo_ild_scores.values())))
    vals = [s_platform, s_provider, s_civic]

    return {
        "ndcg": float(np.mean(list(ndcg_scores.values()))),
        "s_platform (1-jsd)": s_platform,
        "s_provider (ild)": s_provider,
        "s_civic (1-geo_ild/max)": s_civic,
        "fairness_l2": fairness_l2(vals),
        "fairness_chebyshev": fairness_chebyshev(vals),
    }


def score_df_empirical(df, env, nadir_utopia):
    """Same shape as score_df_theoretical, but rescales score_df's raw
    (platform, provider, civic) against the EMPIRICAL (nadir, utopia) from
    the payoff table (rescale_to_achievement) instead of a theoretical bound
    -- the direct empirical counterpart, row-for-row and column-for-column,
    to score_df_theoretical."""
    raw = score_df(df, env)
    s = rescale_to_empirical(raw, nadir_utopia)
    vals = s
    return {
        "ndcg": raw["ndcg"],
        "s_platform": s[0],
        "s_provider": s[1],
        "s_civic": s[2],
        "fairness_l2": fairness_l2(vals),
        "fairness_chebyshev": fairness_chebyshev(vals),
    }


def load_combined_bo_log_weights(model=MODEL, n_resample=N_RESAMPLE):
    """Unique (w_platform, w_provider, w_civic) explored during the combined
    BO's l2 and chebyshev searches (warm-start + new-call phases of both
    runs, deduplicated) -- the reference population percentile-rank
    normalization is measured against, richer than nadir/utopia's implicit
    2-point (min/max) reference."""
    log_path = os.path.join(OUT_DIR, f"combined_bo_log_{model}_N{n_resample}.json")
    with open(log_path) as f:
        data = json.load(f)
    seen = set()
    weights = []
    for run in data["runs"]:
        for entry in run["log"]:
            w = (round(entry["w_platform"], 8), round(entry["w_provider"], 8), round(entry["w_civic"], 8))
            if w not in seen:
                seen.add(w)
                weights.append(w)
    return weights


def build_percentile_reference(env, weights):
    """Raw evaluate() output (platform/provider/civic, same higher=better
    convention score_df/rescale_to_empirical already use) for every unique
    weight vector explored during the combined BO -- replayed here since the
    saved log only kept l2/chebyshev/ndcg per point, not the raw scores."""
    reference = {"platform": [], "provider": [], "civic": []}
    for w_platform, w_provider, w_civic in tqdm(weights, desc="replaying BO log for percentile reference", unit="w"):
        raw = evaluate(w_platform, w_provider, w_civic, env)
        for m in reference:
            reference[m].append(raw[m])
    return reference


def percentile_rank(value, reference_values):
    """Fraction of the reference population <= value -- 0 = worst-observed,
    1 = best-observed (or tied for it); same [0,1]/1=ideal, higher=better
    convention fairness_l2/fairness_chebyshev expect."""
    ref = np.asarray(reference_values, dtype=float)
    return float(np.mean(ref <= value))


def rescale_to_percentile(raw_scores, reference):
    order = ("platform", "provider", "civic")
    return [percentile_rank(raw_scores[m], reference[m]) for m in order]


def score_df_percentile(df, env, reference):
    """Direct percentile-rank counterpart to score_df_empirical: same raw
    score_df() input, but rescaled by rank within the explored-weight
    reference population instead of linear min-max against (nadir, utopia)."""
    raw = score_df(df, env)
    s = rescale_to_percentile(raw, reference)
    return {
        "ndcg": raw["ndcg"],
        "s_platform": s[0],
        "s_provider": s[1],
        "s_civic": s[2],
        "fairness_l2": fairness_l2(s),
        "fairness_chebyshev": fairness_chebyshev(s),
    }


def results_dict_to_df(results_dict, k=top_k_eval):
    """{user_id: [item_ids]} (or the baseline file's [[item_ids]] nesting) -> the
    same user_id:token/item_id:token dataframe shape score_df expects."""
    rows = []
    for user, items in results_dict.items():
        if isinstance(items, list) and len(items) > 0 and isinstance(items[0], list):
            items = items[0]
        for item in items[:k]:
            rows.append({"user_id:token": user, "item_id:token": item})
    return pd.DataFrame(rows)


def evaluate(w_platform, w_provider, w_civic, env):
    """Recombine cached per-agent Borda scores (no reranking) and score the result."""
    df = combined_top_k(w_platform, w_provider, w_civic, env["agent_borda"], k=top_k_eval)
    return score_df(df, env)


# ---------------------------------------------------------------------------
# Bayesian optimization
# ---------------------------------------------------------------------------

DIMENSIONS = [Real(-4.0, 4.0, name="z0"), Real(-4.0, 4.0, name="z1"), Real(-4.0, 4.0, name="z2")]


def z_to_weights(z):
    w = 3.0 * softmax(np.asarray(z))
    return float(w[0]), float(w[1]), float(w[2])


def run_bo(metric_key, env, n_calls=N_CALLS, n_initial_points=N_INITIAL_POINTS, random_state=RANDOM_STATE):
    def objective(z):
        w_platform, w_provider, w_civic = z_to_weights(z)
        return -evaluate(w_platform, w_provider, w_civic, env)[metric_key]

    result = gp_minimize(
        objective,
        DIMENSIONS,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=random_state,
    )
    w_star = z_to_weights(result.x)
    metrics_star = evaluate(*w_star, env)
    return {
        "metric_optimized": metric_key,
        "w_platform": w_star[0],
        "w_provider": w_star[1],
        "w_civic": w_star[2],
        "metrics": metrics_star,
    }


# ---------------------------------------------------------------------------
# Payoff table + nadir/utopia
# ---------------------------------------------------------------------------

def build_payoff_table(baseline_metrics, bo_results):
    rows = {"ndcg (baseline-only)": baseline_metrics}
    for r in bo_results:
        rows[r["metric_optimized"]] = r["metrics"]
    df = pd.DataFrame(rows).T[["ndcg", "platform", "provider", "civic"]]
    return df


def derive_nadir_utopia(payoff_df):
    out = {}
    row_for_metric = {
        "platform": "platform",
        "provider": "provider",
        "civic": "civic",
    }
    for metric, row_label in row_for_metric.items():
        utopia = float(payoff_df.loc[row_label, metric])
        nadir = float(payoff_df.drop(index=row_label)[metric].min())
        out[metric] = {"utopia": utopia, "nadir": nadir}
    return out


# ---------------------------------------------------------------------------
# Empirical rescaling + welfare metrics
#
# fairness_l2 / fairness_chebyshev (evaluation_metrics.py) are compromise-
# programming distances to a fixed utopia point s_i=1 -- they are not
# modified here. What changes is the input: each raw evaluate() output is
# first min-max rescaled against ITS OWN empirical (nadir, utopia) from this
# payoff table, producing an achievement score s_i' in [0,1] with 1 = this
# stakeholder's empirically best-observed value (not the unreachable
# theoretical ceiling). fairness_l2/fairness_chebyshev are then called
# unmodified on [s_platform', s_provider', s_civic'].
# ---------------------------------------------------------------------------

def rescale_to_achievement(raw_value, nadir, utopia):
    if utopia == nadir:
        return 1.0
    return float(np.clip((raw_value - nadir) / (utopia - nadir), 0.0, 1.0))


def build_welfare_table(payoff_df, nadir_utopia):
    metrics = ("platform", "provider", "civic")
    rescaled_rows = {}
    welfare_rows = {}
    for row_label, row in payoff_df.iterrows():
        s = {
            m: rescale_to_achievement(row[m], nadir_utopia[m]["nadir"], nadir_utopia[m]["utopia"])
            for m in metrics
        }
        rescaled_rows[row_label] = s
        vals = [s[m] for m in metrics]
        welfare_rows[row_label] = {
            "s_platform": s["platform"],
            "s_provider": s["provider"],
            "s_civic": s["civic"],
            "fairness_l2": fairness_l2(vals),
            "fairness_chebyshev": fairness_chebyshev(vals),
        }
    return pd.DataFrame(welfare_rows).T[
        ["s_platform", "s_provider", "s_civic", "fairness_l2", "fairness_chebyshev"]
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Building env for dataset={DATASET} model={MODEL} N_RESAMPLE={N_RESAMPLE} ...")
    env = build_env()

    t0 = time.time()
    baseline_metrics = evaluate(0.0, 0.0, 0.0, env)
    elapsed = time.time() - t0
    print(f"Single evaluate() call took {elapsed:.2f}s")
    print(f"Baseline (w=0,0,0) metrics: {baseline_metrics}")
    est_total = elapsed * 3 * N_CALLS
    print(f"Estimated total for 3 x {N_CALLS} BO calls: ~{est_total / 60:.1f} min")

    bo_results = []
    for metric_key in ("platform", "provider", "civic"):
        print(f"\nRunning BO for metric={metric_key} ...")
        t0 = time.time()
        r = run_bo(metric_key, env)
        print(f"  done in {time.time() - t0:.1f}s -- w*=({r['w_platform']:.3f}, {r['w_provider']:.3f}, {r['w_civic']:.3f})")
        print(f"  metrics at w*: {r['metrics']}")
        bo_results.append(r)

    payoff_df = build_payoff_table(baseline_metrics, bo_results)
    nadir_utopia = derive_nadir_utopia(payoff_df)

    payoff_path = os.path.join(OUT_DIR, f"payoff_table_{MODEL}_N{N_RESAMPLE}.csv")
    payoff_df.to_csv(payoff_path)
    print(f"\nSaved payoff table to {payoff_path}")
    print(payoff_df)

    nadir_path = os.path.join(OUT_DIR, f"nadir_utopia_{MODEL}_N{N_RESAMPLE}.json")
    with open(nadir_path, "w") as f:
        json.dump(nadir_utopia, f, indent=2)
    print(f"Saved nadir/utopia pairs to {nadir_path}")
    print(json.dumps(nadir_utopia, indent=2))

    welfare_df = build_welfare_table(payoff_df, nadir_utopia)
    welfare_path = os.path.join(OUT_DIR, f"welfare_table_{MODEL}_N{N_RESAMPLE}.csv")
    welfare_df.to_csv(welfare_path)
    print(f"\nSaved empirically-rescaled welfare table to {welfare_path}")
    print(welfare_df)

    weights_path = os.path.join(OUT_DIR, f"bo_weight_vectors_{MODEL}_N{N_RESAMPLE}.json")
    with open(weights_path, "w") as f:
        json.dump(
            {
                "config": {
                    "dataset": DATASET,
                    "model": MODEL,
                    "n_resample": N_RESAMPLE,
                    "n_calls": N_CALLS,
                    "n_initial_points": N_INITIAL_POINTS,
                    "random_state": RANDOM_STATE,
                },
                "baseline": {"w_platform": 0.0, "w_provider": 0.0, "w_civic": 0.0, "metrics": baseline_metrics},
                "bo_runs": bo_results,
            },
            f,
            indent=2,
        )
    print(f"Saved BO weight vectors + diagnostics to {weights_path}")


# ---------------------------------------------------------------------------
# Combined BO -- optimize the rescaled welfare metrics (L2 / Chebyshev)
# directly, rather than a single raw stakeholder metric. This is the
# deployable result: w*_l2 / w*_chebyshev are the weight vectors you'd
# actually report and plug into the headline Borda comparison, unlike the 3
# per-objective runs above, which exist only to build the payoff table /
# normalization reference points.
# ---------------------------------------------------------------------------

COMBINED_N_NEW_CALLS = 60


def rescale_to_empirical(raw_scores, nadir_utopia):
    """[s_platform', s_provider', s_civic'] -- order doesn't affect fairness_l2/
    fairness_chebyshev under their default uniform weighting, but kept fixed
    for readability/logging."""
    order = ("platform", "provider", "civic")
    return [
        rescale_to_achievement(raw_scores[m], nadir_utopia[m]["nadir"], nadir_utopia[m]["utopia"])
        for m in order
    ]


def evaluate_combined(w_platform, w_provider, w_civic, env, nadir_utopia):
    raw = evaluate(w_platform, w_provider, w_civic, env)
    s = rescale_to_empirical(raw, nadir_utopia)
    return {
        "ndcg": raw["ndcg"],
        "l2": fairness_l2(s),
        "chebyshev": fairness_chebyshev(s),
        "s_platform_prime": s[0],
        "s_provider_prime": s[1],
        "s_civic_prime": s[2],
        "raw": raw,
    }


def weights_to_z(w, eps=1e-6, z_bound=4.0):
    """Inverse of z_to_weights: softmax(log(p)) == p exactly for a
    probability vector p, so this seeds gp_minimize's x0 at (an
    approximation of, once clipped into the search bounds) the given w
    without needing to search for it."""
    p = np.clip(np.asarray(w, dtype=float) / 3.0, eps, None)
    p = p / p.sum()
    z = np.clip(np.log(p), -z_bound, z_bound)
    return z.tolist()


def load_warm_start_points(model=MODEL, n_resample=N_RESAMPLE):
    """Baseline + the 3 per-objective BO optima, with their already-computed
    raw evaluate() output -- reused here without calling evaluate() again."""
    weights_path = os.path.join(OUT_DIR, f"bo_weight_vectors_{model}_N{n_resample}.json")
    with open(weights_path) as f:
        data = json.load(f)

    points = [{
        "label": "baseline",
        "w": (data["baseline"]["w_platform"], data["baseline"]["w_provider"], data["baseline"]["w_civic"]),
        "raw": data["baseline"]["metrics"],
    }]
    for r in data["bo_runs"]:
        points.append({
            "label": r["metric_optimized"],
            "w": (r["w_platform"], r["w_provider"], r["w_civic"]),
            "raw": r["metrics"],
        })
    return points


def run_combined_bo(objective_key, env, nadir_utopia, warm_start_points,
                     n_new_calls=COMBINED_N_NEW_CALLS, random_state=RANDOM_STATE):
    assert objective_key in ("l2", "chebyshev")
    log = []

    x0, y0 = [], []
    for p in warm_start_points:
        s = rescale_to_empirical(p["raw"], nadir_utopia)
        l2_val, cheb_val = fairness_l2(s), fairness_chebyshev(s)
        x0.append(weights_to_z(p["w"]))
        y0.append(l2_val if objective_key == "l2" else cheb_val)
        log.append({
            "phase": "warm_start", "label": p["label"],
            "w_platform": p["w"][0], "w_provider": p["w"][1], "w_civic": p["w"][2],
            "ndcg": p["raw"]["ndcg"], "l2": l2_val, "chebyshev": cheb_val,
        })

    def objective(z):
        w_platform, w_provider, w_civic = z_to_weights(z)
        result = evaluate_combined(w_platform, w_provider, w_civic, env, nadir_utopia)
        log.append({
            "phase": "bo", "label": None,
            "w_platform": w_platform, "w_provider": w_provider, "w_civic": w_civic,
            "ndcg": result["ndcg"], "l2": result["l2"], "chebyshev": result["chebyshev"],
        })
        return result[objective_key]

    result = gp_minimize(
        objective,
        DIMENSIONS,
        x0=x0,
        y0=y0,
        n_calls=n_new_calls,
        n_initial_points=0,
        random_state=random_state,
    )
    w_star = z_to_weights(result.x)
    metrics_star = evaluate_combined(*w_star, env, nadir_utopia)
    return {
        "objective": objective_key,
        "w_platform": w_star[0],
        "w_provider": w_star[1],
        "w_civic": w_star[2],
        "metrics": metrics_star,
        "log": log,
    }


def build_comparison_table(combined_results, warm_start_points, nadir_utopia):
    rows = {}
    for p in warm_start_points:
        s = rescale_to_empirical(p["raw"], nadir_utopia)
        rows[f"{p['label']} (per-objective)"] = {
            "w_platform": p["w"][0], "w_provider": p["w"][1], "w_civic": p["w"][2],
            "ndcg": p["raw"]["ndcg"], "platform": p["raw"]["platform"],
            "provider": p["raw"]["provider"], "civic": p["raw"]["civic"],
            "l2": fairness_l2(s), "chebyshev": fairness_chebyshev(s),
        }
    for r in combined_results:
        raw = r["metrics"]["raw"]
        rows[f"w*_{r['objective']} (combined BO)"] = {
            "w_platform": r["w_platform"], "w_provider": r["w_provider"], "w_civic": r["w_civic"],
            "ndcg": raw["ndcg"], "platform": raw["platform"],
            "provider": raw["provider"], "civic": raw["civic"],
            "l2": r["metrics"]["l2"], "chebyshev": r["metrics"]["chebyshev"],
        }
    return pd.DataFrame(rows).T[
        ["w_platform", "w_provider", "w_civic", "ndcg", "platform", "provider", "civic", "l2", "chebyshev"]
    ]


def main_combined():
    """Runs the two combined BO searches against the already-saved payoff
    table / nadir-utopia / per-objective results on disk (does not repeat
    the expensive 3-run per-objective sweep in main())."""
    nadir_path = os.path.join(OUT_DIR, f"nadir_utopia_{MODEL}_N{N_RESAMPLE}.json")
    with open(nadir_path) as f:
        nadir_utopia = json.load(f)

    warm_start_points = load_warm_start_points()

    print(f"Building env for dataset={DATASET} model={MODEL} N_RESAMPLE={N_RESAMPLE} ...")
    env = build_env()

    combined_results = []
    for objective_key in ("l2", "chebyshev"):
        print(f"\nRunning combined BO for objective={objective_key} "
              f"({len(warm_start_points)} warm-start points + {COMBINED_N_NEW_CALLS} new calls) ...")
        t0 = time.time()
        r = run_combined_bo(objective_key, env, nadir_utopia, warm_start_points)
        print(f"  done in {time.time() - t0:.1f}s -- "
              f"w*=({r['w_platform']:.3f}, {r['w_provider']:.3f}, {r['w_civic']:.3f})")
        print(f"  metrics at w*: ndcg={r['metrics']['ndcg']:.4f} "
              f"l2={r['metrics']['l2']:.4f} chebyshev={r['metrics']['chebyshev']:.4f} "
              f"raw={r['metrics']['raw']}")
        combined_results.append(r)

    w_l2 = np.array([combined_results[0]["w_platform"], combined_results[0]["w_provider"], combined_results[0]["w_civic"]])
    w_cheb = np.array([combined_results[1]["w_platform"], combined_results[1]["w_provider"], combined_results[1]["w_civic"]])
    divergence = float(np.linalg.norm(w_l2 - w_cheb))
    print(f"\n|| w*_l2 - w*_chebyshev || = {divergence:.4f} "
          f"(w*_l2={w_l2.round(3).tolist()}, w*_chebyshev={w_cheb.round(3).tolist()})")

    comparison_df = build_comparison_table(combined_results, warm_start_points, nadir_utopia)
    comparison_path = os.path.join(OUT_DIR, f"combined_bo_comparison_{MODEL}_N{N_RESAMPLE}.csv")
    comparison_df.to_csv(comparison_path)
    print(f"\nSaved comparison table to {comparison_path}")
    print(comparison_df)

    log_path = os.path.join(OUT_DIR, f"combined_bo_log_{MODEL}_N{N_RESAMPLE}.json")
    with open(log_path, "w") as f:
        json.dump(
            {
                "config": {
                    "dataset": DATASET, "model": MODEL, "n_resample": N_RESAMPLE,
                    "n_new_calls": COMBINED_N_NEW_CALLS, "random_state": RANDOM_STATE,
                    "warm_start_labels": [p["label"] for p in warm_start_points],
                },
                "nadir_utopia": nadir_utopia,
                "divergence_l2_vs_chebyshev_weight_vectors": divergence,
                "runs": [
                    {
                        "objective": r["objective"],
                        "w_platform": r["w_platform"], "w_provider": r["w_provider"], "w_civic": r["w_civic"],
                        "metrics": r["metrics"],
                        "log": r["log"],
                    }
                    for r in combined_results
                ],
            },
            f,
            indent=2,
        )
    print(f"Saved full evaluation log (warm-start + new, both runs) to {log_path}")


# ---------------------------------------------------------------------------
# Real production Borda pipeline (social_choice_aggregation.py) at the
# BO-found weights, compared against this experiment's own evaluate()
# recombination ("original metrics") and the existing static/equal-weight
# cached Borda run ("static implementation").
# ---------------------------------------------------------------------------

def run_real_weighted_borda(label, w_platform, w_provider, w_civic,
                             dataset=DATASET, model=MODEL, n_resample=N_RESAMPLE):
    """Runs the actual votekit-Borda election (social_choice_aggregation.py's
    own run_social_choice_for_user, not this experiment's simplified score
    recombination) with the given per-agent weights, one user at a time, and
    saves to <model_dir>/borda_bo_<label>_N<n_resample>/top_k_recommendations.json.

    Uses top_k_resample=n_resample (this experiment's N_RESAMPLE, not
    globals.top_k_resample=150) so the candidate-pool depth matches
    evaluate()'s own recombination for an apples-to-apples comparison.
    """
    sc_recs = get_paths_for_sc_input(dataset)
    paths = sc_recs[model]

    method_data = {}
    for method_name in methods_to_aggregate:
        with open(paths[method_name]) as f:
            method_data[method_name] = json.load(f)

    all_user_ids = set()
    for method_name in method_data:
        all_user_ids.update(method_data[method_name].keys())

    method_weights = {"baseline": 1.0, "platform": w_platform, "provider": w_provider, "civic": w_civic}

    results = {}
    for user_id in tqdm(sorted(all_user_ids), desc=f"real borda ({label})", unit="user"):
        user_recs = {}
        user_candidates = set()
        for method_name in methods_to_aggregate:
            rec_list = get_user_recommendations(method_data[method_name], user_id, top_k_resample=n_resample)
            if rec_list:
                user_recs[method_name] = rec_list
                user_candidates.update(rec_list)
        if not user_recs:
            continue

        result = run_social_choice_for_user(
            user_id, user_recs, list(user_candidates),
            n_seats=top_k_eval, method="borda", method_weights=method_weights,
        )
        if result:
            results[user_id] = flatten_winners(result.get_elected())

    sc_subfolder = f"borda_bo_{label}_N{n_resample}"
    output_dir = get_sc_output_dir(dataset=dataset, model_dir=paths["model_dir"], sc_method=sc_subfolder)
    output_file = os.path.join(output_dir, "top_k_recommendations.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    return output_file, results


def load_static_borda_metrics(env, dataset=DATASET, model=MODEL):
    """Scores the existing cached equal-weight ('static') Borda run already
    on disk -- not regenerated, just read and scored with the same score_df
    used everywhere else in this module."""
    model_dir = find_model_dir(dataset, model)
    with open(os.path.join(model_dir, "borda", "top_k_recommendations.json")) as f:
        static_borda = json.load(f)
    return score_df(results_dict_to_df(static_borda), env)


def main_real_borda_comparison():
    """Runs the real votekit-Borda pipeline at w*_l2 and w*_chebyshev (from
    the already-saved combined BO comparison table), and compares against:
    (a) this experiment's own evaluate() ("original") metrics at those same
    weights, and (b) the existing static/equal-weight cached Borda run."""
    env = build_env()

    comparison_path = os.path.join(OUT_DIR, f"combined_bo_comparison_{MODEL}_N{N_RESAMPLE}.csv")
    combined_df = pd.read_csv(comparison_path, index_col=0)

    rows = {}

    rows["static borda (real, equal-weight)"] = load_static_borda_metrics(env)

    for label, row_name in (("l2", "w*_l2 (combined BO)"), ("chebyshev", "w*_chebyshev (combined BO)")):
        w_platform = float(combined_df.loc[row_name, "w_platform"])
        w_provider = float(combined_df.loc[row_name, "w_provider"])
        w_civic = float(combined_df.loc[row_name, "w_civic"])

        rows[f"{row_name} -- original (evaluate())"] = {
            "ndcg": float(combined_df.loc[row_name, "ndcg"]),
            "platform": float(combined_df.loc[row_name, "platform"]),
            "provider": float(combined_df.loc[row_name, "provider"]),
            "civic": float(combined_df.loc[row_name, "civic"]),
        }

        print(f"\nRunning real weighted Borda for {label} "
              f"(w_platform={w_platform:.3f}, w_provider={w_provider:.3f}, w_civic={w_civic:.3f}) ...")
        t0 = time.time()
        output_file, results = run_real_weighted_borda(label, w_platform, w_provider, w_civic)
        print(f"  done in {time.time() - t0:.1f}s -- saved to {output_file}")
        rows[f"{row_name} -- real (social_choice_aggregation.py)"] = score_df(results_dict_to_df(results), env)

    comparison_table = pd.DataFrame(rows).T[["ndcg", "platform", "provider", "civic"]]
    out_path = os.path.join(OUT_DIR, f"real_borda_comparison_{MODEL}_N{N_RESAMPLE}.csv")
    comparison_table.to_csv(out_path)
    print(f"\nSaved real-vs-original-vs-static comparison to {out_path}")
    print(comparison_table)


def main_raw_comparison():
    """Rescores the already-generated rankings (baseline, static equal-weight
    borda, real borda_bo_l2_N75, real borda_bo_chebyshev_N75 -- none
    regenerated here) with score_df_raw's untransformed convention: raw JSD
    (lower=better) and raw mean GeoILD in km (lower=better, no ideal-ratio
    normalization), instead of score_df's bounded/inverted achievement scores."""
    env = build_env()
    model_dir = find_model_dir(DATASET, MODEL)

    def load_and_score(subfolder):
        with open(os.path.join(model_dir, subfolder, "top_k_recommendations.json")) as f:
            results = json.load(f)
        return score_df_raw(results_dict_to_df(results), env)

    rows = {
        "baseline": load_and_score("baseline"),
        "static borda (equal-weight)": load_and_score("borda"),
        f"w*_l2 (real, N{N_RESAMPLE})": load_and_score(f"borda_bo_l2_N{N_RESAMPLE}"),
        f"w*_chebyshev (real, N{N_RESAMPLE})": load_and_score(f"borda_bo_chebyshev_N{N_RESAMPLE}"),
    }

    table = pd.DataFrame(rows).T[[
        "ndcg", "jsd (lower better)", "ild (higher better)", "geo_ild_km (lower better)",
        "poplift (0=neutral)", "gini (lower better)",
    ]]
    out_path = os.path.join(OUT_DIR, f"raw_metric_comparison_{MODEL}_N{N_RESAMPLE}.csv")
    table.to_csv(out_path)
    print(f"Saved raw-metric comparison to {out_path}")
    print(table)


def main_theoretical_comparison():
    """Same already-generated rankings as main_raw_comparison, rescored with
    score_df_theoretical's [0,1] theoretical-bound achievement scores instead
    of score_df's empirical/mechanism-specific ones -- plus fairness_l2/
    fairness_chebyshev computed on those theoretical scores, for direct
    comparison against build_welfare_table's empirically-rescaled version."""
    env = build_env()
    model_dir = find_model_dir(DATASET, MODEL)

    def load_and_score(subfolder):
        with open(os.path.join(model_dir, subfolder, "top_k_recommendations.json")) as f:
            results = json.load(f)
        return score_df_theoretical(results_dict_to_df(results), env)

    rows = {
        "baseline": load_and_score("baseline"),
        "static borda (equal-weight)": load_and_score("borda"),
        f"w*_l2 (real, N{N_RESAMPLE})": load_and_score(f"borda_bo_l2_N{N_RESAMPLE}"),
        f"w*_chebyshev (real, N{N_RESAMPLE})": load_and_score(f"borda_bo_chebyshev_N{N_RESAMPLE}"),
    }

    table = pd.DataFrame(rows).T[[
        "ndcg", "s_platform (1-jsd)", "s_provider (ild)", "s_civic (1-geo_ild/max)",
        "fairness_l2", "fairness_chebyshev",
    ]]
    out_path = os.path.join(OUT_DIR, f"theoretical_metric_comparison_{MODEL}_N{N_RESAMPLE}.csv")
    table.to_csv(out_path)
    print(f"Saved theoretical-bound comparison to {out_path}")
    print(f"(geo_ild_theoretical_max = {env['geo_ild_theoretical_max']:.2f} km)")
    print(table)


def main_empirical_comparison():
    """Direct empirical counterpart to main_theoretical_comparison -- same 4
    rows, same columns, but s_platform/s_provider/s_civic (and thus
    fairness_l2/fairness_chebyshev) rescaled against the empirical
    (nadir, utopia) from the payoff table instead of a theoretical bound."""
    env = build_env()
    model_dir = find_model_dir(DATASET, MODEL)

    nadir_path = os.path.join(OUT_DIR, f"nadir_utopia_{MODEL}_N{N_RESAMPLE}.json")
    with open(nadir_path) as f:
        nadir_utopia = json.load(f)

    def load_and_score(subfolder):
        with open(os.path.join(model_dir, subfolder, "top_k_recommendations.json")) as f:
            results = json.load(f)
        return score_df_empirical(results_dict_to_df(results), env, nadir_utopia)

    rows = {
        "baseline": load_and_score("baseline"),
        "static borda (equal-weight)": load_and_score("borda"),
        f"w*_l2 (real, N{N_RESAMPLE})": load_and_score(f"borda_bo_l2_N{N_RESAMPLE}"),
        f"w*_chebyshev (real, N{N_RESAMPLE})": load_and_score(f"borda_bo_chebyshev_N{N_RESAMPLE}"),
    }

    table = pd.DataFrame(rows).T[[
        "ndcg", "s_platform", "s_provider", "s_civic", "fairness_l2", "fairness_chebyshev",
    ]]
    out_path = os.path.join(OUT_DIR, f"empirical_metric_comparison_{MODEL}_N{N_RESAMPLE}.csv")
    table.to_csv(out_path)
    print(f"Saved empirical-bound comparison to {out_path}")
    print(table)


def main_percentile_comparison():
    """Direct percentile-rank counterpart to main_empirical_comparison -- same
    4 rows, same columns, but s_platform/s_provider/s_civic (and thus
    fairness_l2/fairness_chebyshev) rescaled by percentile rank within the
    ~124 unique weight vectors explored during the combined BO search,
    instead of linear min-max against just the 2 extreme (nadir, utopia)
    points."""
    env = build_env()
    model_dir = find_model_dir(DATASET, MODEL)

    weights = load_combined_bo_log_weights()
    print(f"Replaying {len(weights)} unique weight vectors from the combined BO log "
          f"to build the percentile-rank reference distribution...")
    reference = build_percentile_reference(env, weights)

    def load_and_score(subfolder):
        with open(os.path.join(model_dir, subfolder, "top_k_recommendations.json")) as f:
            results = json.load(f)
        return score_df_percentile(results_dict_to_df(results), env, reference)

    rows = {
        "baseline": load_and_score("baseline"),
        "static borda (equal-weight)": load_and_score("borda"),
        f"w*_l2 (real, N{N_RESAMPLE})": load_and_score(f"borda_bo_l2_N{N_RESAMPLE}"),
        f"w*_chebyshev (real, N{N_RESAMPLE})": load_and_score(f"borda_bo_chebyshev_N{N_RESAMPLE}"),
    }

    table = pd.DataFrame(rows).T[[
        "ndcg", "s_platform", "s_provider", "s_civic", "fairness_l2", "fairness_chebyshev",
    ]]
    out_path = os.path.join(OUT_DIR, f"percentile_metric_comparison_{MODEL}_N{N_RESAMPLE}.csv")
    table.to_csv(out_path)
    print(f"Saved percentile-rank comparison to {out_path}")
    print(table)


if __name__ == "__main__":
    main_percentile_comparison()
