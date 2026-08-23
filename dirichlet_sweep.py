"""
Dirichlet sweep over social-choice agent weights for Borda and Schulze.

Unlike payoff_table_experiment.py's Bayesian optimization (which searches for
a single per-objective optimum and throws away most of the space to get
there fast), this maps the whole tradeoff surface: many (w_platform,
w_provider, w_civic) vectors are drawn from a Dirichlet distribution (which
guarantees non-negative weights summing to a fixed total, i.e. always a valid
point -- no rejection sampling needed) and each one is scored, so the full
cloud of achievable (platform, provider, civic) outcomes can be plotted
(e.g. a ternary scatter) alongside the BO optima and the equal-weight
baseline as reference points.

baseline stays fixed at 1.0, same convention as payoff_table_experiment.py's
BO -- keeps this sweep's weight space directly comparable to the existing
payoff/BO artifacts (nadir_utopia_*.json etc.) without needing to rebuild
them. Only (w_platform, w_provider, w_civic) are drawn, each with mean 1.0
(Dirichlet(1,1,1) scaled by 3), matching the equal-weight center (1,1,1)
already used as the static baseline.

Both Borda and Schulze are run through the real
social_choice_aggregation.run_social_choice_for_user election pipeline for
every draw -- there is no cheap linear-recombination shortcut for Schulze
(it's a Condorcet/beatpath method, not a weighted sum of per-agent scores),
and Borda is kept on the identical code path so the two methods are directly
comparable rather than one being an approximation.

Scope is controlled by globals.DIRICHLET_SWEEP_MODELS -- a list of
(dataset, model) pairs to sweep. No per-user recommendations are written --
only the weights + resulting metrics are logged, to
datasets/<dataset>_dataset/recommendations/<model_dir>/dirichlet_sweep/
dirichlet_sweep_<dataset>_<model>_N<n_resample>.csv, alongside that model's
other outputs (mirrors get_sc_output_dir's per-model-dir convention in
social_choice_aggregation.py) rather than a separate top-level sweeps/ folder.

Usage:
    python3 dirichlet_sweep.py
"""
import json
import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from globals import (
    DIRICHLET_N_RANDOM_DRAWS,
    DIRICHLET_SWEEP_MODELS,
    methods_to_aggregate,
    top_k_eval,
)
from payoff_table_experiment import N_RESAMPLE, build_env, results_dict_to_df, score_df, score_df_raw
from social_choice_aggregation import (
    BASE_DIR,
    flatten_winners,
    get_paths_for_sc_input,
    get_user_recommendations,
    recommendation_dirpart,
    run_social_choice_for_user,
)

SC_METHODS = ("borda", "schulze")
RANDOM_STATE = 42


def dirichlet_weight_vectors(n_draws, seed=RANDOM_STATE):
    """[(label, (w_platform, w_provider, w_civic)), ...] -- n_draws random
    points from 3*Dirichlet(1,1,1) (mean 1.0 per weight), plus 4 fixed
    anchors (each agent alone dominant, and the equal-weight centroid) so the
    sweep's extremes are guaranteed rather than left to chance at small
    sample sizes."""
    rng = np.random.default_rng(seed)
    random_draws = 3.0 * rng.dirichlet([1.0, 1.0, 1.0], size=n_draws)

    anchors = np.array([
        [3.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 3.0],
        [1.0, 1.0, 1.0],
    ])
    labels = ["corner_platform", "corner_provider", "corner_civic", "centroid"] + [
        f"random_{i}" for i in range(n_draws)
    ]
    weights = np.vstack([anchors, random_draws])
    return list(zip(labels, weights.tolist()))


def run_real_weighted_sc(method, w_platform, w_provider, w_civic, dataset, model, n_resample):
    """Runs the real votekit election (run_social_choice_for_user) with the
    given per-agent weights, one user at a time. Returns {user_id: [item_ids]}
    in memory -- nothing written to disk (unlike
    payoff_table_experiment.run_real_weighted_borda, which persists its
    output; a 40+-draw sweep would otherwise litter the model directory with
    one folder per draw)."""
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

    # Corner/edge draws can set an agent's weight to exact 0.0 -- votekit has
    # never actually been fed a literal zero-weight ballot before (the BO
    # softmax reparam in payoff_table_experiment.py never hits exactly 0, and
    # the SCRUF-D dynamic mechanisms' method_weights.get(name, 1.0) fallback
    # means an "excluded" agent silently gets weight 1.0, not 0) and its
    # internal candidate bookkeeping doesn't tolerate one (a candidate only
    # present on a zero-weight ballot goes missing from profile.candidates,
    # raising a KeyError deep in its Borda scoring). Weight 0 = this agent
    # doesn't vote, so just don't build a ballot for it at all.
    zero_weight_methods = {m for m in ("platform", "provider", "civic") if method_weights[m] == 0.0}

    results = {}
    for user_id in all_user_ids:
        user_recs = {}
        user_candidates = set()
        for method_name in methods_to_aggregate:
            if method_name in zero_weight_methods:
                continue
            rec_list = get_user_recommendations(method_data[method_name], user_id, top_k_resample=n_resample)
            if rec_list:
                user_recs[method_name] = rec_list
                user_candidates.update(rec_list)
        if not user_recs:
            continue

        result = run_social_choice_for_user(
            user_id, user_recs, list(user_candidates),
            n_seats=top_k_eval, method=method, method_weights=method_weights,
        )
        if result:
            results[user_id] = flatten_winners(result.get_elected())

    return results


def _append_row_to_csv(row, out_path):
    """Appends one row to out_path, writing the header only if the file
    doesn't exist yet -- a checkpoint written after every draw, so a crash or
    interrupted run partway through (a real risk here: each draw is a full
    real-election pass, tens of minutes total) still leaves every
    already-scored draw on disk instead of losing the whole run."""
    pd.DataFrame([row]).to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)


def sweep_out_path(dataset, model, n_resample=N_RESAMPLE):
    """.../datasets/<dataset>_dataset/recommendations/<model_dir>/dirichlet_sweep/... --
    lives inside that model's own recommendation directory, alongside its
    borda/schulze social-choice outputs (get_sc_output_dir's convention in
    social_choice_aggregation.py), rather than a separate top-level sweeps/
    folder unrelated to any specific dataset/model."""
    model_dir = get_paths_for_sc_input(dataset)[model]["model_dir"]
    out_dir = os.path.join(BASE_DIR, f"{dataset}_dataset", recommendation_dirpart, model_dir, "dirichlet_sweep")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"dirichlet_sweep_{dataset}_{model}_N{n_resample}.csv")


def run_sweep_for_model(dataset, model, out_path, n_resample=N_RESAMPLE, n_random_draws=DIRICHLET_N_RANDOM_DRAWS):
    print(f"Building env for dataset={dataset} model={model} N_RESAMPLE={n_resample} ...")
    env = build_env(dataset=dataset, model=model, n_resample=n_resample)

    # Fresh run -- start out_path from scratch rather than appending onto a
    # previous run's (possibly differently-scoped) checkpoint file.
    if os.path.exists(out_path):
        os.remove(out_path)

    weight_vectors = dirichlet_weight_vectors(n_random_draws)
    rows = []
    for sc_method in SC_METHODS:
        for label, (w_platform, w_provider, w_civic) in tqdm(
            weight_vectors, desc=f"{dataset}/{model}/{sc_method}", unit="draw"
        ):
            t0 = time.time()
            results = run_real_weighted_sc(sc_method, w_platform, w_provider, w_civic, dataset, model, n_resample)
            result_df = results_dict_to_df(results, k=top_k_eval)
            metrics = score_df(result_df, env)
            # Achievement scores (metrics) are each their own transform of the
            # underlying per-user metric -- provider is already untransformed
            # (mean ILD), platform is 1-mean(JSD) (exactly invertible), but
            # civic (min(1, ideal/geo_ild) per user, then averaged) is a
            # lossy nonlinear transform with no way back to raw mean GeoILD
            # km. Log score_df_raw's untransformed numbers alongside the
            # achievement ones rather than relying on that partial inversion,
            # and pick up poplift/gini for free (score_df doesn't compute
            # either).
            raw_metrics = score_df_raw(result_df, env)
            elapsed = time.time() - t0
            row = {
                "dataset": dataset,
                "model": model,
                "sc_method": sc_method,
                "label": label,
                "w_baseline": 1.0,
                "w_platform": w_platform,
                "w_provider": w_provider,
                "w_civic": w_civic,
                **metrics,
                "jsd_raw": raw_metrics["jsd (lower better)"],
                "ild_raw": raw_metrics["ild (higher better)"],
                "geo_ild_km_raw": raw_metrics["geo_ild_km (lower better)"],
                "poplift_raw": raw_metrics["poplift (0=neutral)"],
                "gini_raw": raw_metrics["gini (lower better)"],
                "elapsed_s": elapsed,
            }
            _append_row_to_csv(row, out_path)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Multi-objective greedy reranker leg -- scaffolded but not wired into main()
# yet. MultiObjectiveGreedyReranker has no agent-recombination shortcut
# (each greedy step's candidate ranking depends on the weights, so there's no
# cached per-agent score to linearly recombine after the fact the way Borda's
# n-minus-rank scores can be) -- every draw is a full rerank_all() pass over
# all users, once per normalization variant ("bounded"/"mo_greedy_pctrank").
# Left as a separate opt-in entry point (run_mo_reranker_sweep) until the
# sample budget/normalization choice for it is settled.
# ---------------------------------------------------------------------------

def dirichlet_weight_vectors_mo(n_draws, seed=RANDOM_STATE):
    """[(label, {"relevance":.., "diversity":.., "geo":.., "calibration":..}), ...]
    -- same construction as dirichlet_weight_vectors, over the 4 MO-reranker
    terms instead of the 3 non-baseline social-choice agents (all 4 terms are
    free here; MO_GREEDY_WEIGHTS has no fixed-at-1.0 "baseline" term)."""
    rng = np.random.default_rng(seed)
    terms = ("relevance", "diversity", "geo", "calibration")
    random_draws = 4.0 * rng.dirichlet([1.0, 1.0, 1.0, 1.0], size=n_draws)

    anchors = np.eye(4) * 4.0
    anchors = np.vstack([anchors, np.ones(4)])  # 4 one-hot corners + equal-weight centroid
    labels = [f"corner_{t}" for t in terms] + ["centroid"] + [f"random_{i}" for i in range(n_draws)]
    weights = np.vstack([anchors, random_draws])
    return [(label, dict(zip(terms, w.tolist()))) for label, w in zip(labels, weights)]


def run_mo_reranker_sweep(dataset, model, n_random_draws=DIRICHLET_N_RANDOM_DRAWS,
                           normalizations=("bounded", "percentile_rank")):
    """Not called from main() -- see module docstring. Each (draw x
    normalization) pair is a full MultiObjectiveGreedyReranker.rerank_all()
    pass over every user; budget n_random_draws accordingly before wiring
    this in."""
    from civic_reranker import load_coordinates
    from evaluation_metrics import max_pairwise_haversine
    from multi_objective_reranker import MultiObjectiveGreedyReranker
    from platform_reranker import (
        calculate_user_popularity_distributions,
        create_base_recommendations,
        dataset_metadata,
        open_ground_truth_user_group,
        recommender_dir_combiner,
        rerank_upd,
    )
    from provider_reranker import build_item_similarity, open_training_data
    from globals import top_k_resample, valid_popularity

    data = dataset_metadata(dataset)
    result = next(r for r in data if r["model"] == model)
    train_data, test_data, user_groups = open_ground_truth_user_group(dataset)

    coords_df = load_coordinates(dataset)
    item_coords = dict(zip(coords_df["item_id:token"], zip(coords_df["lat:float"], coords_df["lon:float"])))
    max_geo_km = max_pairwise_haversine(item_coords)

    interaction_df = open_training_data(dataset)
    item_sim, item_id_to_idx = build_item_similarity(interaction_df)

    baseline_topk_dir, basedir = recommender_dir_combiner(dataset, result["directory"])
    base_resample, _ = create_base_recommendations(
        baseline_topk_dir, top_k_resample=top_k_resample, top_k_eval=top_k_eval
    )
    _, item_popularity, _ = rerank_upd(
        dataset, base_resample, top_k_eval, valid_popularity, dir_to_save=None, train_data=train_data,
    )
    item_pop_group = dict(zip(item_popularity["item_id:token"], item_popularity["item_pop_group"]))
    base_resample = base_resample.merge(item_popularity, on="item_id:token")

    train_data_with_tiers = train_data.merge(item_popularity, on="item_id:token", how="left")
    user_profiles = calculate_user_popularity_distributions(train_data_with_tiers, item_popularity)

    env = build_env(dataset=dataset, model=model)  # for score_df's ndcg/jsd/ild/geo scoring

    weight_vectors = dirichlet_weight_vectors_mo(n_random_draws)
    rows = []
    for normalization in normalizations:
        for label, weights in tqdm(weight_vectors, desc=f"{dataset}/{model}/mo_greedy/{normalization}", unit="draw"):
            t0 = time.time()
            reranker = MultiObjectiveGreedyReranker(
                item_sim, item_id_to_idx, item_coords, item_pop_group, max_geo_km,
                weights=weights, normalization=normalization,
            )
            out_df = reranker.rerank_all(base_resample, user_profiles, top_k=top_k_eval)
            df = out_df.rename(columns={"item_id:token": "item_id:token", "user_id:token": "user_id:token"})
            metrics = score_df(df[["user_id:token", "item_id:token"]], env)
            elapsed = time.time() - t0
            rows.append({
                "dataset": dataset, "model": model, "normalization": normalization, "label": label,
                **weights, **metrics, "elapsed_s": elapsed,
            })
    return pd.DataFrame(rows)


def main():
    for dataset, model in DIRICHLET_SWEEP_MODELS:
        out_path = sweep_out_path(dataset, model)
        df = run_sweep_for_model(dataset, model, out_path)
        print(f"\nSaved {len(df)} rows to {out_path} (checkpointed incrementally, one row per draw)")
        print(df.groupby(["sc_method"])[["ndcg", "platform", "provider", "civic"]].describe())


if __name__ == "__main__":
    main()
