import os
import json
import random
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile, RankProfile
from votekit.elections import Schulze, Borda
from evaluation_metrics import rank_biased_overlap
from compute_user_compatibility import load_user_compatibility
from dynamic_allocation import (
    FairnessTracker,
    compute_weights,
    select_least_fair,
    select_lottery,
    stream_order,
)
from globals import (
    top_k_resample,
    top_k_eval,
    available_datasets,
    recommendation_dirpart,
    sc_models,
    methods_to_aggregate,
    boosting,
    run_static_sc,
    BASE_DIR,
    run_least_fair,
    run_weighted_mi,
    run_weighted_ci,
    run_weighted_mi_ci,
    run_lottery_mi,
    run_lottery_ci,
    run_lottery_mi_ci,
    fairness_agents,
    dynamic_window,
    dynamic_weight_floor,
    dynamic_seed,
)


def _defragmented_group_ballots(self):
    """
    Drop-in replacement for votekit's RankProfile.group_ballots(). The original
    groups ballots via groupby().aggregate(...).reset_index(), and reset_index()
    inserts one column per Ranking_i level one at a time -- with enough ranking
    columns (i.e. large top_k_resample/candidate pools) that triggers pandas'
    "DataFrame is highly fragmented" PerformanceWarning on every user, since this
    runs once per user in run_social_choice_for_user. Building the ranking columns
    and the aggregated columns as two frames and joining them with a single
    pd.concat(axis=1) avoids the per-column inserts entirely (verified to produce
    an identical df, just without the fragmentation).
    """
    if len(self.df) == 0:
        return RankProfile(candidates=self.candidates, max_ranking_length=self.max_ranking_length)

    ranking_cols = [c for c in self.df.columns if "Ranking_" in c]
    group_df = self.df.groupby(ranking_cols, dropna=False)
    aggregated = group_df.aggregate(
        {
            "Weight": "sum",
            "Voter Set": (lambda sets: set().union(*sets)),
        }
    )
    index_df = aggregated.index.to_frame(index=False)
    index_df.index = aggregated.index
    new_df = pd.concat([index_df, aggregated], axis=1).reset_index(drop=True)
    new_df.index.name = "Ballot Index"

    return RankProfile(df=new_df, candidates=self.candidates, max_ranking_length=self.max_ranking_length)


RankProfile.group_ballots = _defragmented_group_ballots


def candidates_to_ballot(candidates_list):
    """
    Converts a list of candidates into a ballot ranking,
    where each candidate is in its own rank (tier).
    """
    return [frozenset([candidate]) for candidate in candidates_list]

def get_user_recommendations(data, user_id, top_k_resample=top_k_resample, verify=False):
    """
    Extract the list of item_ids for a given user from the data.
    Handles two structures:
    - Nested list: [[item1, item2, ...]]
    - Direct list: [item1, item2, ...]
    """
    if user_id not in data:
        return []
    
    user_data = data[user_id]
    items = []
    
    # Check if it's a nested list structure
    if isinstance(user_data, list):
        if len(user_data) > 0 and isinstance(user_data[0], list):
            # Nested list: [[items]] -> flatten to [items]
            items = user_data[0]
        else:
            # Direct list: [items]
            items = user_data
    
    original_length = len(items)
    sliced_items = items[:top_k_resample]
    
    if verify:
        print(f"    SLICE VERIFICATION: Original={original_length}, Sliced={len(sliced_items)}, top_k_resample={top_k_resample}")
        print(f"    Are they different? {original_length != len(sliced_items)}")
        if original_length > top_k_resample:
            print(f"    Successfully sliced from {original_length} to {len(sliced_items)}")
        else:
            print(f"    Original length ({original_length}) <= top_k_resample ({top_k_resample}), no slicing occurred")
    
    return sliced_items

def run_social_choice_for_user(user_id, method_recommendations, all_candidates, n_seats=top_k_eval, method="schulze", verify=False, method_weights=None):
    """
    Run a specific social choice aggregation for a single user across multiple methods (baseline, platform, civic, provider).
    """
    ballots = []
    for method_name, rec_list in method_recommendations.items():
        if rec_list:
            if verify:
                print(f"    {method_name}: Creating ballot with {len(rec_list)} items")
                
            weight = method_weights.get(method_name, 1.0) if method_weights else 1.0
            ballot = Ballot(ranking=candidates_to_ballot(rec_list), weight=weight)
            ballots.append(ballot)
            
            if verify:
                ranking_lengths = [len(rank) for rank in ballot.ranking]
                print(f"    {method_name}: Creating ballot with {len(rec_list)} items (weight={weight})")
                print(f"      Ballot ranking has {len(ballot.ranking)} tiers with {sum(ranking_lengths)} total items")

    if not ballots:
        return None

    if verify:
        print(f"    Total ballots created: {len(ballots)}")
        print(f"    Candidate universe size: {len(all_candidates)}")

    profile = PreferenceProfile(ballots=ballots, candidates=all_candidates)
    profile = profile.group_ballots()

    if method == "schulze":
        return Schulze(profile=profile, tie_break="first_place", n_seats=n_seats)
    elif method == "borda":
        return Borda(profile=profile, tiebreak="first_place", n_seats=n_seats)
    else:
        raise ValueError(f"Unknown method: {method}")

def flatten_winners(elected_list):
    """Convert list of frozensets to flat list of item IDs"""
    winners = []
    for item in elected_list:
        if isinstance(item, frozenset):
            winners.extend(list(item))
        else:
            winners.append(str(item))
    return winners


def run_and_save_boosted_method(sc_method, model_name, model_dir, method_data, all_user_ids, dataset,
                                 boost_method=None, boost_factor=2.0, n_seats=top_k_eval):
    """
    Run social choice with one method boosted (weight=boost_factor, others=1.0).
    If boost_method is None, runs the baseline (all equal weights).
    Saves to e.g. 'schulze2baseline/' or 'schulze2platform/'.
    """
    method_names = list(methods_to_aggregate)

    if boost_method is None:
        method_weights = None
        folder_suffix = ""
    else:
        method_weights = {m: (boost_factor if m == boost_method else 1.0) for m in method_names}
        folder_suffix = "2"+boost_method

    # e.g. "schulze2baseline", "schulze2platform"
    sc_subfolder = f"{sc_method}{folder_suffix}"
    results = {}
    user_list = list(all_user_ids)


    for user_id in tqdm(user_list, desc=f"Processing {model_name}-{sc_subfolder}", unit="user"):

        user_recs = {}
        user_candidates = set()

        for method_name in methods_to_aggregate:
            rec_list = get_user_recommendations(
                method_data[method_name],
                user_id,
            )
            if rec_list:
                user_recs[method_name] = rec_list
                user_candidates.update(rec_list)

        if not user_recs:
            continue

        result = run_social_choice_for_user(
            user_id,
            user_recs,
            list(user_candidates),
            n_seats=n_seats,
            method=sc_method,
            method_weights=method_weights
        )

        if result:
            results[user_id] = flatten_winners(result.get_elected())

    # Save using the sc_subfolder name instead of sc_method
    output_dir = get_sc_output_dir(
        dataset=dataset,
        model_dir=model_dir,
        sc_method=sc_subfolder          # <-- "schulze2baseline", "schulze2platform", etc.
    )
    output_file = os.path.join(output_dir, "top_k_recommendations.json")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n{sc_subfolder} results for {model_name} saved to: {output_file}")
    print(f"Total users processed: {len(results)}")



def build_user_recs(user_id, method_data):
    """Collect each method's recommendation list and candidate pool for a single user."""
    user_recs = {}
    user_candidates = set()

    for method_name in methods_to_aggregate:
        rec_list = get_user_recommendations(method_data[method_name], user_id)
        if rec_list:
            user_recs[method_name] = rec_list
            user_candidates.update(rec_list)

    return user_recs, user_candidates


_WEIGHTED_SUFFIXES = {"mi": "_weighted_mi", "ci": "_weighted_ci", "mi_ci": "_weighted_mi_ci"}
_LOTTERY_SUFFIXES = {"mi": "_lottery_mi", "ci": "_lottery_ci", "mi_ci": "_lottery_mi_ci"}


def run_and_save_mechanism(sc_method, model_name, model_dir, method_data, all_user_ids, dataset,
                            mechanism, weighting_source=None, n_seats=top_k_eval, user_compatibility=None):
    """
    Run one SCRUF-D dynamic-allocation variant and save it. Covers all three mechanisms:

    - "least_fair": deterministically pick the single lowest-m_i fairness agent each round
      (ignores compatibility entirely -- that's the mechanism's definition).
    - "weighted":   redistribute continuous weight across all fairness agents, beta_i ~
      raw_weight_score(..., weighting_source); baseline stays fixed at 1.0.
    - "lottery":    draw a single fairness agent each round with probability ~
      raw_weight_score(..., weighting_source); the drawn agent gets the full
      fairness-agent mass (len(fairness_agents)) alongside baseline, so it's
      comparable in total voting power to "weighted" and "least_fair".

    `weighting_source` (required for "weighted"/"lottery", ignored for "least_fair"):
      "mi" (1 - m_i alone), "ci" (c_i alone), or "mi_ci" ((1 - m_i) * c_i, SCRUF-D default).

    m_i is each agent's rank-biased overlap with the delivered output over a sliding
    window of recently processed users (see FairnessTracker); c_i is its precomputed
    per-user compatibility (compute_user_compatibility.py), a proxy for the user
    profile omega. Users are processed in a fixed-seed shuffled order to simulate a
    stream; the window is expanding (no separate burn-in), so only the very first
    user has no history to react to.

    Saves to '<sc_method>_leastfair/', '<sc_method>_weighted_<mi|ci|mi_ci>/', or
    '<sc_method>_lottery_<mi|ci|mi_ci>/', plus a per-user log (weights for "weighted",
    chosen agent + mi/ci for "least_fair"/"lottery").
    """
    if mechanism not in ("least_fair", "weighted", "lottery"):
        raise ValueError(f"Unknown mechanism: {mechanism!r}")
    if mechanism != "least_fair" and weighting_source not in ("mi", "ci", "mi_ci"):
        raise ValueError(f"weighting_source must be 'mi', 'ci', or 'mi_ci' for mechanism={mechanism!r}")

    needs_ci = weighting_source in ("ci", "mi_ci")

    if mechanism == "least_fair":
        sc_subfolder = f"{sc_method}_leastfair"
    elif mechanism == "weighted":
        sc_subfolder = f"{sc_method}{_WEIGHTED_SUFFIXES[weighting_source]}"
    else:
        sc_subfolder = f"{sc_method}{_LOTTERY_SUFFIXES[weighting_source]}"

    tracker = FairnessTracker(fairness_agents, window=dynamic_window)
    rng = random.Random(dynamic_seed) if mechanism == "lottery" else None
    results = {}
    per_user_log = {}

    user_list = stream_order(all_user_ids, seed=dynamic_seed)

    for user_id in tqdm(user_list, desc=f"Processing {model_name}-{sc_subfolder}", unit="user"):

        user_recs, user_candidates = build_user_recs(user_id, method_data)
        if not user_recs:
            continue

        mi_scores = tracker.mi()

        ci_scores = None
        if needs_ci:
            user_comp = user_compatibility.get(user_id, {}) if user_compatibility else {}
            ci_scores = {
                agent: user_comp.get(agent, 1.0)
                for agent in fairness_agents
                if agent in user_recs
            }

        if mechanism == "weighted":
            active_recs = user_recs
            method_weights = compute_weights(
                mi_scores,
                fairness_agents,
                ci_scores=ci_scores,
                source=weighting_source,
                baseline_weight=1.0,
                floor=dynamic_weight_floor,
            )
        else:
            chosen_agent = (
                select_least_fair(mi_scores, fairness_agents)
                if mechanism == "least_fair"
                else select_lottery(mi_scores, fairness_agents, rng, ci_scores=ci_scores, source=weighting_source)
            )
            active_recs = {name: recs for name, recs in user_recs.items() if name in ("baseline", chosen_agent)}
            if not active_recs:
                continue
            method_weights = {chosen_agent: float(len(fairness_agents)), "baseline": 1.0}

        result = run_social_choice_for_user(
            user_id,
            active_recs,
            list(user_candidates),
            n_seats=n_seats,
            method=sc_method,
            method_weights=method_weights,
        )

        if not result:
            continue

        delivered_list = flatten_winners(result.get_elected())
        results[user_id] = delivered_list
        per_user_log[user_id] = (
            method_weights if mechanism == "weighted" else {"chosen": chosen_agent, "mi": mi_scores, "ci": ci_scores}
        )

        agreement_scores = {
            agent: rank_biased_overlap(user_recs[agent], delivered_list, k=n_seats)
            for agent in fairness_agents
            if agent in user_recs
        }
        tracker.update(agreement_scores)

    output_dir = get_sc_output_dir(
        dataset=dataset,
        model_dir=model_dir,
        sc_method=sc_subfolder
    )
    output_file = os.path.join(output_dir, "top_k_recommendations.json")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    log_name = "dynamic_weights_log.json" if mechanism == "weighted" else "choice_log.json"
    with open(os.path.join(output_dir, log_name), "w") as f:
        json.dump(per_user_log, f, indent=4)

    print(f"\n{sc_subfolder} results for {model_name} saved to: {output_file}")
    print(f"Total users processed: {len(results)}")


def dataset_metadata(dataset, recommendation_dirpart, base_dir):
    """Extract metadata for each dataset and model"""
    data = []

    recs = [
        d
        for d in os.listdir(os.path.join(base_dir, f"{dataset}_dataset", recommendation_dirpart))
        if os.path.isdir(
            os.path.join(base_dir, f"{dataset}_dataset", recommendation_dirpart, d)
        )
    ]

    for dir in recs:
        json_file = os.path.join(base_dir, f"{dataset}_dataset", recommendation_dirpart, dir, "general_evaluation.json")

        if not os.path.exists(json_file):
            # print(f"Skipping {json_file} - File not found.")
            continue

        with open(json_file, "r") as f:
            eval_data = json.load(f)

        test_results = eval_data.get("test_result", {})
        test_results["directory"] = dir

        test_results["dataset"] = dir.split("-")[0]
        parts = dir.split("-")

        if parts[1] == "debias":
            test_results["model_type"] = "debias"
            test_results["model"] = parts[2]
            test_results["date"] = "-".join(parts[3:])
        elif parts[1] == "contextpoi":
            test_results["model_type"] = "contextpoi"
            test_results["model"] = parts[2]
            test_results["date"] = "-".join(parts[3:])
        else:
            test_results["model_type"] = "general"
            test_results["model"] = parts[1]
            test_results["date"] = "-".join(parts[2:])

        if test_results["model"] == "MF":
            test_results["model_type"] = "general (via RecBole debias)"

        data.append(test_results)

    return data

def create_model_directories(dataset, data, base_dir, recommendation_dirpart):
    """
    Create output directories
    """
    model_directories = {}
    methods = methods_to_aggregate  # right now we only use the recs from the RS output without regarding CP

    for result in data:
        model_name = result["model"]
        model_directories[model_name] = {}
        
        for method in methods:
            path = os.path.join(base_dir, f"{dataset}_dataset", recommendation_dirpart, result['directory'], method, "top_k_recommendations.json")
            model_directories[model_name][method] = path

    return model_directories


def get_paths_for_sc_input(dataset, recommendation_dirpart=recommendation_dirpart, base_dir=BASE_DIR):
    recommender_metadata = dataset_metadata(dataset=dataset, recommendation_dirpart=recommendation_dirpart, base_dir=base_dir)
    model_dirs = create_model_directories(dataset=dataset, data=recommender_metadata, recommendation_dirpart=recommendation_dirpart, base_dir=base_dir)
    
    sc_recs = {}
    for element in recommender_metadata:
        print(element)
        if element["model"] in sc_models:
            model_name = element["model"]
            sc_recs[model_name] = {
                "baseline": model_dirs[model_name]["baseline"],
                "platform": model_dirs[model_name]["platform"],
                "civic": model_dirs[model_name]["civic"],
                "provider": model_dirs[model_name]["provider"],
                "model_dir": element['directory']
            }

    return sc_recs

def get_sc_output_dir(dataset, model_dir, sc_method, recommendation_dirpart=recommendation_dirpart, base_dir=BASE_DIR):
    """
    Create output directory path for social choice results within the model directory
    Returns:
        Path like: .../foursquaretky_sample-BPR-Apr-21-2025_21-54-55/borda/
    """
    sc_output_dir = os.path.join(base_dir, f'{dataset}_dataset', recommendation_dirpart, model_dir, sc_method)
    os.makedirs(sc_output_dir, exist_ok=True)
    return sc_output_dir

def timestamp_creator():
    """Generate a base timestamp as str"""
    return datetime.now().strftime("%b-%d-%Y_%H-%M-%S")

# (mechanism, weighting_source, on/off flag) -- drives the dynamic-allocation loop in main().
# weighting_source is None for "least_fair" (it ignores compatibility entirely, per the mechanism).
DYNAMIC_VARIANTS = [
    ("least_fair", None, run_least_fair),
    ("weighted", "mi", run_weighted_mi),
    ("weighted", "ci", run_weighted_ci),
    ("weighted", "mi_ci", run_weighted_mi_ci),
    ("lottery", "mi", run_lottery_mi),
    ("lottery", "ci", run_lottery_ci),
    ("lottery", "mi_ci", run_lottery_mi_ci),
]
_NEEDS_USER_COMPATIBILITY = any(
    flag for _, source, flag in DYNAMIC_VARIANTS if source in ("ci", "mi_ci")
)


def main():
    for dataset in available_datasets:
        print(f"\n{'='*60}")
        print("SOCIAL CHOICE AGGREGATION")
        print(f"{'='*60}\n")

        sc_dict = get_paths_for_sc_input(dataset)

        print(f"Models to process: {list(sc_dict.keys())}")
        print(f"Expected models: {sc_models}")

        user_compatibility = load_user_compatibility(dataset) if _NEEDS_USER_COMPATIBILITY else None

        # Process each model separately
        for model_name, paths in sc_dict.items():
            print(f"\n{'='*60}")
            print(f"PROCESSING MODEL: {model_name}")
            print(f"{'='*60}")
            for method_name, file_path in paths.items():
                if method_name == "model_dir":
                    print(f"  Model directory: {file_path}")
                    continue
                exists = "✓" if os.path.exists(file_path) else "✗ MISSING"
                print(f"  {method_name}: {exists}")
                print(f"    {file_path}")
            

            method_data = {}
            unique_items = set()

            print(f"Loading data for {model_name}...")
            for method_name in methods_to_aggregate:
                file_path = paths[method_name]
                print(file_path)
                
                if not os.path.exists(file_path):
                    print(f"ERROR: File not found for {model_name}/{method_name}: {file_path}")
                    continue
                    
                with open(file_path, "r") as f:
                    method_data[method_name] = json.load(f)
                

                for user_id in method_data[method_name].keys():
                    user_data = method_data[method_name][user_id]
                    if isinstance(user_data, list):
                        if user_data and isinstance(user_data[0], list):
                            # Nested list: [[items]]
                            unique_items.update(user_data[0])
                        else:
                            # Direct list: [items]
                            unique_items.update(user_data)

            all_user_ids = set()
            for method_name in method_data:
                all_user_ids.update(method_data[method_name].keys())

            for sc_method in ["borda", "schulze"]:
                print(f"\n{'='*60}")
                print(f"Running experiments for {sc_method.upper()} | {model_name}")
                print(f"{'='*60}")

                if run_static_sc:
                    if boosting:
                        for boost_method in methods_to_aggregate:
                            print(f"\n{'='*60}")
                            print(f"Running boosting experiments for {sc_method.upper()} | {model_name} boosting {boost_method}")
                            print(f"{'='*60}")
                            run_and_save_boosted_method(
                                sc_method=sc_method,
                                model_name=model_name,
                                model_dir=paths['model_dir'],
                                method_data=method_data,
                                all_user_ids=all_user_ids,
                                dataset=dataset,
                                boost_method=boost_method,
                                boost_factor=2.0,
                            )
                    else:
                        print(f"\n{'='*60}")
                        print(f"Running equal-weight experiments for {sc_method.upper()} | {model_name}")
                        print(f"{'='*60}")
                        run_and_save_boosted_method(
                        sc_method=sc_method,
                        model_name=model_name,
                        model_dir=paths['model_dir'],
                        method_data=method_data,
                        all_user_ids=all_user_ids,
                        dataset=dataset,
                        boost_method=None,          # equal weights
                    )

                for mechanism, weighting_source, enabled in DYNAMIC_VARIANTS:
                    if not enabled:
                        continue
                    label = mechanism if weighting_source is None else f"{mechanism}/{weighting_source}"
                    print(f"\n{'='*60}")
                    print(f"Running dynamic allocation ({label}) experiments for {sc_method.upper()} | {model_name}")
                    print(f"{'='*60}")
                    run_and_save_mechanism(
                        sc_method=sc_method,
                        model_name=model_name,
                        model_dir=paths['model_dir'],
                        method_data=method_data,
                        all_user_ids=all_user_ids,
                        dataset=dataset,
                        mechanism=mechanism,
                        weighting_source=weighting_source,
                        user_compatibility=user_compatibility,
                    )

        print("\n" + "="*60)
        print("All social choice aggregation results saved successfully!")
        print("="*60)


if __name__ == "__main__":
    main()