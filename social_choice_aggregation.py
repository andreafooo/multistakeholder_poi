import os
import json
from tqdm import tqdm
from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
from votekit.elections import Schulze, Borda
from config import BaseConfig
from utils import get_paths_for_sc_input, get_sc_output_dir

config = BaseConfig

def candidates_to_ballot(candidates_list):
    """
    Converts a list of candidates into a ballot ranking,
    where each candidate is in its own rank (tier).
    """
    return [frozenset([candidate]) for candidate in candidates_list]

def get_user_recommendations(data, user_id, top_k_resample=config.top_k_resample, verify=False):
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
            print(f"    ✓ Successfully sliced from {original_length} to {len(sliced_items)}")
        else:
            print(f"    ⚠ Original length ({original_length}) <= top_k_resample ({top_k_resample}), no slicing occurred")
    
    return sliced_items

def run_social_choice_for_user(user_id, method_recommendations, all_candidates, n_seats=config.top_k_eval, method="schulze", verify=False, method_weights=None):
    """
    Run a specific social choice aggregation for a single user across multiple methods (baseline, cp_min_js).
    """
    ballots = []
    
    if verify:
        print(f"\n  BALLOT CREATION VERIFICATION:")
    
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
        return Schulze(profile=profile, n_seats=n_seats)
    elif method == "borda":
        return Borda(profile=profile, tiebreak="borda", n_seats=n_seats)
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
                                 boost_method=None, boost_factor=2.0, n_seats=config.top_k_eval):
    """
    Run social choice with one method boosted (weight=boost_factor, others=1.0).
    If boost_method is None, runs the baseline (all equal weights).
    Saves to e.g. 'schulze2baseline/' or 'schulze2cp_min_js/'.
    """
    method_names = list(config.methods_to_aggregate)

    if boost_method is None:
        method_weights = None
        folder_suffix = ""
    else:
        method_weights = {m: (boost_factor if m == boost_method else 1.0) for m in method_names}
        folder_suffix = boost_method

    # e.g. "schulze2baseline", "schulze2cp_min_js"
    sc_subfolder = f"{sc_method}2{folder_suffix}"

    results = {}

    user_list = list(all_user_ids)
    sample_users = user_list[:3] if len(user_list) >= 3 else user_list

    for user_id in tqdm(user_list, desc=f"Processing {model_name}-{sc_subfolder}", unit="user"):
        is_sample = user_id in sample_users

        user_recs = {}
        user_candidates = set()

        for method_name in config.methods_to_aggregate:
            rec_list = get_user_recommendations(
                method_data[method_name],
                user_id,
                verify=is_sample
            )
            if rec_list:
                user_recs[method_name] = rec_list
                user_candidates.update(rec_list)

        if is_sample:
            print(f"\n{'='*60}")
            print(f"VERIFICATION FOR USER: {user_id} | {model_name} | {sc_subfolder}")
            print(f"Weights: {method_weights}")
            print(f"{'='*60}")

        if not user_recs:
            continue

        result = run_social_choice_for_user(
            user_id,
            user_recs,
            list(user_candidates),
            n_seats=n_seats,
            method=sc_method,
            verify=is_sample,
            method_weights=method_weights
        )

        if result:
            results[user_id] = flatten_winners(result.get_elected())

    # Save using the sc_subfolder name instead of sc_method
    output_dir = get_sc_output_dir(
        dataset=dataset,
        model_dir=model_dir,
        sc_method=sc_subfolder          # <-- "schulze2baseline", "schulze2cp_min_js", etc.
    )
    output_file = os.path.join(output_dir, "top_k_recommendations.json")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n{sc_subfolder} results for {model_name} saved to: {output_file}")
    print(f"Total users processed: {len(results)}")

def main():
    for dataset in config.available_datasets:
        print(f"\n{'='*60}")
        print("SOCIAL CHOICE AGGREGATION")
        print(f"{'='*60}\n")

        sc_dict = get_paths_for_sc_input()

        print(f"Models to process: {list(sc_dict.keys())}")
        print(f"Expected models: {config.sc_models}")
        
        # Process each model separately
        for model_name, paths in sc_dict.items():
            print(f"\n{'='*60}")
            print(f"PROCESSING MODEL: {model_name}")
            print(f"{'='*60}")
            
            print(f"\nFile paths:")
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
            for method_name in config.methods_to_aggregate:
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

            # Run each social choice method for this model
            for sc_method in ["borda", "schulze"]:
                print(f"\n{'='*60}")
                print(f"Running experiments for {sc_method.upper()} | {model_name}")
                print(f"{'='*60}")

                if config.boosting:
                    for boost_method in config.methods_to_aggregate:
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

        print("\n" + "="*60)
        print("All social choice aggregation results saved successfully!")
        print("="*60)


if __name__ == "__main__":
    main()