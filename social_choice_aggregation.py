import os
import json
from tqdm import tqdm
from datetime import datetime
from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
from votekit.elections import Schulze, Borda
from globals import top_k_resample, top_k_eval, available_datasets, recommendation_dirpart, sc_models, methods_to_aggregate, boosting, BASE_DIR


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
    Run a specific social choice aggregation for a single user across multiple methods (baseline, cp_min_js).
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
                                 boost_method=None, boost_factor=2.0, n_seats=top_k_eval):
    """
    Run social choice with one method boosted (weight=boost_factor, others=1.0).
    If boost_method is None, runs the baseline (all equal weights).
    Saves to e.g. 'schulze2baseline/' or 'schulze2cp_min_js/'.
    """
    method_names = list(methods_to_aggregate)

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
        sc_method=sc_subfolder          # <-- "schulze2baseline", "schulze2cp_min_js", etc.
    )
    output_file = os.path.join(output_dir, "top_k_recommendations.json")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

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
                "cp_min_js": model_dirs[model_name]["cp_min_js"],
                "geo": model_dirs[model_name]["geo"],
                "mmr": model_dirs[model_name]["mmr"],
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

def main():
    for dataset in available_datasets:
        print(f"\n{'='*60}")
        print("SOCIAL CHOICE AGGREGATION")
        print(f"{'='*60}\n")

        sc_dict = get_paths_for_sc_input(dataset)

        print(f"Models to process: {list(sc_dict.keys())}")
        print(f"Expected models: {sc_models}")
        
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

        print("\n" + "="*60)
        print("All social choice aggregation results saved successfully!")
        print("="*60)


if __name__ == "__main__":
    main()