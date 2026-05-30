import os
import json
from tqdm import tqdm
from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
from votekit.elections import Schulze, Borda, CondoBorda, PluralityVeto, FastSTV, STV, RankedPairs, SimultaneousVeto
from config import BaseConfig
from utils import get_paths_for_sc_input, get_sc_output_dir, timestamp_creator

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

def run_social_choice_for_user(user_id, method_recommendations, all_candidates, n_seats=config.top_k_eval, method="schulze", verify=False):
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
            ballot = Ballot(ranking=candidates_to_ballot(rec_list))
            ballots.append(ballot)
            
            if verify:
                # Check the ballot's actual ranking length
                ranking_lengths = [len(rank) for rank in ballot.ranking]
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
    elif method == "simulveto":
        return SimultaneousVeto(profile=profile, tiebreak="random", n_seats=n_seats)
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

def run_and_save_method(sc_method, model_name, model_dir, method_data, all_user_ids, n_seats=config.top_k_eval):
    """
    Run a specific social choice method for all users within ONE model and save the results.
    """
    results = {}
    
    # VERIFICATION: Sample THREE users to inspect
    user_list = list(all_user_ids)
    sample_users = user_list[:3] if len(user_list) >= 3 else user_list
    
    for idx, user_id in enumerate(tqdm(user_list, desc=f"Processing {model_name}-{sc_method}", unit="user")):
        is_sample = user_id in sample_users
        
        user_recs = {}
        user_candidates = set()  # USER-SPECIFIC candidates
        
        # Aggregate agents for this user
        for method_name in config.methods_to_aggregate:
            rec_list = get_user_recommendations(
                method_data[method_name], 
                user_id, 
                verify=is_sample
            )
            if rec_list:
                user_recs[method_name] = rec_list
                user_candidates.update(rec_list)  # Add this user"s candidates
        
        # VERIFICATION: Print details for sample users
        if is_sample:
            print(f"\n{'='*60}")
            print(f"VERIFICATION FOR USER {sample_users.index(user_id)+1}/3: {user_id}")
            print(f"MODEL: {model_name}, SC METHOD: {sc_method}")
            print(f"{'='*60}")
            print(f"Number of methods providing recommendations: {len(user_recs)}")
            for method_name, rec_list in user_recs.items():
                print(f"  Method: {method_name}")
                print(f"    Recommendation list length: {len(rec_list)}")
                print(f"    Expected length (top_k_resample): {config.top_k_resample}")
                print(f"    Match? {len(rec_list) == config.top_k_resample}")
                print(f"    First 3 items: {rec_list[:3]}")
                print(f"    Last 3 items: {rec_list[-3:]}")
            print(f"  User-specific candidate universe size: {len(user_candidates)}")

        if not user_recs:
            continue

        result = run_social_choice_for_user(
            user_id, 
            user_recs, 
            list(user_candidates),  # Use USER-SPECIFIC candidates
            n_seats=n_seats, 
            method=sc_method,
            verify=is_sample
        )
        
        if result:
            results[user_id] = flatten_winners(result.get_elected())
            
            # VERIFICATION: Print result for sample users
            if is_sample:
                print(f"\n  RESULT:")
                print(f"    Winners: {results[user_id][:5]}... (showing first 5)")
                print(f"    Number of winners: {len(results[user_id])}")
                print(f"    Expected (n_seats): {n_seats}")
                print(f"    Match? {len(results[user_id]) == n_seats}")
                print(f"{'='*60}\n")

    # Save to model-specific directory
    output_dir = get_sc_output_dir(
        dataset=config.available_datasets[0],
        model_dir=model_dir,
        sc_method=sc_method
    )
    output_file = os.path.join(output_dir, "top_k_recommendations.json")
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\n{sc_method.capitalize()} results for {model_name} saved to: {output_file}")
    print(f"Total users processed: {len(results)}")

def main():
    print(f"\n{'='*60}")
    print("SOCIAL CHOICE AGGREGATION - MODEL-INTERNAL AGGREGATION")
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
        

        # Load data for baseline and cp_min_js
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
            
            # VERIFICATION: Check ORIGINAL data structure
            sample_user_id = list(method_data[method_name].keys())[0]
            sample_data = method_data[method_name][sample_user_id]
            
            print(f"\n  Method: {method_name}")
            print(f"    Users: {len(method_data[method_name])}")
            print(f"    Sample user {sample_user_id}:")
            print(f"    Type of sample_data: {type(sample_data)}")
            
            if isinstance(sample_data, list) and len(sample_data) > 0:
                if isinstance(sample_data[0], list):
                    print(f"    NESTED LIST STRUCTURE - {len(sample_data[0])} items in inner list")
                    print(f"    First 5 items: {sample_data[0][:5]}")
                else:
                    print(f"    DIRECT LIST - {len(sample_data)} items")
                    print(f"    First 5 items: {sample_data[:5]}")
            
            # Collect unique items
            for user_id in method_data[method_name].keys():
                user_data = method_data[method_name][user_id]
                if isinstance(user_data, list):
                    if user_data and isinstance(user_data[0], list):
                        # Nested list: [[items]]
                        unique_items.update(user_data[0])
                    else:
                        # Direct list: [items]
                        unique_items.update(user_data)



        # Get all user IDs for this model
        all_user_ids = set()
        for method_name in method_data:
            all_user_ids.update(method_data[method_name].keys())

        print(f"  Total unique users for {model_name}: {len(all_user_ids)}")
        print(f"\n  ⚠️  CRITICAL PARAMETERS:")
        print(f"    top_k_resample (input to social choice): {config.top_k_resample}")
        print(f"    top_k_eval (output winners): {config.top_k_eval}")
        print()

        # Run each social choice method for this model
        for sc_method in ["borda", "schulze"]:
            print(f"\n{'='*60}")
            print(f"Running {sc_method.upper()} aggregation for {model_name}")
            print(f"{'='*60}")
            run_and_save_method(
                sc_method=sc_method,
                model_name=model_name,
                model_dir=paths['model_dir'],
                method_data=method_data,
                all_user_ids=all_user_ids,
                n_seats=config.top_k_eval
            )

    print("\n" + "="*60)
    print("All social choice aggregation results saved successfully!")
    print("="*60)


if __name__ == "__main__":
    main()