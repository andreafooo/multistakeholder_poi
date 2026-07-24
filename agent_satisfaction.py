import os
import json
from tqdm import tqdm
from evaluation_metrics import agent_agreement
from social_choice_aggregation import (
    get_paths_for_sc_input,
    get_user_recommendations,
    get_sc_output_dir,
)
from globals import (
    available_datasets,
    methods_to_aggregate,
    boosting_methods,
    top_k_eval,
    BASE_DIR,
    recommendation_dirpart,
)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def evaluate_agent_agreement_for_model(dataset, model_name, paths, sc_methods, k=top_k_eval):
    """
    For each user, compares every agent's own re-ranking (baseline, cp_min_js, geo, mmr)
    against the delivered social choice output, using nDCG-style agreement.
    Returns {sc_method: {agent_name: mean_agreement}}.
    """
    method_data = {}
    for method_name in methods_to_aggregate:
        file_path = paths[method_name]
        if not os.path.exists(file_path):
            print(f"  Missing {method_name} recommendations for {model_name}, skipping method")
            continue
        method_data[method_name] = load_json(file_path)

    model_dir = paths["model_dir"]
    results = {}

    for sc_method in sc_methods:
        sc_dir = get_sc_output_dir(dataset=dataset, model_dir=model_dir, sc_method=sc_method)
        sc_file = os.path.join(sc_dir, "top_k_recommendations.json")
        if not os.path.exists(sc_file):
            continue

        sc_data = load_json(sc_file)
        per_agent_scores = {method_name: [] for method_name in method_data}

        for user_id in tqdm(sc_data.keys(), desc=f"{model_name}-{sc_method} agreement", leave=False):
            delivered_list = get_user_recommendations(sc_data, user_id, top_k_resample=k)
            if not delivered_list:
                continue

            for method_name, data in method_data.items():
                agent_list = get_user_recommendations(data, user_id, top_k_resample=k)
                if not agent_list:
                    continue
                score = agent_agreement(agent_list, delivered_list, k=k)
                per_agent_scores[method_name].append(score)

        results[sc_method] = {
            method_name: (sum(scores) / len(scores) if scores else None)
            for method_name, scores in per_agent_scores.items()
        }

    return results


def main():
    for dataset in available_datasets:
        print(f"\n{'='*60}\nAGENT AGREEMENT | {dataset}\n{'='*60}")
        sc_dict = get_paths_for_sc_input(dataset)

        # Evaluate against whatever social choice outputs actually exist on disk:
        # plain "borda"/"schulze" (equal-weight) and/or the boosted variants.
        sc_methods = ["borda", "schulze"] + boosting_methods

        for model_name, paths in sc_dict.items():
            print(f"\n--- {model_name} ---")
            results = evaluate_agent_agreement_for_model(dataset, model_name, paths, sc_methods)
            print(json.dumps(results, indent=2))

            out_dir = os.path.join(BASE_DIR, f"{dataset}_dataset", recommendation_dirpart, paths["model_dir"])
            out_file = os.path.join(out_dir, "agent_agreement.json")
            with open(out_file, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Saved agent agreement results to {out_file}")


if __name__ == "__main__":
    main()
