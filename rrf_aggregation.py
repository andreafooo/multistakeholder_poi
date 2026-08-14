import os
import json

from ranx import Run, fuse

from social_choice_aggregation import (
    get_user_recommendations,
    get_paths_for_sc_input,
    get_sc_output_dir,
)
from globals import (
    top_k_eval,
    available_datasets,
    sc_models,
    methods_to_aggregate,
    rrf_k,
)


def build_method_run(method_data, method_name, all_user_ids):
    """
    One ranx Run for a single stakeholder method across all users. Scores are
    just (list_length - rank) since RRF only needs each list's ordering, not
    cardinal scores -- ranx.fuse(method="rrf") re-derives everything from
    position. Users missing from this method's output get an empty result
    set, contributing nothing to the fused score for that user -- same
    "absent methods don't count" behavior as the votekit ballots in
    social_choice_aggregation.py.
    """
    run_dict = {}
    for user_id in all_user_ids:
        rec_list = get_user_recommendations(method_data.get(method_name, {}), user_id)
        run_dict[user_id] = {item: float(len(rec_list) - idx) for idx, item in enumerate(rec_list)}
    return Run.from_dict(run_dict)


def run_and_save_rrf(model_name, model_dir, method_data, all_user_ids, dataset, n_seats=top_k_eval):
    """
    Reciprocal Rank Fusion (Cormack, Clarke & Buttcher, "Reciprocal Rank
    Fusion Outperforms Condorcet and Individual Rank Learning Methods",
    SIGIR 2009) over the 4 stakeholder lists, via ranx
    (github.com/AmenRa/ranx) -- an independently maintained rank-fusion
    library -- rather than a hand-rolled weighted sum. Saves to 'rrf/'.
    """
    runs = [
        build_method_run(method_data, method_name, all_user_ids)
        for method_name in methods_to_aggregate
    ]
    combined = fuse(runs=runs, norm=None, method="rrf", params={"k": rrf_k})

    results = {
        user_id: list(scores.keys())[:n_seats]
        for user_id, scores in combined.to_dict().items()
        if scores
    }

    output_dir = get_sc_output_dir(dataset=dataset, model_dir=model_dir, sc_method="rrf")
    output_file = os.path.join(output_dir, "top_k_recommendations.json")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nrrf results for {model_name} saved to: {output_file}")
    print(f"Total users processed: {len(results)}")


def main():
    for dataset in available_datasets:
        print(f"\n{'='*60}")
        print("RECIPROCAL RANK FUSION")
        print(f"{'='*60}\n")

        sc_dict = get_paths_for_sc_input(dataset)
        print(f"Models to process: {list(sc_dict.keys())}")
        print(f"Expected models: {sc_models}")

        for model_name, paths in sc_dict.items():
            print(f"\n{'='*60}")
            print(f"PROCESSING MODEL: {model_name}")
            print(f"{'='*60}")

            method_data = {}
            for method_name in methods_to_aggregate:
                file_path = paths[method_name]
                if not os.path.exists(file_path):
                    print(f"ERROR: File not found for {model_name}/{method_name}: {file_path}")
                    continue
                with open(file_path, "r") as f:
                    method_data[method_name] = json.load(f)

            all_user_ids = set()
            for method_name in method_data:
                all_user_ids.update(method_data[method_name].keys())

            run_and_save_rrf(
                model_name=model_name,
                model_dir=paths["model_dir"],
                method_data=method_data,
                all_user_ids=all_user_ids,
                dataset=dataset,
            )

        print("\n" + "="*60)
        print("All RRF aggregation results saved successfully!")
        print("="*60)


if __name__ == "__main__":
    main()
