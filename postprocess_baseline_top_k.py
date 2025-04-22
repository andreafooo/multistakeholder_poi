import os
import json
from globals import BASE_DIR, available_datasets, top_k_eval, recommendation_dirpart

# Constants
general_models = ["BPR"]
context_models = ["LORE", "USG"]
############################################################################################################


def process_top_k_json(input_file, output_file, k=top_k_eval):
    """
    Process top-k recommendations from a JSON file, keeping only the item IDs for each user.

    Args:
    - input_file (str): Path to the original JSON file.
    - output_file (str): Path to save the processed JSON file.
    - k (int): Number of top-k recommendations to keep per user.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    try:
        with open(input_file, "r") as infile:
            data = json.load(infile)

        top_k_result = {}
        for user_id, recommendations in data.items():
            if recommendations and isinstance(recommendations[0], dict):
                item_ids = [rec["item_id"] for rec in recommendations][:k]
                top_k_result[user_id] = item_ids

        with open(output_file, "w") as outfile:
            json.dump(top_k_result, outfile, indent=4)
        print(f"Processed file saved to: {output_file}")

    except Exception as e:
        print(f"Error processing {input_file}: {e}")


def dataset_metadata(dataset, recommendation_dirpart):
    """Extract metadata for each dataset and model"""
    data = []

    # Ensure only directories are listed
    recs = [
        d
        for d in os.listdir(f"{BASE_DIR}{dataset}_dataset/{recommendation_dirpart}")
        if os.path.isdir(
            os.path.join(BASE_DIR, f"{dataset}_dataset", recommendation_dirpart, d)
        )
    ]

    for dir in recs:
        json_file = f"{BASE_DIR}{dataset}_dataset/{recommendation_dirpart}/{dir}/general_evaluation.json"

        if not os.path.exists(json_file):
            # print(f"Skipping {json_file} - File not found.")
            continue

        with open(json_file, "r") as f:
            eval_data = json.load(f)

        test_results = eval_data.get("test_result", {})
        test_results["directory"] = dir

        # Extracting model and type
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
    """Create model directories for each method"""
    model_directories = {}
    methods = ["baseline", "cp", "cp_min_js", "upd"]

    def recommender_dir_combiner(dataset, modelpart, method):
        return f"{base_dir}{dataset}_dataset/{recommendation_dirpart}/{modelpart}/{method + '/' if method != 'baseline' else ''}top_k_recommendations.json"

    for result in data:
        model_name = result["model"]
        model_directories[model_name] = {}

        for method in methods:
            model_directories[model_name][method] = recommender_dir_combiner(
                dataset, result["directory"], method
            )

    return model_directories


def main():
    for dataset in available_datasets:
        data = dataset_metadata(dataset, recommendation_dirpart)
        model_dirs = create_model_directories(
            dataset, data, BASE_DIR, recommendation_dirpart
        )

        for result in data:
            input_path = model_dirs[result["model"]]["baseline"]
            directory = input_path.rsplit("/", 1)[0]
            output_dir = os.path.join(directory, "baseline")
            output_path = os.path.join(output_dir, "top_k_recommendations.json")
            process_top_k_json(input_path, output_path, k=top_k_eval)


if __name__ == "__main__":
    main()
