import json
import os
from datetime import datetime
from config import BaseConfig

config = BaseConfig


def dataset_metadata(dataset, recommendation_dirpart, base_dir):
    """Extract metadata for each dataset and model"""
    data = []

    # Ensure only directories are listed
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
    """
    Create output directories
    """
    model_directories = {}
    methods = config.methods_to_aggregate  # right now we only use the recs from the RS output without regarding CP

    for result in data:
        model_name = result["model"]
        model_directories[model_name] = {}
        
        for method in methods:
            path = os.path.join(base_dir, f"{dataset}_dataset", recommendation_dirpart, result['directory'], method, "top_k_recommendations.json")
            model_directories[model_name][method] = path

    return model_directories


def get_paths_for_sc_input(dataset=config.available_datasets[0], recommendation_dirpart=config.recommendation_dirpart, base_dir=config.RECS_BASE_DIR):
    recommender_metadata = dataset_metadata(dataset=dataset, recommendation_dirpart=recommendation_dirpart, base_dir=base_dir)
    model_dirs = create_model_directories(dataset=dataset, data=recommender_metadata, recommendation_dirpart=recommendation_dirpart, base_dir=base_dir)
    
    sc_recs = {}
    for element in recommender_metadata:
        print(element)
        if element["model"] in config.sc_models:
            model_name = element["model"]
            # Create a dict with baseline and cp_min_js paths for this model
            sc_recs[model_name] = {
                "baseline": model_dirs[model_name]["baseline"],
                "cp_min_js": model_dirs[model_name]["cp_min_js"],
                "geo": model_dirs[model_name]["geo"],
                "mmr": model_dirs[model_name]["mmr"],
                "model_dir": element['directory']  # Store the model directory path
            }

    print(sc_recs)
    return sc_recs

def get_sc_output_dir(dataset, model_dir, sc_method, recommendation_dirpart=config.recommendation_dirpart, base_dir=config.RECS_BASE_DIR):
    """
    Create output directory path for social choice results within the model directory
    
    Args:
        dataset: Dataset name
        model_dir: The model's directory name (e.g., 'foursquaretky_sample-BPR-Apr-21-2025_21-54-55')
        sc_method: Social choice method name (e.g., 'borda', 'plurality', 'schulze')
        
    Returns:
        Path like: .../foursquaretky_sample-BPR-Apr-21-2025_21-54-55/borda/
    """
    sc_output_dir = os.path.join(base_dir, f'{dataset}_dataset', recommendation_dirpart, model_dir, sc_method)
    os.makedirs(sc_output_dir, exist_ok=True)
    return sc_output_dir

def timestamp_creator():
    """Generate a base timestamp as str"""
    return datetime.now().strftime("%b-%d-%Y_%H-%M-%S")