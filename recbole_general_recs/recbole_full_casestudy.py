from recbole.quick_start import run_recbole
from recbole.utils.case_study import full_sort_topk
from recbole.quick_start import load_data_and_model
import pandas as pd
import json
import os
import yaml
import glob

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import BASE_DIR

""" In case of an error, comment out #from kmeans_pytorch import kmeans in the recbole package: recbole/model/general_recommender/ldiffrec.py """


datasets = ["foursquaretky_sample"]


models = ["BPR"]


def extract_test_data(model_file):
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file
    )

    print("Extracting test data...")

    for batch_data in test_data:
        print("Batch data structure:", batch_data)


def find_newest_model(directory):
    """If there are several model files in the directory, this function will return the newest one."""
    files = glob.glob(os.path.join(directory, "*"))
    if not files:
        return None

    newest_file = max(files, key=os.path.getctime)
    return newest_file


def run_configurations(config):
    output_dict = run_recbole(config_file_list=[config])
    directory = "saved/"
    newest_model_file = find_newest_model(directory)

    if newest_model_file:
        model_file = newest_model_file.split("/")[-1]
    else:
        print("No model files found in the directory.")

    with open(config, "r") as file:
        config_dict = yaml.safe_load(file)

    with open(config, "r") as file:
        config_dict = yaml.safe_load(file)

    OUTPUT_DIR = (
        f"{BASE_DIR}{config_dict['dataset'].split('_')[0]}_dataset/recommendations/"
    )

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df = pd.read_csv(
        f"recbole_general_recs/dataset/{config_dict['dataset']}/{config_dict['dataset']}.test.inter",
        sep="\t",
    )
    user_ids = list(set(df["user_id:token"].values.tolist()))

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=f"saved/{model_file}",
    )

    error_count = 0
    recommendations = {}

    for uid in user_ids:
        try:
            # Convert external user ID to internal ID
            uid_series = dataset.token2id(dataset.uid_field, [uid])

            # Compute top-k recommendations
            topk_score, topk_iid_list = full_sort_topk(
                uid_series, model, test_data, k=150, device=config["device"]
            )

            if topk_score is None or topk_iid_list is None:
                print(f"Model did not return valid predictions for user {uid}")
                continue

            # print(f"TOP K SCORES for {uid} :")
            # print(topk_score)  # scores of top 10 items
            # print(topk_iid_list)  # internal id of top 10 items
            # Convert internal item IDs to external tokens
            external_item_list = dataset.id2token(
                dataset.iid_field, topk_iid_list.cpu()
            )

            # Print or store recommendations
            # print(f"EXTERNAL TOKENS OF TOP k ITEMS FOR USER {uid} :")
            # print(external_item_list)

            # Computing the full scores
            # score = full_sort_scores(uid_series, model, test_data, device=config["device"])
            # print("Score of all items")
            # print(score)

            # print(
            #     score[0, dataset.token2id(dataset.iid_field, ["242", "302"])]
            # )  # score of item ['242', '302'] for user '196'.

            recommendations[uid] = [
                {"item_id": item_id, "score": score}
                for item_id, score in zip(
                    external_item_list.tolist(), topk_score.cpu().tolist()
                )
            ]

        except Exception as e:
            print(e)
            print(f"Error for user {uid}: {e}")
            error_count += 1
            continue

    print(
        f"The error count was {error_count}, which is {error_count / len(user_ids) * 100:.2f}% of the total users."
    )

    model_file_cleaned = model_file.split(".")[0]

    FINAL_OUTPUT_DIR = f"{OUTPUT_DIR + model_file_cleaned}/"

    if not os.path.exists(FINAL_OUTPUT_DIR):
        os.makedirs(FINAL_OUTPUT_DIR)

    with open(f"{FINAL_OUTPUT_DIR}top_k_recommendations.json", "w") as f:
        json.dump(recommendations, f, indent=4)

    with open(f"{FINAL_OUTPUT_DIR}config.yaml", "w") as f:
        yaml.dump(config_dict, f, indent=4)

    with open(f"{FINAL_OUTPUT_DIR}general_evaluation.json", "w") as f:
        json.dump(output_dict, f, indent=4)


if __name__ == "__main__":
    # config = "config_test.yaml" # To-Do: add if clause if there is no specific config_test because they are only created after hyperopt
    for dataset in datasets:
        for model in models:
            config = f"recbole_general_recs/config/{dataset}/{model}/config_test.yaml"
            run_configurations(config)
            print(f"Finished {model} for {dataset}")
            print("--------------------------------------------------------")
    print("All models finished.")
