import pandas as pd
import json
import numpy as np
from globals import BASE_DIR, available_datasets
import os
import random
import traceback
from evaluation_metrics import *

pd.options.mode.copy_on_write = True


# global settings
top_k_resample = 150
top_k_eval = 10
gridsearch = False  # set to true for new datasets
save_upd = False
valid_popularity = "item_pop"
recommendation_dirpart = "recommendations"


# Define the datasets you want to process
def dataset_metadata(dataset, recommendation_dirpart=recommendation_dirpart):
    data = []

    recs = os.listdir(f"{BASE_DIR}{dataset}_dataset/{recommendation_dirpart}")
    if ".DS_Store" in recs:
        recs.remove(".DS_Store")

    for dir in recs:
        json_file = f"{BASE_DIR}{dataset}_dataset/{recommendation_dirpart}/{dir}/general_evaluation.json"

        with open(json_file, "r") as f:
            eval_data = json.load(f)

        # Extract the test_result data and flatten it
        test_results = eval_data.get("test_result", {})
        test_results["directory"] = dir

        test_results["dataset"] = test_results["directory"].split("-")[0]
        test_results["model"] = test_results["directory"].split("-")
        if test_results["directory"].split("-")[1] == "debias":
            test_results["model_type"] = "debias"
            test_results["date"] = "-".join(test_results["directory"].split("-")[3:])

        elif test_results["directory"].split("-")[1] == "contextpoi":
            test_results["model_type"] = "contextpoi"
            test_results["date"] = "-".join(test_results["directory"].split("-")[3:])
        else:
            test_results["model_type"] = "general"
            test_results["date"] = "-".join(test_results["directory"].split("-")[2:])

        if test_results["model_type"] == "debias":
            test_results["model"] = test_results["model"][2]

        elif test_results["model_type"] == "contextpoi":
            test_results["model"] = test_results["model"][2]

        else:
            test_results["model"] = test_results["model"][1]

        if test_results["model"] == "MF":
            test_results["model_type"] = "general (via RecBole debias)"

        test_results["dataset"] = test_results["dataset"].split("_")[0]
        if test_results["model_type"] != "debias":
            data.append(test_results)

    return data


def unstack_recommendations(df):
    # Repeat each user_id for the length of their item_id:token list
    unstacked_df = df.explode(["item_id:token", "score"]).reset_index(drop=True)
    return unstacked_df


def create_base_recommendations(
    recommender_dir, top_k_resample=top_k_resample, top_k_eval=top_k_eval
):
    with open(recommender_dir) as f:
        data = json.load(f)

    base_recommendations = []

    for user, items in data.items():
        for item in items:
            base_recommendations.append(
                {
                    "user_id:token": user,
                    "item_id:token": item["item_id"],
                    "score": item["score"],
                }
            )

    base_df = pd.DataFrame(base_recommendations)
    base_df = unstack_recommendations(base_df)

    try:
        base_df["score"] = base_df.groupby("user_id:token")["score"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )
    except ZeroDivisionError:
        print("ZeroDivisionError, using unnormalized scores for this dataset")
        base_df["score"] = base_df["score"]

    # top k recommendations for resampling
    base_top_k_df = (
        base_df.groupby("user_id:token").head(top_k_resample).reset_index(drop=True)
    )

    # top k recommendations for evaluation
    base_eval_df = (
        base_df.groupby("user_id:token").head(top_k_eval).reset_index(drop=True)
    )

    return base_top_k_df, base_eval_df


def save_top_k(sorted_top_k_df, base_dir, reranking_method):
    grouped_data = (
        sorted_top_k_df.groupby("user_id:token")["item_id:token"].apply(list).to_dict()
    )

    save_path = os.path.join(base_dir, reranking_method)
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, "top_k_recommendations.json")

    with open(file_path, "w") as f:
        json.dump(grouped_data, f, indent=4)

    print(f"Saved recommendations to {file_path}")


def save_cp_metadata(base_dir, file, filename):
    save_path = os.path.join(base_dir, "cp")
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f"{filename}.json")

    with open(file_path, "w") as f:
        json.dump(file, f, indent=4)

    print(f"Saved recommendations to {file_path}")


def rerank_upd(
    dataset,
    base_df,
    top_k_eval,
    valid_popularity,
    dir_to_save,
    train_data,
    calibrate_on="mean",
):
    checkin_df = train_data.copy()

    # Calculate item popularity
    value_counts = checkin_df["item_id:token"].value_counts().reset_index()
    value_counts.columns = ["item_id:token", "count"]
    value_counts[valid_popularity] = value_counts["count"] / len(value_counts)
    checkin_df = checkin_df.merge(
        value_counts[["item_id:token", valid_popularity]],
        on="item_id:token",
        how="left",
    )

    # Assign item popularity groups
    checkin_df.sort_values(by=valid_popularity, ascending=False, inplace=True)
    item_popularity = checkin_df.drop_duplicates(subset="item_id:token", keep="first")[
        ["item_id:token", valid_popularity]
    ]

    h_group = item_popularity.head(int(len(item_popularity) * 0.2))
    h_group["item_pop_group"] = "h"
    t_group = item_popularity.tail(int(len(item_popularity) * 0.2))
    t_group["item_pop_group"] = "t"
    m_group = item_popularity[
        ~item_popularity["item_id:token"].isin(h_group["item_id:token"])
        & ~item_popularity["item_id:token"].isin(t_group["item_id:token"])
    ]
    m_group["item_pop_group"] = "m"

    item_popularity = pd.concat([h_group, m_group, t_group])
    item_popularity.sort_values(by=valid_popularity, inplace=True, ascending=False)

    checkin_df = checkin_df.merge(
        item_popularity[["item_id:token", "item_pop_group"]],
        on="item_id:token",
        how="left",
    )

    # Calculate user popularity tolerance score (UPTS)
    if calibrate_on == "mean":
        upts = (
            checkin_df.groupby("user_id:token")[valid_popularity].mean().reset_index()
        )
    else:
        upts = (
            checkin_df.groupby("user_id:token")[valid_popularity].median().reset_index()
        )

    upts.columns = ["user_id:token", "upts"]

    # Merge and calculate UPD
    merged_df = base_df.merge(upts, on="user_id:token").merge(
        item_popularity, on="item_id:token"
    )
    merged_df["upd"] = (merged_df[valid_popularity] - merged_df["upts"]).abs()

    # Extract calibrated top-k recommendations
    sorted_top_k_df = (
        merged_df.sort_values(by=["user_id:token", "upd"], ascending=True)
        .groupby("user_id:token")
        .head(top_k_eval)
        .reset_index(drop=True)
    )

    return sorted_top_k_df, item_popularity, upts


def open_ground_truth_user_group(dataset):
    # Stays the same across all models
    train_data = pd.read_csv(
        f"{BASE_DIR}{dataset}_dataset/processed_data_recbole/{dataset}_sample.train.inter",
        sep="\t",
    )
    test_data = pd.read_csv(
        f"{BASE_DIR}{dataset}_dataset/processed_data_recbole/{dataset}_sample.test.inter",
        sep="\t",
    )
    valid_data = pd.read_csv(
        f"{BASE_DIR}{dataset}_dataset/processed_data_recbole/{dataset}_sample.valid.inter",
        sep="\t",
    )
    # valid_data = pd.read_csv(f"{BASE_DIR}{dataset}_dataset/processed_data_recbole/{dataset}_sample.valid.inter", sep="\t") # originale struktur !!!
    train_data = pd.concat([train_data, valid_data])
    user_group_dir = f"{BASE_DIR}{dataset}_dataset/{dataset}_user_id_popularity.json"
    with open(user_group_dir) as f:
        user_groups = json.load(f)

    return train_data, test_data, user_groups


def recommender_dir_combiner(dataset, modelpart):
    top_k_dir = f"{BASE_DIR}{dataset}_dataset/{recommendation_dirpart}/{modelpart}/top_k_recommendations.json"
    recommendation_folder = (
        f"{BASE_DIR}{dataset}_dataset/{recommendation_dirpart}/{modelpart}"
    )
    return top_k_dir, recommendation_folder


def calculate_user_popularity_distributions(df, item_popularity):
    # df = df.merge(item_popularity, on="item_id:token", how="left")
    # Calculate mean, median, and variance of `business_popularity` for each user
    user_stats = (
        df.groupby("user_id:token")[valid_popularity]
        .agg(["mean", "median", "var"])
        .reset_index()
    )

    # Calculate normalized ratios of `item_pop_group` for each user
    pop_group_counts = (
        df.groupby(["user_id:token", "item_pop_group"]).size().unstack(fill_value=0)
    )

    # Normalize by row sum to get ratios
    pop_group_ratios = pop_group_counts.div(
        pop_group_counts.sum(axis=1), axis=0
    ).reset_index()

    # Merge the statistics and ratios into a single DataFrame
    user_pop_ratio_df = pd.merge(
        user_stats, pop_group_ratios, on="user_id:token", how="left"
    )

    # Rename columns for clarity
    user_pop_ratio_df.rename(
        columns={
            "h": "h_ratio",
            "m": "m_ratio",
            "t": "t_ratio",
            "mean": "mean_pop",
            "median": "median_pop",
            "var": "variance_pop",
        },
        inplace=True,
    )

    num_interactions = (
        df.groupby("user_id:token").size().reset_index(name="num_interactions")
    )

    # Merge the num_interactions into the user_pop_ratio_df
    user_pop_ratio_df = pd.merge(
        user_pop_ratio_df, num_interactions, on="user_id:token", how="left"
    )

    return user_pop_ratio_df


def get_profile_and_recommended_ratios_for_js(test_item_groups, test_user_profile):
    recommended_ratios_nested = test_item_groups.to_dict()
    recommended_ratios = {
        key: list(value.values())[0] for key, value in recommended_ratios_nested.items()
    }

    # Extract user profile ratios
    profile_ratios = {
        "h_ratio": test_user_profile.get("h_ratio", pd.Series([0])).iloc[0],
        "m_ratio": test_user_profile.get("m_ratio", pd.Series([0])).iloc[0],
        "t_ratio": test_user_profile.get("t_ratio", pd.Series([0])).iloc[0],
    }

    return recommended_ratios, profile_ratios


def rerank_for_user(
    initial_list, scores, item_popularity, user_profile, delta, k, user_id
):
    """
    Perform the re-ranking for a single user using the CP algorithm.
    Implementation based on: https://github.com/rUngruh/mitigatingPopularityBiasInMRS/blob/main/studytool/Tool-Module/scripts/LFMRecommendations/Models/mitigation.py
    """
    reranked_list = []
    category_counts = {"h": 0, "m": 0, "t": 0}
    score_count = 0

    # Iteratively build the re-ranked list for the user
    for i in range(k):
        criterion = marginal_relevances(
            score_count,
            scores,
            item_popularity,
            category_counts,
            len(reranked_list),
            user_profile,
            delta,
        )

        # Exclude zero values from the criterion
        non_zero_indices = [idx for idx, value in enumerate(criterion) if value != 0]

        if non_zero_indices:
            # Extract the non-zero values
            non_zero_values = [criterion[idx] for idx in non_zero_indices]

            if all(
                value < 0 for value in non_zero_values
            ):  # All remaining values are negative
                # Choose the value closest to zero
                closest_to_zero_value = min(non_zero_values, key=lambda x: abs(x))
                selected_idx = non_zero_indices[
                    non_zero_values.index(closest_to_zero_value)
                ]
            else:
                selected_idx = non_zero_indices[np.argmax(non_zero_values)]
        else:
            # Fallback: All values are zero
            selected_idx = np.argmax(criterion)

        score_count += scores[selected_idx]
        reranked_list.append(initial_list[selected_idx])

        category_counts[item_popularity[selected_idx]] += 1

        del initial_list[selected_idx]
        del scores[selected_idx]
        del item_popularity[selected_idx]

    return reranked_list


def marginal_relevances(
    score_count,
    item_scores,
    item_popularities,
    category_counts,
    list_len,
    user_profile,
    delta,
):
    """
    Computes the marginal relevance, the criterion for CP
    Implementation based on: https://github.com/rUngruh/mitigatingPopularityBiasInMRS/blob/main/studytool/Tool-Module/scripts/LFMRecommendations/Models/mitigation.py
    """
    relevances = np.zeros(len(item_scores))
    recommendation_counts = pd.DataFrame(
        {
            "h_ratio": [category_counts["h"]],
            "m_ratio": [category_counts["m"]],
            "t_ratio": [category_counts["t"]],
        }
    )
    computed_categories = set()

    for i, (score, popularity) in enumerate(zip(item_scores, item_popularities)):
        if popularity in computed_categories:
            continue  # Avoid duplicate calculations for the same popularity class

        # print(f"Item {i}: score={score}, popularity={popularity}")
        computed_categories.add(popularity)

        # Increment the count temporarily
        recommendation_counts[popularity + "_ratio"] += 1
        recommendation_ratios = recommendation_counts / (list_len + 1)

        rec_ratios, profile_ratios = get_profile_and_recommended_ratios_for_js(
            recommendation_ratios, user_profile
        )
        js_divergence = jensen_shannon(profile_ratios, rec_ratios)

        relevances[i] = (1 - delta) * (score_count + score) - delta * js_divergence

        recommendation_counts[popularity + "_ratio"] -= 1
    return relevances


def get_individual_user_data(df, user_profiles, user_id):
    test_user_id = user_id
    test_recs = df.loc[df["user_id:token"] == test_user_id][
        "item_id:token"
    ].values.tolist()
    test_scores = df.loc[df["user_id:token"] == test_user_id]["score"].values.tolist()
    test_item_pops = df.loc[df["user_id:token"] == test_user_id][
        valid_popularity
    ].values.tolist()
    test_item_groups = df.loc[df["user_id:token"] == test_user_id][
        "item_pop_group"
    ].values.tolist()
    test_user_profile = user_profiles.loc[
        user_profiles["user_id:token"] == test_user_id
    ]
    return (
        test_scores,
        test_recs,
        test_item_pops,
        test_item_groups,
        test_user_id,
        test_user_profile,
    )


def rerank_cp_all_users(df, user_profiles, top_k_eval, delta):
    reranked_results = {}

    for i, user_id in enumerate(df["user_id:token"].unique()):
        (
            test_scores,
            test_recs,
            test_item_pops,
            test_item_groups,
            test_user_id,
            test_user_profile,
        ) = get_individual_user_data(df, user_profiles, user_id)
        reranked_list = rerank_for_user(
            test_recs,
            test_scores,
            test_item_groups,
            test_user_profile,
            delta,
            top_k_eval,
            test_user_id,
        )
        reranked_results[test_user_id] = reranked_list

    cp_results = pd.DataFrame([reranked_results]).T.reset_index()
    cp_results.columns = ["user_id:token", "item_id:token"]
    cp_reranked_df = cp_results.explode("item_id:token").reset_index(drop=True)

    return cp_reranked_df


def sample_user_groups(user_groups, sample_size=100):
    sampled_groups = {}
    for group, ids in user_groups.items():
        if len(ids) >= sample_size:
            sampled_groups[group] = random.sample(ids, sample_size)
        else:
            print(
                f"Warning: Group '{group}' has less than {sample_size} users. Sampling all users."
            )
            sampled_groups[group] = ids
    return sampled_groups


def calculate_group_ratios(user_groups, df):
    """Use for plotting"""
    group_results = {}

    for group_name, user_ids in user_groups.items():
        reranked_group_df = df.loc[df["user_id:token"].isin(user_ids)]

        columns_to_check = ["m_ratio", "t_ratio", "h_ratio"]

        for column in columns_to_check:
            if column not in reranked_group_df.columns:
                reranked_group_df[column] = 0

        means = reranked_group_df[["h_ratio", "m_ratio", "t_ratio"]].mean().to_dict()

        group_results[group_name] = means

    return group_results


def get_extreme(results, metric, mode="max", use_abs=False):
    """
    Finds the delta that gives the extreme value (argmax or argmin) for a given metric across all groups.

    Parameters:
    - results (dict): The nested results dictionary.
    - metric (str): The metric to compute the extreme for (e.g., "ndcg", "arp", "poplift", "js_divergence").
    - mode (str): Either "max" (default) or "min" to compute the argmax or argmin.
    - use_abs (bool): Whether to compute extremes based on the absolute value of the metric.

    Returns:
    - dict: A dictionary with group names as keys and the corresponding delta and extreme value.
    """
    if mode not in ["max", "min"]:
        raise ValueError("Mode must be either max or min.")

    extreme_dict = {}
    compare = max if mode == "max" else min
    extreme_value_init = float("-inf") if mode == "max" else float("inf")

    for group_name in next(
        iter(results.values())
    ).keys():  # Get group names from the first delta
        extreme_value = extreme_value_init
        best_delta = None

        for delta, group_data in results.items():
            if metric in group_data[group_name]:  # Check if the metric exists
                value = (
                    abs(group_data[group_name][metric])
                    if use_abs
                    else group_data[group_name][metric]
                )
                if compare(value, extreme_value) == value:
                    extreme_value = value
                    best_delta = delta

        extreme_dict[group_name] = best_delta

    return extreme_dict


def cp_gridsearch(
    base_resample,
    user_profiles,
    top_k_eval,
    item_popularity,
    train_data,
    test_data,
    user_groups,
    upts,
    recommendation_folder,
):
    deltas = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    results = {}
    gridsearch_best_deltas = {}
    for i, delta in enumerate(deltas):
        print("Processing delta:", delta)
        results[delta] = {}
        reranked_df = rerank_cp_all_users(
            base_resample, user_profiles, top_k_eval, delta=delta
        )
        reranked_df = reranked_df.merge(item_popularity, on="item_id:token")
        (
            base_arp_scores,
            base_poplift_scores,
            calibrated_arp_scores,
            calibrated_poplift_scores,
            base_ndcg_scores,
            calibrated_ndcg_scores,
        ) = calculate_deltas(
            test_data,
            base_resample,
            reranked_df,
            item_popularity,
            upts,
            valid_popularity,
            top_k_eval,
        )
        reranked_df_user = calculate_user_popularity_distributions(
            reranked_df, item_popularity
        )
        user_profiles = calculate_user_popularity_distributions(
            train_data, item_popularity
        )

        group_hmt_means_up = calculate_group_ratios(user_groups, user_profiles)
        group_hmt_means_reranked = calculate_group_ratios(user_groups, reranked_df_user)

        group_means, _, _, _ = evaluation_user_group_means(
            calibrated_ndcg_scores,
            calibrated_arp_scores,
            calibrated_poplift_scores,
            user_groups,
            reranked_df,
        )

        print("group_means contents:")
        print(json.dumps(group_means, indent=2))
        for group_name, user_ids in user_groups.items():
            js = jensen_shannon(
                group_hmt_means_up[group_name], group_hmt_means_reranked[group_name]
            )
            harmonic_mean = (
                group_means[group_name]["ndcg"]
                * (1 - js)
                / (group_means[group_name]["ndcg"] + (1 - js))
            )

            results[delta][group_name] = {
                "js_divergence": js,
                "harmonic_mean": harmonic_mean,
                "ndcg": group_means[group_name]["ndcg"],
                "arp": group_means[group_name]["arp"],
                "poplift": group_means[group_name]["poplift"],
            }

    gridsearch_best_deltas["ndcg"] = get_extreme(results, "ndcg")
    gridsearch_best_deltas["arp"] = get_extreme(results, "arp", mode="min")
    gridsearch_best_deltas["harmonic_mean"] = get_extreme(results, "harmonic_mean")
    gridsearch_best_deltas["js"] = get_extreme(results, "js_divergence", mode="min")
    gridsearch_best_deltas["poplift"] = get_extreme(
        results, "poplift", mode="min", use_abs=True
    )

    save_cp_metadata(
        recommendation_folder, gridsearch_best_deltas, "gridsearch_best_deltas"
    )
    save_cp_metadata(recommendation_folder, results, "gridsearch_results")

    return gridsearch_best_deltas


def main(available_datasets):
    for dataset in available_datasets:
        data = dataset_metadata(dataset)
        for result in data:
            try:
                print(
                    f"Processing model {result['model']} on dataset {result['dataset']}"
                )

                # Combine the baseline directory and check if the cp directory exists
                baseline_topk_dir, basedir = recommender_dir_combiner(
                    dataset, result["directory"]
                )

                train_data, test_data, user_groups = open_ground_truth_user_group(
                    dataset
                )
                base_resample, base_eval = create_base_recommendations(
                    baseline_topk_dir,
                    top_k_resample=top_k_resample,
                    top_k_eval=top_k_eval,
                )

                upd_eval, item_popularity, upts = rerank_upd(
                    dataset,
                    base_resample,
                    top_k_eval,
                    valid_popularity,
                    dir_to_save=basedir,
                    train_data=train_data,
                    calibrate_on="mean",
                )

                if save_upd:
                    save_top_k(upd_eval, basedir, "upd")

                ##### Using a smaller subsample for testing deltas in cp
                # user_groups = sample_user_groups(
                #     user_groups, sample_size=50
                # )  # for testing
                # sampled_user_ids = set(
                #     id_ for group in user_groups.values() for id_ in group
                # )  # for testing
                # dataframes_to_filter = [
                #     train_data,
                #     test_data,
                #     base_resample,
                #     base_eval,
                #     upts,
                # ]  # for testing
                # filtered_dataframes = [
                #     df.loc[df["user_id:token"].isin(sampled_user_ids)]
                #     for df in dataframes_to_filter
                # ]  # for testing
                # train_data, test_data, base_resample, base_eval, upts = (
                #     filtered_dataframes  # for testing
                # )
                #### End of using a smaller subsample for testing deltas in cp

                all_user_ids = (
                    set(user_groups["high"])
                    | set(user_groups["medium"])
                    | set(user_groups["low"])
                )
                user_groups["all"] = list(all_user_ids)

                base_resample = base_resample.merge(item_popularity, on="item_id:token")
                test_data = test_data.merge(
                    item_popularity, on="item_id:token", how="left"
                )
                train_data = train_data.merge(
                    item_popularity, on="item_id:token", how="left"
                )
                user_profiles = calculate_user_popularity_distributions(
                    train_data, item_popularity
                )

                if gridsearch:
                    cp_gridsearch_best_deltas = cp_gridsearch(
                        base_resample,
                        user_profiles,
                        top_k_eval,
                        item_popularity,
                        train_data,
                        test_data,
                        user_groups,
                        upts,
                        basedir,
                    )
                else:
                    cp_gridsearch_best_deltas = json.load(
                        open(f"{basedir}/cp/gridsearch_best_deltas.json")
                    )

                print(f"Best deltas: {cp_gridsearch_best_deltas}")

                cp_dfs = []
                for group in user_groups.keys():
                    if group != "all":
                        base_resample_group = base_resample.loc[
                            base_resample["user_id:token"].isin(user_groups[group])
                        ]
                        user_profiles_group = user_profiles.loc[
                            user_profiles["user_id:token"].isin(user_groups[group])
                        ]
                        reranked_df_group = rerank_cp_all_users(
                            base_resample_group,
                            user_profiles_group,
                            top_k_eval,
                            delta=cp_gridsearch_best_deltas["harmonic_mean"][group],
                        )
                        cp_dfs.append(reranked_df_group)

                reranked_df = pd.concat(cp_dfs)
                save_top_k(reranked_df, basedir, "cp")

                cp_min_dfs = []
                for group in user_groups.keys():
                    if group != "all":
                        base_resample_group = base_resample.loc[
                            base_resample["user_id:token"].isin(user_groups[group])
                        ]
                        user_profiles_group = user_profiles.loc[
                            user_profiles["user_id:token"].isin(user_groups[group])
                        ]
                        reranked_df_group = rerank_cp_all_users(
                            base_resample_group,
                            user_profiles_group,
                            top_k_eval,
                            delta=cp_gridsearch_best_deltas["js"][group],
                        )
                        cp_min_dfs.append(reranked_df_group)

                reranked_df = pd.concat(cp_min_dfs)
                save_top_k(reranked_df, basedir, "cp_min_js")

            # except KeyError:
            #     print(
            #         f"Error: In Model {result['model']} on dataset {result['dataset']}"
            #     )

            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                # continue


if __name__ == "__main__":
    main(available_datasets)
