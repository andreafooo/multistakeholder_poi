import numpy as np
import math


def ndcg(test_data, df, top_k_eval=10):
    test_data = test_data.copy()  # Prevent modifying the original data
    ndcg_scores = {}

    for user_id in df["user_id:token"].unique():
        user_recommendations = df[df["user_id:token"] == user_id]
        recommended_items = user_recommendations["item_id:token"].tolist()[:top_k_eval]

        true_items = test_data[test_data["user_id:token"] == user_id][
            "item_id:token"
        ].values
        true_relevance = [1 if item in true_items else 0 for item in recommended_items]

        # Compute DCG@k
        dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(true_relevance))

        # Compute iDCG@k
        idcg = sum(
            1 / np.log2(idx + 2) for idx in range(min(len(true_items), top_k_eval))
        )

        ndcg_scores[user_id] = dcg / idcg if idcg > 0 else 0

    return ndcg_scores


def calculate_arp_poplift(
    df, item_popularity, user_profile_popularity, valid_popularity
):
    # df = df.merge(item_popularity, on="item_id:token", how="left")
    df = df.merge(user_profile_popularity, on="user_id:token", how="left")

    arp_scores = df.groupby("user_id:token")[valid_popularity].mean().to_dict()
    upts_scores = df.groupby("user_id:token")["upts"].mean().to_dict()

    # Calculate poplift as the percentage deviation (ARP - UPP) / UPP for each user
    poplift_scores = {
        user_id: ((arp_scores[user_id] - upts_scores[user_id]) / upts_scores[user_id])
        for user_id in arp_scores
        if upts_scores[user_id] != 0
    }

    return arp_scores, poplift_scores


def calculate_deltas(
    test_data,
    base_df,
    calibrated_df,
    item_popularity,
    user_profile_popularity,
    valid_popularity,
    top_k_eval,
):
    base_ndcg_scores = ndcg(test_data, base_df, top_k_eval)
    calibrated_ndcg_scores = ndcg(test_data, calibrated_df, top_k_eval)

    base_arp_scores, base_poplift_scores = calculate_arp_poplift(
        base_df, item_popularity, user_profile_popularity, valid_popularity
    )
    calibrated_arp_scores, calibrated_poplift_scores = calculate_arp_poplift(
        calibrated_df, item_popularity, user_profile_popularity, valid_popularity
    )

    # print(
    #     f"nDCG base: {sum(base_ndcg_scores.values()) / len(base_ndcg_scores):,.5f}; calibrated: {sum(calibrated_ndcg_scores.values()) / len(calibrated_ndcg_scores):,.5f}"
    # )
    # print(
    #     f"ARP base: {sum(base_arp_scores.values()) / len(base_arp_scores):,.5f}; calibrated: {sum(calibrated_arp_scores.values()) / len(calibrated_arp_scores):,.5f}"
    # )
    # print(
    #     f"Poplift base: {sum(base_poplift_scores.values()) / len(base_poplift_scores):,.5f}; calibrated: {sum(calibrated_poplift_scores.values()) / len(calibrated_poplift_scores):,.5f}"
    # )
    # print("*" * 50)

    return (
        base_arp_scores,
        base_poplift_scores,
        calibrated_arp_scores,
        calibrated_poplift_scores,
        base_ndcg_scores,
        calibrated_ndcg_scores,
    )


def jensen_shannon(profile_ratios, recommended_ratios):
    """
    Computes the Jensen-Shannon divergence for the given recommendations and user profile.
    """
    epsilon = 1e-8  # Small non-zero value

    # Compute JS divergence
    A = 0
    B = 0
    for c in ["h_ratio", "m_ratio", "t_ratio"]:
        profile_ratio = profile_ratios[c]
        recommended_ratio = recommended_ratios[c]

        if profile_ratio == 0:
            profile_ratio += epsilon

        if recommended_ratio == 0:
            recommended_ratio += epsilon

        A += profile_ratio * math.log2(
            (2 * profile_ratio) / (profile_ratio + recommended_ratio)
        )
        B += recommended_ratio * math.log2(
            (2 * recommended_ratio) / (profile_ratio + recommended_ratio)
        )

    js = (A + B) / 2

    return js


def evaluation_user_group_means(ndcg_scores, arp_scores, poplift_scores, user_groups):
    group_means = {}
    for group_name, user_ids in user_groups.items():
        group_ndcg_scores = {
            user_id: ndcg_scores[user_id]
            for user_id in user_ids
            if user_id in ndcg_scores
        }
        group_arp_scores = {
            user_id: arp_scores[user_id]
            for user_id in user_ids
            if user_id in arp_scores
        }
        group_poplift_scores = {
            user_id: poplift_scores[user_id]
            for user_id in user_ids
            if user_id in poplift_scores
        }

        group_means[group_name] = {
            "ndcg_mean": sum(group_ndcg_scores.values()) / len(group_ndcg_scores),
            "arp_mean": sum(group_arp_scores.values()) / len(group_arp_scores),
            "poplift_mean": sum(group_poplift_scores.values())
            / len(group_poplift_scores),
        }

    return group_means
