import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm

from platform_reranker import (
    create_base_recommendations,
    dataset_metadata,
    recommender_dir_combiner,
    save_top_k,
    open_ground_truth_user_group,
    calculate_user_popularity_distributions,
    rerank_upd,
)
from provider_reranker import open_training_data, build_item_similarity
from civic_reranker import load_coordinates
from compute_user_compatibility import percentile_rank_normalize
from evaluation_metrics import haversine, max_pairwise_haversine, jensen_shannon
from globals import (
    available_datasets,
    top_k_resample,
    top_k_eval,
    valid_popularity,
    MO_GREEDY_WEIGHTS,
)


class MultiObjectiveGreedyReranker:
    """
    Generalizes platform_reranker.py's CP algorithm (Steck-style calibrated
    greedy re-ranking -- rerank_for_user/marginal_relevances, 2 terms:
    relevance + JS calibration) to 4 terms: relevance, behavioral diversity
    gain (MMR-style, provider_reranker.py's item similarity matrix),
    geographic compactness cost (civic_reranker.py's coordinates), and
    popularity-tier calibration cost (platform_reranker.py's h/m/t tiers).
    One weighted-sum objective, one greedy loop over the raw baseline
    candidate pool -- no separate civic/platform/provider passes.

    Diversity and geo terms use the mean distance to ALL already-selected
    items (not just the nearest), matching the all-pairs-mean definition
    evaluation_metrics.py's behavioral_ild_per_user/geographic_ild_per_user
    already use to grade a list -- so the greedy objective directly targets
    what gets measured.

    `normalization` controls how the 4 raw per-candidate values are scaled
    before the weighted sum, at every greedy step:
      - "bounded" (default): each term's own fixed [0,1] bound -- min-max
        relevance, cosine-bounded diversity, geo cost / catalog-wide max
        distance, JS divergence's natural log2 bound.
      - "percentile_rank": ignores those fixed bounds and instead ranks the
        current step's remaining candidates against each other per term
        (compute_user_compatibility.percentile_rank_normalize -- same
        pandas .rank(pct=True) approach used there for per-user compatibility
        scores, just with "population" = this step's remaining candidates
        instead of "all users"). Outlier-robust and always uniformly spread
        across [0,1] regardless of the term's native scale, but discards
        magnitude -- a candidate barely better than the runner-up on some
        term looks identical to one that dominates it.
    """

    def __init__(self, item_sim, item_id_to_idx, item_coords, item_pop_group,
                 max_geo_km, weights=None, normalization="bounded"):
        self.item_sim = item_sim
        self.item_id_to_idx = item_id_to_idx
        self.item_coords = item_coords
        self.item_pop_group = item_pop_group  # {item_id: "h"/"m"/"t"}
        self.max_geo_km = max_geo_km
        self.weights = weights or MO_GREEDY_WEIGHTS
        if normalization not in ("bounded", "percentile_rank"):
            raise ValueError(f"Unknown normalization: {normalization!r}")
        self.normalization = normalization

    def _sim(self, item_a, item_b):
        idx_a = self.item_id_to_idx.get(item_a)
        idx_b = self.item_id_to_idx.get(item_b)
        if idx_a is None or idx_b is None:
            return 0.0
        return float(self.item_sim[idx_a, idx_b])

    def _geo_km(self, item_a, item_b):
        coords_a = self.item_coords.get(item_a)
        coords_b = self.item_coords.get(item_b)
        if coords_a is None or coords_b is None:
            return self.max_geo_km  # unknown location -> treat as maximally spread, not a free ride
        return haversine(coords_a[0], coords_a[1], coords_b[0], coords_b[1])

    def _select(self, candidates, relevance_scores, user_profile_ratios, top_k):
        w = self.weights
        selected = []
        remaining = set(candidates)
        tier_counts = {"h": 0, "m": 0, "t": 0}

        while remaining and len(selected) < top_k:
            relevance_raw, diversity_raw, geo_raw, calibration_raw = {}, {}, {}, {}

            for c in remaining:
                relevance_raw[c] = relevance_scores.get(c, 0.0)

                if selected:
                    diversity_raw[c] = float(np.mean([1.0 - self._sim(c, s) for s in selected]))
                    geo_raw[c] = float(np.mean([self._geo_km(c, s) for s in selected])) / self.max_geo_km
                else:
                    diversity_raw[c] = 0.0
                    geo_raw[c] = 0.0

                tier = self.item_pop_group.get(c)
                trial_counts = dict(tier_counts)
                if tier in trial_counts:
                    trial_counts[tier] += 1
                n = len(selected) + 1
                rec_ratios = {f"{g}_ratio": trial_counts[g] / n for g in ("h", "m", "t")}
                calibration_raw[c] = jensen_shannon(user_profile_ratios, rec_ratios)

            if self.normalization == "percentile_rank":
                relevance_term = percentile_rank_normalize(relevance_raw)
                diversity_term = percentile_rank_normalize(diversity_raw)
                geo_term = percentile_rank_normalize(geo_raw)
                calibration_term = percentile_rank_normalize(calibration_raw)
            else:
                relevance_term, diversity_term = relevance_raw, diversity_raw
                geo_term, calibration_term = geo_raw, calibration_raw

            best_item, best_score = None, float("-inf")
            for c in remaining:
                combined = (
                    w["relevance"] * relevance_term[c]
                    + w["diversity"] * diversity_term[c]
                    - w["geo"] * geo_term[c]
                    - w["calibration"] * calibration_term[c]
                )
                if combined > best_score:
                    best_score = combined
                    best_item = c

            selected.append(best_item)
            remaining.discard(best_item)
            tier = self.item_pop_group.get(best_item)
            if tier in tier_counts:
                tier_counts[tier] += 1

        return selected

    def rerank_all(self, recommendations_df, user_profiles_df, top_k):
        profiles = user_profiles_df.set_index("user_id:token")[["h_ratio", "m_ratio", "t_ratio"]]
        results = []

        for user_id, group in tqdm(
            recommendations_df.groupby("user_id:token"), desc="MO-greedy re-ranking users"
        ):
            if group.duplicated("item_id:token").any():
                group = group.drop_duplicates("item_id:token")

            candidates = group["item_id:token"].tolist()
            scores = dict(zip(group["item_id:token"], group["score"]))

            if user_id in profiles.index:
                user_profile_ratios = profiles.loc[user_id].to_dict()
            else:
                user_profile_ratios = {"h_ratio": 0.0, "m_ratio": 0.0, "t_ratio": 0.0}

            reranked = self._select(candidates, scores, user_profile_ratios, top_k)

            for rank, item_id in enumerate(reranked):
                results.append(
                    {
                        "user_id:token": user_id,
                        "item_id:token": item_id,
                        "rank": rank,
                        "score": scores[item_id],
                    }
                )

        out_df = pd.DataFrame(results)
        out_df = out_df.sort_values(["user_id:token", "rank"]).reset_index(drop=True)
        return out_df


def main(available_datasets):
    for dataset in tqdm(available_datasets, desc="Processing datasets"):
        data = dataset_metadata(dataset)

        train_data, test_data, user_groups = open_ground_truth_user_group(dataset)

        coords_df = load_coordinates(dataset)
        item_coords = dict(
            zip(coords_df["item_id:token"], zip(coords_df["lat:float"], coords_df["lon:float"]))
        )
        max_geo_km = max_pairwise_haversine(item_coords)

        interaction_df = open_training_data(dataset)
        item_sim, item_id_to_idx = build_item_similarity(interaction_df)

        for result in tqdm(data, desc=f"Processing models for {dataset}", leave=False):
            try:
                print(f"Processing model {result['model']} on dataset {result['dataset']}")

                baseline_topk_dir, basedir = recommender_dir_combiner(dataset, result["directory"])

                base_resample, _ = create_base_recommendations(
                    baseline_topk_dir, top_k_resample=top_k_resample, top_k_eval=top_k_eval
                )

                _, item_popularity, _ = rerank_upd(
                    dataset, base_resample, top_k_eval, valid_popularity,
                    dir_to_save=None, train_data=train_data,
                )
                item_pop_group = dict(zip(item_popularity["item_id:token"], item_popularity["item_pop_group"]))

                # Inner merge, same as platform_reranker.main(): candidates without a known
                # popularity tier are dropped, so the calibration term never sees a blank tier.
                base_resample = base_resample.merge(item_popularity, on="item_id:token")

                train_data_with_tiers = train_data.merge(item_popularity, on="item_id:token", how="left")
                user_profiles = calculate_user_popularity_distributions(train_data_with_tiers, item_popularity)

                for normalization, folder_name in [
                    # ("bounded", "mo_greedy"),
                    ("percentile_rank", "mo_greedy_pctrank"),
                ]:
                    reranker = MultiObjectiveGreedyReranker(
                        item_sim, item_id_to_idx, item_coords, item_pop_group, max_geo_km,
                        normalization=normalization,
                    )
                    out_df = reranker.rerank_all(base_resample, user_profiles, top_k=top_k_resample)
                    save_top_k(out_df, basedir, folder_name)

            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)


if __name__ == "__main__":
    main(available_datasets)
