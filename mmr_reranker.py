import pandas as pd
import os
from sklearn.preprocessing import normalize
import json
from tqdm import tqdm
import traceback
import numpy as np
from scipy.sparse import csr_matrix
from reranker import (
    create_base_recommendations,
    dataset_metadata,
    open_ground_truth_user_group,
    recommender_dir_combiner,
    save_top_k,
)
from globals import (
    BASE_DIR,
    available_datasets,
    recommendation_dirpart,
    top_k_eval,
    top_k_resample,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MMR_LAMBDA = 1  # 0 = pure relevance, 1 = pure diversity


# ---------------------------------------------------------------------------
# MMRReranker
# ---------------------------------------------------------------------------
class MMRReranker:
    def __init__(self, item_sim_matrix, item_id_to_idx, lambda_=0.5):
        self.item_sim = item_sim_matrix      # numpy array (n_items x n_items)
        self.item_id_to_idx = item_id_to_idx # dict: item_id str -> int index
        self.lambda_ = lambda_

    # --- core greedy MMR selection ---
    def _mmr_select(self, candidates, relevance_scores, top_k):
        """
        Greedy MMR selection.
        candidates       : list of item_id strings
        relevance_scores : dict {item_id: float}  (min-max normalised scores)
        top_k            : int
        """
        selected = []
        remaining = set(candidates)  # set for O(1) discard

        while remaining and len(selected) < top_k:
            if not selected:
                best = max(remaining, key=lambda i: relevance_scores[i])
            else:
                best = max(
                    remaining,
                    key=lambda i: (
                        (1 - self.lambda_) * relevance_scores[i]
                        - self.lambda_ * max(self._sim(i, j) for j in selected
                        )
                    ),
                )
            selected.append(best)
            remaining.discard(best)

        return selected

    def _sim(self, item_a, item_b):
        """Cosine similarity lookup between two item string IDs.
        Returns 0.0 for cold-start items not present in the training data."""
        idx_a = self.item_id_to_idx.get(item_a)
        idx_b = self.item_id_to_idx.get(item_b)
        if idx_a is None or idx_b is None:
            return 0.0
        return float(self.item_sim[idx_a, idx_b])

    # --- OFFLINE: full recommendation frame ---
    def rerank_all(self, recommendations_df, top_k):
        """
        Re-rank a full recommendations DataFrame and return a sorted DataFrame.

        Parameters
        ----------
        recommendations_df : pd.DataFrame
            Columns: user_id:token, item_id:token, score
            Scores should already be min-max normalised per user
            (create_base_recommendations in reranker.py handles this).
        top_k : int

        Returns
        -------
        pd.DataFrame  columns: user_id:token, item_id:token, rank, score
                      sorted by user_id:token, rank  (ready for save_top_k)
        """
        results = []

        for user_id, group in tqdm(
            recommendations_df.groupby("user_id:token"),
            desc="MMR re-ranking users",
        ):
            # Guard against duplicate items per user
            if group.duplicated("item_id:token").any():
                group = group.drop_duplicates("item_id:token")

            candidates = group["item_id:token"].tolist()
            scores = dict(zip(group["item_id:token"], group["score"]))

            reranked = self._mmr_select(candidates, scores, top_k)

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
        # Explicit sort so save_top_k JSON preserves MMR order
        out_df = out_df.sort_values(["user_id:token", "rank"]).reset_index(drop=True)
        return out_df

    # --- ONLINE: single user at inference time ---
    def rerank_user(self, user_candidates, relevance_scores, top_k):
        """
        Re-rank candidates for a single user on the fly.

        Parameters
        ----------
        user_candidates  : list of item_id strings
        relevance_scores : dict {item_id: float}
        top_k            : int

        Returns
        -------
        list of item_id strings in MMR order
        """
        return self._mmr_select(user_candidates, relevance_scores, top_k)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def open_training_data(dataset):
    """Load and concatenate train + validation interaction files."""
    train_data = pd.read_csv(
        os.path.join(
            BASE_DIR,
            f"{dataset}_dataset",
            "processed_data_recbole",
            f"{dataset}_sample.train.inter",
        ),
        sep="\t",
    )
    valid_data = pd.read_csv(
        os.path.join(
            BASE_DIR,
            f"{dataset}_dataset",
            "processed_data_recbole",
            f"{dataset}_sample.valid.inter",
        ),
        sep="\t",
    )
    print("Train + valid shapes:", train_data.shape, valid_data.shape)
    return pd.concat([train_data, valid_data], ignore_index=True)


def build_item_similarity(train_df):
    """
    Build an item-item cosine similarity matrix from interaction data.

    Strategy
    --------
    1. Log-normalise checkin counts to dampen power-user dominance.
    2. Build a sparse user x item interaction matrix.
    3. L2-normalise item vectors (columns).
    4. Cosine similarity = normalised_item_vectors @ normalised_item_vectors.T

    Parameters
    ----------
    train_df : pd.DataFrame  (not mutated)

    Returns
    -------
    item_sim      : np.ndarray  shape (n_items, n_items)
    item_id_to_idx: dict  {item_id_str: int}
    """
    df = train_df.copy()  # never mutate the caller's DataFrame

    df["weight"] = np.log1p(df["checkin_count:float"])

    # Build categorical codes once so mapping and matrix indices are guaranteed
    # to be consistent
    user_cat = df["user_id:token"].astype("category")
    item_cat = df["item_id:token"].astype("category")

    df["user_idx"] = user_cat.cat.codes
    df["item_idx"] = item_cat.cat.codes

    n_users = user_cat.cat.categories.size
    n_items = item_cat.cat.categories.size

    interaction_matrix = csr_matrix(
        (df["weight"].values, (df["user_idx"].values, df["item_idx"].values)),
        shape=(n_users, n_items),
    )
    print(f"Interaction matrix shape: {interaction_matrix.shape}")

    # Item vectors: transpose so shape is (n_items, n_users), then L2-normalise
    item_vectors = normalize(interaction_matrix.T, norm="l2")

    # Full cosine similarity matrix
    item_sim = np.dot(item_vectors, item_vectors.T).toarray()
    print(f"Item similarity matrix shape: {item_sim.shape}")

    # Mapping derived from the same categorical to guarantee alignment
    item_id_to_idx = {
        item_id: idx for idx, item_id in enumerate(item_cat.cat.categories)
    }

    return item_sim, item_id_to_idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(available_datasets):
    for dataset in tqdm(available_datasets, desc="Processing datasets"):
        data = dataset_metadata(dataset)

        # Build similarity matrix once per dataset, reuse across all models
        train_df = open_training_data(dataset)
        item_sim, item_id_to_idx = build_item_similarity(train_df)
        reranker = MMRReranker(item_sim, item_id_to_idx, lambda_=MMR_LAMBDA)

        for result in tqdm(
            data, desc=f"Processing models for {dataset}", leave=False
        ):
            try:
                print(
                    f"Processing model {result['model']} on dataset {result['dataset']}"
                )

                baseline_topk_dir, basedir = recommender_dir_combiner(
                    dataset, result["directory"]
                )

                base_resample, _ = create_base_recommendations(
                    baseline_topk_dir,
                    top_k_resample=top_k_resample,
                    top_k_eval=top_k_eval,
                )

                out_df = reranker.rerank_all(
                    recommendations_df=base_resample,
                    top_k=top_k_resample,
                )
                save_top_k(out_df, basedir, "mmr")

            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)


if __name__ == "__main__":
    main(available_datasets)