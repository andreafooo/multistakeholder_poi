import os

available_datasets = [
    "foursquaretky", "yelp"
]  # choose betweeen "yelp" and "foursquaretky", and make sure to add the datasets to your BASE_DIR

PROJECT_BASE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(PROJECT_BASE, "datasets") 

datasets_for_recbole = [
    "yelp_sample", "foursquaretky_sample"
]  # add datasets for recbole from above with "_sample" suffix - make sure to add them to recbole_general_recs/dataset

models_for_recbole = [
    "BPR"
]  # add general recommendation models as baseline (e.g. BPR, SimpleX, ItemKNN, etc.)
top_k_resample = 150
top_k_eval = 10
valid_popularity = "item_pop"
recommendation_dirpart = "recommendations"
methods_to_aggregate = ["baseline", "cp_min_js", "geo", "mmr"]
full_eval_methods = ["baseline", "cp_min_js", "mmr", "geo", "borda", "schulze"]
boosting_methods = ["borda2baseline", "borda2cp_min_js", "borda2geo", "borda2mmr", 
                    "schulze2baseline", "schulze2cp_min_js", "schulze2geo", "schulze2mmr"]

MMR_LAMBDA = 1  # 0 = pure relevance, 1 = pure diversity
sc_models = models_for_recbole
boosting = False
run_static_sc = False  # whether to (re-)run the static equal-weight/boosted social choice methods at all

# --- Dynamic allocation (SCRUF-D "Weighted" mechanism) ---
dynamic_allocation = True
fairness_agents = ["cp_min_js", "geo", "mmr"]  # excludes "baseline": always-on, fixed-weight personalization agent
dynamic_window = 30          # sliding window size (# users) for each agent's fairness-so-far (mi); expands until full, no separate burn-in
dynamic_weight_floor = 0     # floor on raw (1-mi)*ci per agent; 0 = off by default, sweepable (see design discussion)
use_ci = True                # compatibility term c_i: per-user nDCG-style agreement between an agent's own re-ranking and baseline (proxy for user profile); 1.0 (neutral) for every agent when False
dynamic_seed = 42            # fixed seed for shuffling user processing order (stream simulation) and lottery draws
run_least_fair = False        # SCRUF-D "Least Fair" mechanism: single lowest-m_i agent active each round
run_lottery = False         # SCRUF-D "Lottery" mechanism: single agent drawn ~ (1-m_i) each round
_ci_suffix = "_ci" if use_ci else ""
dynamic_methods = [
    f"borda_weighted{_ci_suffix}", f"schulze_weighted{_ci_suffix}",
    "borda_leastfair", "schulze_leastfair",  # leastfair ignores compatibility, never gets "_ci"
    f"borda_lottery{_ci_suffix}", f"schulze_lottery{_ci_suffix}",
]
