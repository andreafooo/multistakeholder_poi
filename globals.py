import os

available_datasets = [
    "foursquaretky", "yelp"
]  # choose betweeen "yelp" and "foursquaretky", and make sure to add the datasets to your BASE_DIR

PROJECT_BASE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(PROJECT_BASE, "datasets") 

datasets_for_recbole = [
 "foursquaretky_sample", "yelp_sample"
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

# --- Platform / Calibration Re-Ranker ---
gridsearch = False  # set to True to run a fresh delta grid search for new datasets
use_saved_gridsearch_deltas = False  # set to True to load per-group deltas from a previous gridsearch run (cp/gridsearch_best_deltas.json)
CP_DELTA = 1  # fixed delta used for all users when neither gridsearch nor use_saved_gridsearch_deltas is set
save_upd = False

# --- Dynamic allocation (SCRUF-D mechanisms) ---
fairness_agents = ["cp_min_js", "geo", "mmr"]  # excludes "baseline": always-on, fixed-weight personalization agent
dynamic_window = 30          # sliding window size (# users) for each agent's fairness-so-far (mi); expands until full, no separate burn-in
dynamic_weight_floor = 0     # floor on raw weighting-source score per agent; 0 = off by default, sweepable (see design discussion)
dynamic_seed = 42            # fixed seed for shuffling user processing order (stream simulation) and lottery draws

# Per-variant on/off switches -- each is independently re-runnable without touching the others.
# "mi" = weighted/drawn from (1 - m_i) alone; "ci" = from c_i alone; "mi_ci" = from (1 - m_i) * c_i (SCRUF-D default).
run_static_sc = False        # equal-weight run (borda/schulze as-is, no fairness reweighting)
run_least_fair = True       # SCRUF-D "Least Fair": deterministic single lowest-m_i agent active each round
run_weighted_mi = True      # SCRUF-D "Weighted", source="mi"
run_weighted_ci = True      # SCRUF-D "Weighted", source="ci"
run_weighted_mi_ci = False    # SCRUF-D "Weighted", source="mi_ci"
run_lottery_mi = True       # SCRUF-D "Lottery", source="mi"
run_lottery_ci = True       # SCRUF-D "Lottery", source="ci"
run_lottery_mi_ci = True    # SCRUF-D "Lottery", source="mi_ci"

dynamic_methods = [
    "borda_leastfair", "borda_weighted_mi", "borda_weighted_ci", "borda_weighted_mi_ci",
    "borda_lottery_mi", "borda_lottery_ci", "borda_lottery_mi_ci",
    "schulze_leastfair", "schulze_weighted_mi", "schulze_weighted_ci", "schulze_weighted_mi_ci",
    "schulze_lottery_mi", "schulze_lottery_ci", "schulze_lottery_mi_ci",
]