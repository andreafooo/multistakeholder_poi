import os

# -----------------------------------------
# General Config
# -----------------------------------------
PROJECT_BASE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(PROJECT_BASE, "datasets") 
top_k_resample = int(os.environ.get("TOP_K_RESAMPLE", 50))  # env override lets sweep_top_k_resample.py
                                                               # vary this per subprocess without editing this file
top_k_eval = 10
valid_popularity = "item_pop"
recommendation_dirpart = "recommendations"
available_datasets = [
    "foursquaretky", 
    "yelpphl", 
]  # choose betweeen "yelp", "foursquaretky", "yelpphl", and "yelpno", and make sure to add the datasets to your BASE_DIR

# -----------------------------------------
# Data Sampling (preprocessing/data_sampling.py)
# -----------------------------------------
# Some dataset keys above are city-restricted variants of a raw dataset rather than raw
# datasets of their own -- they reuse the raw yelp_academic_dataset_*.json files from the
# source dataset's BASE_DIR folder (no need to duplicate the multi-GB raw files), but write
# their own sample into their own BASE_DIR/<dataset>_dataset folder, so the source dataset's
# existing sample and everything built on it (recommendations, RecBole datasets, logs) stays
# untouched.
raw_source_dataset = {"yelpphl": "yelp"}  # <output dataset key> -> <raw-data source dataset key>
city_filters = {"yelpphl": "Philadelphia"}  # restrict a dataset's POIs/check-ins to a single city

# Foursquare check-ins tagged with these venue categories are private residences, not real
# POIs -- excluded for both user privacy and recommendation quality.
foursquare_excluded_categories = ["Home (private)", 
                                  "Train Station", 
                                  "Subway", 
                                  "Bus Station",
                                  "Light Rail",
                                  "Airport",
                                  "Taxi", 
                                  "Road",
                                  "Residential Building/Apartment/Condo",
                                  "General Travel",
                                  "Travel & Transport"]





# -----------------------------------------
# RecBole Config for baseline creation
# -----------------------------------------
datasets_for_recbole = [
 "foursquaretky_sample", "yelpphl_sample"
]  # add datasets for recbole from above with "_sample" suffix - make sure to add them to recbole_general_recs/dataset
models_for_recbole = [
    "BPR", "LightGCN", "NeuMF"
]  # add general recommendation models as baseline (e.g. BPR, SimpleX, ItemKNN, etc.)

# -----------------------------------------
# OSRM (real travel distance for civic reranker)
# -----------------------------------------
OSRM_HOST = "localhost"
OSRM_PORTS = {
    "foursquaretky": {"car": 5000, "foot": 5001},
    "yelp": {"car": 5002, "foot": 5003},
}  # see osrm/docker-compose.yml -- run osrm/prepare_data.py once per dataset first
OSRM_DEFAULT_PROFILE = "foot"  # civic/local access is typically pedestrian-scale
OSRM_REQUEST_TIMEOUT = 5  # seconds -- for single-pair /route and /nearest calls
OSRM_TABLE_REQUEST_TIMEOUT = 30  # seconds -- /table cost grows ~quadratically with point count;
                                  # a 100-point chunk took ~4.7s against the foursquaretky/foot graph,
                                  # so this needs real headroom above OSRM_REQUEST_TIMEOUT
OSRM_MAX_SNAP_DISTANCE_M = 150  # beyond this, a point is treated as unroutable -> haversine fallback
OSRM_TABLE_MAX_COORDS = 200  # chunk /table requests above this many points per call -- must stay
                              # <= osrm-routed's --max-matrix-size (see osrm/docker-compose.yml);
                              # kept below that 250 cap so a full top_k_resample=150 group always
                              # fits in a single /table call (chunking still creates a cross-chunk
                              # gap resolved via slow per-pair fallback -- avoid needing it at all)
OSRM_CACHE_DIR = os.path.join(PROJECT_BASE, "osrm", "cache")

# -----------------------------------------
# Provider / MMR Re-Ranker
# -----------------------------------------
MMR_LAMBDA = 1  # 0 = pure relevance, 1 = pure diversity


# -----------------------------------------
# Platform / Calibration Re-Ranker
# -----------------------------------------
gridsearch = False  # set to True to run a fresh delta grid search for new datasets
use_saved_gridsearch_deltas = False  # set to True to load per-group deltas from a previous gridsearch run (cp/gridsearch_best_deltas.json)
CP_DELTA = 1  # fixed delta used for all users when neither gridsearch nor use_saved_gridsearch_deltas is set
save_upd = False


# -----------------------------------------
# Social Choice Aggregation
# -----------------------------------------
fairness_agents = ["platform", "civic", "provider"]  # excludes "baseline": always-on, fixed-weight personalization agent
sc_models = models_for_recbole
boosting = False # Naive boosting experiment from CIKM where 1 agents counts double
methods_to_aggregate = ["baseline", "platform", "civic", "provider"]
sc_methods = ["borda", "schulze"]

# full_eval_methods = ["baseline", "platform", "provider", "civic", "borda", "schulze", "rrf"]
full_eval_methods = ["baseline", "platform", "provider", "civic", "borda", "schulze", "rrf", "mo_greedy", "mo_greedy_pctrank"]
boosting_methods = ["borda2baseline", "borda2platform", "borda2civic", "borda2provider",
                    "schulze2baseline", "schulze2platform", "schulze2civic", "schulze2provider"]

# -----------------------------------------
# Alternatives to social choice (static, equal-weight -- no dynamic reweighting)
# -----------------------------------------
# Reciprocal Rank Fusion (rrf_aggregation.py, via the ranx package): fuses the 4
# already-reranked stakeholder lists purely by rank position, no cardinal scores needed.
rrf_k = 60  # RRF's k constant (Cormack, Clarke & Buttcher 2009 default)

# Multi-objective greedy re-ranker (multi_objective_reranker.py): one greedy loop over the
# raw baseline candidate pool, combining relevance + behavioral diversity + geo compactness +
# popularity-tier calibration into a single weighted-sum criterion per step. Equal weights by
# default -- a "true" unweighted combination of all 4 objectives, no single one boosted.
# Run twice per model, once per normalization scheme (see MultiObjectiveGreedyReranker
# docstring): "mo_greedy" (each term's own fixed [0,1] bound) and "mo_greedy_pctrank"
# (percentile-rank normalized across the remaining candidates at each step, same
# pandas .rank(pct=True) approach compute_user_compatibility.py uses across users).
MO_GREEDY_WEIGHTS = {"relevance": 1.0, "diversity": 1.0, "geo": 1.0, "calibration": 1.0}

# -----------------------------------------
# Dynamic allocation (SCRUF-D mechanisms)
# -----------------------------------------
dynamic_methods = [
    "borda_leastfair", "borda_weighted_mi", "borda_weighted_ci", "borda_weighted_mi_ci",
    "borda_lottery_mi", "borda_lottery_ci", "borda_lottery_mi_ci",
    "schulze_leastfair", "schulze_weighted_mi", "schulze_weighted_ci", "schulze_weighted_mi_ci",
    "schulze_lottery_mi", "schulze_lottery_ci", "schulze_lottery_mi_ci"
]
dynamic_window = 30          # sliding window size (# users) for each agent's fairness-so-far (mi); expands until full, no separate burn-in
dynamic_weight_floor = 0     # floor on raw weighting-source score per agent; 0 = off by default, sweepable (see design discussion)
dynamic_seed = 42            # fixed seed for shuffling user processing order (stream simulation) and lottery draws

# Per-variant on/off switches -- each is independently re-runnable without touching the others.
# "mi" = weighted/drawn from (1 - m_i) alone; "ci" = from c_i alone; "mi_ci" = from (1 - m_i) * c_i (SCRUF-D default).
run_static_sc = True        # equal-weight run (borda/schulze as-is, no fairness reweighting)
run_least_fair = False       # SCRUF-D "Least Fair": deterministic single lowest-m_i agent active each round
run_weighted_mi = False      # SCRUF-D "Weighted", source="mi"
run_weighted_ci = False      # SCRUF-D "Weighted", source="ci"
run_weighted_mi_ci = False    # SCRUF-D "Weighted", source="mi_ci"
run_lottery_mi = False       # SCRUF-D "Lottery", source="mi"
run_lottery_ci = False       # SCRUF-D "Lottery", source="ci"
run_lottery_mi_ci = False    # SCRUF-D "Lottery", source="mi_ci"
