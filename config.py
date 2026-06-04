import os

class BaseConfig:
    HOSTNAME = os.uname()[1]
    USER_DIR = os.path.expanduser("~")
    RECS_BASE_DIR = os.path.join(USER_DIR, "dev", "multistakeholder_poi", "datasets")
    SC_BASE_DIR = os.path.join(USER_DIR, "dev", "social-choice-engine")
    BASE = os.path.dirname(os.path.abspath(__file__))
    available_datasets = [
        "yelp", "foursquaretky"
    ]  # choose betweeen "yelp" and "foursquaretky"

    top_k_resample = 150
    top_k_eval = 10
    recommendation_dirpart = "recommendations"
    sc_models = ["BPR"]
    methods_to_aggregate = ["baseline", "cp_min_js", "geo", "mmr"]
    boosting = True


