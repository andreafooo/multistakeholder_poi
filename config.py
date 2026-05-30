import os
import sys
# from dotenv import load_dotenv


class BaseConfig:
    HOSTNAME = os.uname()[1]
    USER_DIR = os.path.expanduser("~")
    RECS_BASE_DIR = os.path.join(USER_DIR, "dev", "multistakeholder_poi", "datasets") # output directory from the old repo so eval can be done together
    SC_BASE_DIR = os.path.join(USER_DIR, "dev", "social-choice-engine")
    BASE = os.path.dirname(os.path.abspath(__file__))
    TIMESTAMP_FORMAT_DATE = "%Y-%m-%d"
    TIMESTAMP_FORMAT_LONG = "%Y-%m-%dT%H:%M"
    WEEKDAYS = set(range(0, 5))  # Monday to Friday
    PYTHON_PATH = sys.executable
    available_datasets = [
        "yelp", "foursquaretky"
    ]  # choose betweeen "yelp", "gowalla", "foursquaretky", and "brightkite" and make sure to add the datasets to your BASE_DIR


    top_k_resample = 150
    top_k_eval = 10
    valid_popularity = "item_pop"
    recommendation_dirpart = "recommendations"
    sc_models = ["BPR", "LORE", "LightGCN"]
    SC_OUTPUT_DIR = os.path.join(RECS_BASE_DIR, f"{available_datasets[0]}_dataset", recommendation_dirpart, f"{available_datasets[0]}_sample-socialchoice")
    methods_to_aggregate = ["baseline", "cp_min_js", "geo", "mmr"]


