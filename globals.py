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
full_eval_methods = ["baseline", "cp_min_js", "mmr", "geo", "borda", "schulze"]
boosting_methods = ["borda2baseline", "borda2cp_min_js", "borda2geo", "borda2mmr", 
                    "schulze2baseline", "schulze2cp_min_js", "schulze2geo", "schulze2mmr"]
MMR_LAMBDA = 1  # 0 = pure relevance, 1 = pure diversity

