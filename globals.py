available_datasets = [
    "foursquaretky",
]  # choose betweeen "yelp", "gowalla", "foursquaretky", and "brightkite" and make sure to add the datasets to your BASE_DIR

BASE_DIR = "./datasets/"


datasets_for_recbole = [
    "foursquaretky_sample"
]  # add datasets for recbole from above with "_sample" suffix - make sure to add them to recbole_general_recs/dataset

models_for_recbole = [
    "BPR"
]  # add general recommendation models as baseline (e.g. BPR, SimpleX, ItemKNN, etc.)

top_k_resample = 150
top_k_eval = 10
valid_popularity = "item_pop"
recommendation_dirpart = "recommendations"
