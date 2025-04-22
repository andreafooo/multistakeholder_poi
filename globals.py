# BASE_DIR = "./datasets/"
available_datasets = [
    "foursquaretky",
    "yelp",
    "gowalla",
    "brightkite",
]  # choose betweeen "yelp", "gowalla", "foursquaretky", and "brightkite" and make sure to add the datasets to your BASE_DIR


BASE_DIR = "/Volumes/Forster Neu/Masterarbeit Data/"


datasets_for_recbole = [
    "foursquaretky_sample"
]  # add datasets for recbole from above with "_sample" suffix - make sure to add them to recbole_general_recs/dataset

models_for_recbole = [
    "BPR"
]  # add general recommendation models as baseline (e.g. BPR, SimpleX, ItemKnn, etc.)
