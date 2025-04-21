import os

# Paths
dataDirectory = os.path.abspath("./Data/")
outputsDir = os.path.abspath("./Outputs/")

# Framework Parameter
isInteractive = True  # Should be false when the Terminal is not available

# Default Parameters
topK = 10  # Top-k items to evaluate (default: 10)
limitUsers = -1  # Limit the number of users (default: -1)
listLimit = 150  # Limit the length of recommendation list (default: 10)
activeUsersPercentage = [5, 20]  # Calculate [n] percents of users as active

# Default Parameters (non-interactive)
defaultModel = "GeoSoCa"  # Can be "GeoSoCa", "LORE", or "USG"
defaultDataset = "Gowalla"  # Can be "Gowalla", "Yelp", or "Foursquare"
defaultFusion = "Product"  # Can be "Product" or "Sum"
defaultEvaluation = [
    "Precision"
]  # Can be a set of "Precision", "Recall", "mAP", or "NDCG"

# Key: Model name, Value: Covered Contexts
models = {
    "GeoSoCa": ["Geographical", "Social", "Categorical"],
    "LORE": ["Geographical", "Social", "Temporal"],
    "USG": ["Interaction", "Social", "Geographical"],
}

# Key: Dataset name, Value: Covered Contexts
datasets = {
    "Gowalla": ["Geographical", "Social", "Temporal", "Interaction"],
    "Yelp": ["Geographical", "Social", "Temporal", "Categorical", "Interaction"],
    "Foursquare": ["Geographical", "Social", "Temporal", "Interaction"],
    "gowalla_sample": ["Geographical", "Temporal", "Interaction"],
    "brightkite_sample": ["Geographical", "Temporal", "Interaction"],
    "yelp_sample": ["Geographical", "Temporal", "Interaction"],
    "foursquaretky_sample": ["Geographical", "Temporal", "Interaction"],
}

# An array of selected operations
# TODO: "WeightedSum" is not implemented yet
fusions = ["Product", "Sum"]

# List of evaluation metrics
evaluationMetrics = ["Precision", "Recall", "mAP", "NDCG"]

# Models Dictionaries
GeoSoCaDict = {"alpha": 0.5}
LoreDict = {"alpha": 0.05, "deltaT": 3600 * 24}
USGDict = {"alpha": 0, "beta": 0.2, "eta": 0.05}
