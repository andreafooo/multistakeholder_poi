"""
Builds P-MMF's expected input for foursquaretky/BPR: a full user x item
preference-score matrix (not just each user's top-150 candidates, which is
all this repo's own pipeline exports -- P-MMF's dual-ascent loop needs to
rank the ENTIRE catalog per round) plus two `.simulation.inter` files that
differ only in what counts as a "provider":
  - foursquaretky_item:     each of the 2804 POIs is its own provider
                             (matches this repo's own framing of the
                             provider agent -- "diversity across local
                             businesses and POIs", see README)
  - foursquaretky_category: POIs grouped by venueCategory (132 categories
                             present in the sampled catalog) -- closer to
                             how P-MMF's own paper and most provider-fairness
                             literature frame "provider"

Trains a fresh BPR (reusing this repo's own recbole_general_recs config, so
it's the same model/hyperparameters as the rest of the pipeline) rather than
touching the already-trained checkpoint, since none is saved on disk here.
Deliberately does NOT call recbole_full_casestudy.run_configurations() --
that also writes a new timestamped entry under
datasets/foursquaretky_dataset/recommendations/, which would leak an extra
BPR run into every other script that scans that directory
(dataset_metadata()). This only trains + reads the checkpoint back out.

Usage:
    python3 build_pmmf_input.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

# recbole_general_recs/requirements.txt pins numpy==1.26.4, but this venv has
# numpy 2.x -- recbole 1.2.0's Config.compatibility_settings() does
# `np.float = np.float_` etc. at import time, and float_/complex_/unicode_
# were removed in numpy 2.0 (unlike bool_/int_/object_/str_, which survived).
# Same pattern as recbole_full_casestudy.py's dok_matrix/kmeans_pytorch
# patches: this is a latent bug in this environment, not specific to P-MMF --
# any recbole_general_recs training script hits it.
for _removed_alias, _replacement in [("float_", "float64"), ("complex_", "complex128"), ("unicode_", "str_")]:
    if not hasattr(np, _removed_alias):
        setattr(np, _removed_alias, getattr(np, _replacement))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "recbole_general_recs"))
from recbole_full_casestudy import find_newest_model  # noqa: E402 -- also patches kmeans_pytorch/dok_matrix imports
from recbole.quick_start import run_recbole, load_data_and_model  # noqa: E402
from recbole.utils.case_study import full_sort_scores  # noqa: E402

from globals import PROJECT_BASE, BASE_DIR  # noqa: E402

REPO = PROJECT_BASE
PMMF_DIR = os.path.join(REPO, "P-MMF")
BPR_CONFIG = os.path.join(REPO, "recbole_general_recs", "config", "foursquaretky_sample", "BPR", "config_test.yaml")


def _external_token_to_int(tokens):
    return np.array([int(t.split("_")[0]) for t in tokens])


def train_and_export_scores():
    """Trains BPR fresh, returns a (1500, 2804) score matrix ordered by
    external token int (row i = user "{i}_x", col j = item "{j}_x")."""
    data_path = os.path.join(REPO, "recbole_general_recs", "dataset") + os.sep
    run_recbole(config_file_list=[BPR_CONFIG], config_dict={"data_path": data_path})

    model_file = find_newest_model("saved/")
    print(f"Loading checkpoint: {model_file}")
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file)

    n_users, n_items = dataset.user_num, dataset.item_num  # includes the reserved [pad] id 0
    uid_series = np.arange(1, n_users)

    scores = full_sort_scores(uid_series, model, test_data, device=config["device"])
    scores = scores.cpu().numpy()[:, 1:]  # drop the [pad] item column

    user_tokens = dataset.id2token(dataset.uid_field, uid_series)
    item_tokens = dataset.id2token(dataset.iid_field, np.arange(1, n_items))
    user_order = np.argsort(_external_token_to_int(user_tokens))
    item_order = np.argsort(_external_token_to_int(item_tokens))

    ordered = scores[user_order][:, item_order]
    assert ordered.shape == (1500, 2804), ordered.shape
    # full_sort_scores masks the [pad] item and each user's own train/valid/test
    # history with -inf so the dual-ascent loop never "recommends" a visited POI
    n_masked = int(np.isneginf(ordered).sum())
    print(f"Score matrix {ordered.shape}, {n_masked} history entries masked to -inf")
    return ordered


def build_item_provider_pool():
    checkins = pd.read_csv(
        os.path.join(BASE_DIR, "foursquaretky_dataset", "processed_data_capri", "checkins.txt"),
        sep="\t", header=None, names=["user_id", "item_id", "timestamp"],
    )
    checkins["provider_item"] = checkins["item_id"]
    return checkins


def add_category_provider(checkins):
    mappings = json.load(open(os.path.join(BASE_DIR, "foursquaretky_dataset", "id_mappings.json")))
    item_tok_to_raw_venue = {int(tok.split("_")[0]): raw for tok, raw in mappings["business"].items()}

    meta = pd.read_csv(os.path.join(BASE_DIR, "foursquaretky_dataset", "foursquare_data.csv"))
    venue_to_cat = meta.drop_duplicates("venueId").set_index("venueId")["venueCategory"].to_dict()

    item_cat = checkins["item_id"].map(item_tok_to_raw_venue).map(venue_to_cat)
    missing = item_cat.isna().sum()
    if missing:
        raise ValueError(f"{missing} check-ins have no resolvable venueCategory -- check id_mappings.json coverage")
    checkins["provider_category"] = pd.Categorical(item_cat).codes
    return checkins


def write_simulation_inter(checkins, provider_col, dataset_name):
    out_dir = os.path.join(PMMF_DIR, "dataset", dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    out = checkins[["user_id", "item_id"]].copy()
    out["label"] = 1  # implicit feedback -- every check-in is a positive
    out["timestamp"] = checkins["timestamp"]
    out["provider"] = checkins[provider_col]
    out.columns = ["user_id:token", "item_id:token", "label:float", "timestamp:float", "provider:token"]
    out_path = os.path.join(out_dir, f"{dataset_name}.simulation.inter")
    out.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {len(out)} rows, {out['provider:token'].nunique()} providers -> {out_path}")


def main():
    scores = train_and_export_scores()

    tmp_dir = os.path.join(PMMF_DIR, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    for dataset_name in ["foursquaretky_item", "foursquaretky_category"]:
        np.save(os.path.join(tmp_dir, f"bpr_{dataset_name}_simulation.npy"), scores)

    checkins = build_item_provider_pool()
    checkins = add_category_provider(checkins)

    write_simulation_inter(checkins, "provider_item", "foursquaretky_item")
    write_simulation_inter(checkins, "provider_category", "foursquaretky_category")

    print("\nDone. Run from inside P-MMF/, e.g.:")
    print("  python3 oracle.py --Dataset foursquaretky_item")
    print("  python3 P-MMF.py --Dataset foursquaretky_item --gpu=false")
    print("  python3 oracle.py --Dataset foursquaretky_category")
    print("  python3 P-MMF.py --Dataset foursquaretky_category --gpu=false")


if __name__ == "__main__":
    main()
