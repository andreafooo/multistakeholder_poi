"""
One-time (per dataset) OSRM data preparation.

For each dataset this:
  1. Figures out which OSM extract(s) cover its POIs.
  2. Downloads the extract(s) from Geofabrik (merging multiple states via
     `osmium merge`, for a dataset whose POIs span more than one -- neither
     current dataset does: foursquaretky is a single Kanto extract, yelpphl
     is city-filtered to Philadelphia and resolves to the single Pennsylvania
     state extract).
  3. Runs osrm-extract / osrm-partition / osrm-customize (via the
     osrm/osrm-backend Docker image, MLD algorithm) once per routing profile
     (car, foot), producing routable .osrm files under osrm/data/<dataset>/<profile>/.

Requires: docker, `osmium` CLI (`sudo apt install osmium-tool` on
Debian/Ubuntu) only when merging more than one extract, and internet access
to download OSM extracts. Re-run is safe -- steps are skipped if their output
already exists; pass --force to redo a dataset from scratch.

Usage:
    python3 osrm/prepare_data.py --dataset foursquaretky
    python3 osrm/prepare_data.py --dataset yelpphl
    python3 osrm/prepare_data.py --dataset yelpphl --force
"""

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from civic_reranker import load_coordinates  # noqa: E402
from us_state_bboxes import geofabrik_url, states_for_coords  # noqa: E402

OSRM_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(OSRM_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROFILES = ["car", "foot"]
OSRM_IMAGE = "osrm/osrm-backend"

# Tokyo (foursquaretky) is fully contained in Geofabrik's "kanto" sub-region
# of Japan -- much smaller than the whole-Japan extract.
#
# Philadelphia (yelpphl) sits near the DE/NJ/PA tri-state border, so the
# generic bbox-based state detection below would otherwise pull in all three
# and require `osmium merge`. Pinned to the single Pennsylvania extract
# instead -- the handful of POIs just across the DE/NJ line fall back to
# haversine (no route found) rather than needing that extra dependency.
SINGLE_EXTRACT_DATASETS = {
    "foursquaretky": "https://download.geofabrik.de/asia/japan/kanto-latest.osm.pbf",
    "yelpphl": "https://download.geofabrik.de/north-america/us/pennsylvania-latest.osm.pbf",
}


def _run(cmd, **kwargs):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def download(url, dest):
    if os.path.exists(dest):
        print(f"Already downloaded: {dest}")
        return
    print(f"Downloading {url} -> {dest}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def resolve_region_pbf(dataset, force=False):
    """Returns the path to a single merged .osm.pbf covering `dataset`."""
    os.makedirs(RAW_DIR, exist_ok=True)
    merged_path = os.path.join(RAW_DIR, f"{dataset}-region.osm.pbf")
    if os.path.exists(merged_path) and not force:
        print(f"Region extract already prepared: {merged_path}")
        return merged_path

    if dataset in SINGLE_EXTRACT_DATASETS:
        raw_path = os.path.join(RAW_DIR, f"{dataset}-raw.osm.pbf")
        download(SINGLE_EXTRACT_DATASETS[dataset], raw_path)
        shutil.copyfile(raw_path, merged_path)
        return merged_path

    # Detect state(s) the dataset's POIs fall in, download each, merge if >1
    # (yelpphl is city-filtered to Philadelphia, so this resolves to a single
    # Pennsylvania extract -- no country-wide download).
    coords_df = load_coordinates(dataset)
    states = states_for_coords(coords_df)
    if not states:
        raise RuntimeError(f"No US states matched any coordinates for '{dataset}'")
    print(f"Dataset '{dataset}' spans states: {', '.join(states)}")

    state_paths = []
    for state in states:
        path = os.path.join(RAW_DIR, f"{state}-latest.osm.pbf")
        download(geofabrik_url(state), path)
        state_paths.append(path)

    if len(state_paths) == 1:
        shutil.copyfile(state_paths[0], merged_path)
        return merged_path

    if shutil.which("osmium") is None:
        raise RuntimeError(
            "Need the `osmium` CLI to merge multiple state extracts for "
            f"'{dataset}' (states: {', '.join(states)}). Install with: "
            "sudo apt install osmium-tool"
        )
    _run(["osmium", "merge", *state_paths, "-o", merged_path, "--overwrite"])
    return merged_path


def build_profile(dataset, profile, region_pbf, force=False):
    profile_dir = os.path.join(DATA_DIR, dataset, profile)
    os.makedirs(profile_dir, exist_ok=True)
    osrm_pbf = os.path.join(profile_dir, "region.osm.pbf")
    osrm_file = os.path.join(profile_dir, "region.osrm")

    if os.path.exists(osrm_file + ".partition") and not force:
        print(f"Already built: {dataset}/{profile}")
        return

    shutil.copyfile(region_pbf, osrm_pbf)
    mount = f"{profile_dir}:/data"
    profile_lua = f"/opt/{profile}.lua"

    _run(["docker", "run", "-t", "-v", mount, OSRM_IMAGE,
          "osrm-extract", "-p", profile_lua, "/data/region.osm.pbf"])
    _run(["docker", "run", "-t", "-v", mount, OSRM_IMAGE,
          "osrm-partition", "/data/region.osrm"])
    _run(["docker", "run", "-t", "-v", mount, OSRM_IMAGE,
          "osrm-customize", "/data/region.osrm"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["foursquaretky", "yelpphl"])
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=PROFILES,
                         help="Which profile(s) to build -- extract/partition/customize run "
                              "once per profile, so pass just one (e.g. --profiles foot) to "
                              "skip building a profile you don't need yet")
    parser.add_argument("--force", action="store_true", help="Rebuild even if output exists")
    args = parser.parse_args()

    region_pbf = resolve_region_pbf(args.dataset, force=args.force)
    for profile in args.profiles:
        build_profile(args.dataset, profile, region_pbf, force=args.force)

    services = " ".join(f"{args.dataset}-{p}" for p in args.profiles)
    print(f"\nDone. Start with: docker compose -f osrm/docker-compose.yml up -d {services}")


if __name__ == "__main__":
    main()
