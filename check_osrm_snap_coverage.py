"""
Diagnostic: for every POI in a dataset, checks how well it snaps to the OSRM
road network (per profile), before wiring OSRM into the reranker for real.

Run after `osrm/prepare_data.py --dataset <name>` and starting the matching
docker-compose services. Writes a JSON report to osrm/reports/ and prints a
summary of the snap-failure / far-snap rate, so you know up front how often
_distance() will need to fall back to haversine.

Usage:
    python3 check_osrm_snap_coverage.py --dataset foursquaretky
    python3 check_osrm_snap_coverage.py --dataset yelp --profile car
"""

import argparse
import json
import os

from tqdm import tqdm

from civic_reranker import load_coordinates
from globals import OSRM_MAX_SNAP_DISTANCE_M, PROJECT_BASE
from osrm_client import OSRMClient

REPORTS_DIR = os.path.join(PROJECT_BASE, "osrm", "reports")


def check_dataset(dataset, profile):
    coords_df = load_coordinates(dataset)
    client = OSRMClient(dataset, profile=profile)
    client.wait_until_ready()

    flagged = []
    snap_distances = []
    failures = 0

    for item_id, lat, lon in tqdm(
        list(zip(coords_df["item_id:token"], coords_df["lat:float"], coords_df["lon:float"])),
        desc=f"Checking {dataset}/{profile} snap coverage",
    ):
        dist = client.nearest(lat, lon)
        if dist is None:
            failures += 1
            flagged.append({"item_id": item_id, "lat": lat, "lon": lon, "status": "failed"})
            continue
        snap_distances.append(dist)
        if dist > OSRM_MAX_SNAP_DISTANCE_M:
            flagged.append(
                {"item_id": item_id, "lat": lat, "lon": lon, "status": "far", "snap_distance_m": dist}
            )

    total = len(coords_df)
    far = sum(1 for f in flagged if f["status"] == "far")
    summary = {
        "dataset": dataset,
        "profile": profile,
        "total_pois": total,
        "request_failures": failures,
        "far_snaps": far,
        "far_snap_threshold_m": OSRM_MAX_SNAP_DISTANCE_M,
        "ok_pois": total - failures - far,
        "ok_rate": round((total - failures - far) / total, 4) if total else None,
        "mean_snap_distance_m": round(sum(snap_distances) / len(snap_distances), 2)
        if snap_distances
        else None,
        "max_snap_distance_m": round(max(snap_distances), 2) if snap_distances else None,
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, f"{dataset}_{profile}_snap_coverage.json")
    with open(report_path, "w") as f:
        json.dump({"summary": summary, "flagged": flagged}, f, indent=2)

    print(f"\n{dataset}/{profile}: {summary['ok_pois']}/{total} POIs OK "
          f"({summary['ok_rate'] * 100 if summary['ok_rate'] is not None else 0:.1f}%), "
          f"{failures} failed to snap, {far} snapped >{OSRM_MAX_SNAP_DISTANCE_M}m away")
    print(f"Report written to {report_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["foursquaretky", "yelp"])
    parser.add_argument("--profile", choices=["car", "foot"], default=None,
                         help="Defaults to checking both car and foot")
    args = parser.parse_args()

    profiles = [args.profile] if args.profile else ["car", "foot"]
    for profile in profiles:
        check_dataset(args.dataset, profile)


if __name__ == "__main__":
    main()
