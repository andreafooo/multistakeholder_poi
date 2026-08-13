"""
One-time (per dataset+profile) precompute of the FULL item-item walking/car
distance matrix, so civic_reranker.py, evaluation_metrics.py's *_network
functions, and compute_user_compatibility.py never make a live OSRM call
again -- they all read the same on-disk cache (osrm/cache/<dataset>_<profile>.json)
that OSRMClient already uses, this script just fills it exhaustively up front
instead of lazily per-run.

Why this matters: civic_reranker.py's _prewarm does one /table call per user
over that user's candidate pool (e.g. 150 items at top_k_resample=150), and
OSRM table cost grows ~quadratically with point count. With ~1500 users each
needing a mostly-distinct 150-item table (candidates are personalized, so
little cross-user cache reuse), a full civic rerun took ~3 hours. The
catalog itself is fixed and far smaller (e.g. 2804 items for foursquaretky)
-- precomputing its full pairwise matrix ONCE, tiled via OSRM's /table
sources+destinations params (rectangular blocks, not the whole square),
means every later run -- any model, any top_k_resample, any stage that wants
real-distance GeoILD instead of haversine -- is a pure dict lookup.

Usage:
    python3 osrm/precompute_distance_matrix.py --dataset foursquaretky --profile foot
    python3 osrm/precompute_distance_matrix.py --dataset foursquaretky --profile foot --resume
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from civic_reranker import load_coordinates  # noqa: E402
from evaluation_metrics import haversine  # noqa: E402
from globals import OSRM_TABLE_MAX_COORDS  # noqa: E402
from osrm_client import OSRMClient  # noqa: E402

# Two blocks (sources + destinations) share one /table request, so each
# block must be at most half of osrm-routed's --max-table-size headroom.
BLOCK_SIZE = OSRM_TABLE_MAX_COORDS // 2


def _chunk(seq, size):
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--profile", default="foot", choices=["foot", "car"])
    parser.add_argument("--save-every", type=int, default=20, help="flush cache to disk every N block requests")
    args = parser.parse_args()

    coords_df = load_coordinates(args.dataset)
    items_with_coords = list(
        zip(coords_df["item_id:token"], coords_df["lat:float"], coords_df["lon:float"])
    )
    n = len(items_with_coords)
    blocks = _chunk(items_with_coords, BLOCK_SIZE)
    total_requests = len(blocks) * (len(blocks) + 1) // 2
    total_pairs = n * (n - 1) // 2
    print(
        f"{n} items, {len(blocks)} blocks of <= {BLOCK_SIZE}, "
        f"{total_requests} /table requests to cover {total_pairs} pairs"
    )

    client = OSRMClient(args.dataset, profile=args.profile)
    if client.unavailable:
        print(f"OSRM ({args.dataset}/{args.profile}) is not reachable -- start it first (see osrm/docker-compose.yml)")
        sys.exit(1)

    done = 0
    start = time.monotonic()
    for i in range(len(blocks)):
        for j in range(i, len(blocks)):
            client.table_block_km(blocks[i], blocks[j], haversine)
            done += 1
            if done % args.save_every == 0 or done == total_requests:
                client.save_cache()
                elapsed = time.monotonic() - start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total_requests - done) / rate if rate > 0 else float("inf")
                print(
                    f"  {done}/{total_requests} blocks "
                    f"({len(client._cache)} pairs cached, {elapsed:.0f}s elapsed, ETA {eta:.0f}s)"
                )

    client.save_cache()
    print(f"Done. {len(client._cache)} pairs cached at {client._cache_path}")


if __name__ == "__main__":
    main()
