"""
Verifies civic_reranker's OSRM integration degrades to haversine instead of
crashing or returning inf/nan, in the two situations that come up with a
per-dataset OSRM extract that only covers the states/regions a dataset's
POIs actually fall in:

  1. OSRM isn't reachable at all (not started yet, wrong port, ...).
  2. OSRM is reachable, but a specific pair has no path in the loaded graph
     -- e.g. a cross-metro pair for yelp (Philadelphia <-> Reno), where the
     merged extract has no connecting road data between the two regions.
     OSRM's /table endpoint returns `null` for that matrix cell rather than
     erroring the whole request.

No live OSRM server is required -- the OSRM-reachable case is simulated by
mocking requests.Session.get.
"""

import math
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from civic_reranker import GeoReranker
from osrm_client import OSRMClient

PHILADELPHIA = ("phila_x", 39.953949, -75.1432262)
RENO = ("reno_x", 39.4761165, -119.7893392)
INDIANAPOLIS = ("indy_x", 39.9062952, -86.0474634)


def _coords_df():
    ids, lats, lons = zip(PHILADELPHIA, RENO, INDIANAPOLIS)
    return pd.DataFrame({"item_id:token": ids, "lat:float": lats, "lon:float": lons})


def test_distance_falls_back_when_osrm_unreachable():
    """No OSRM server listening on the configured port at all."""
    reranker = GeoReranker(_coords_df(), min_distance_km=0.01, dataset="yelp", osrm_profile="car")

    expected = GeoReranker._haversine(PHILADELPHIA[1], PHILADELPHIA[2], RENO[1], RENO[2])
    dist = reranker._distance(PHILADELPHIA[0], RENO[0])

    assert math.isfinite(dist)
    assert dist == pytest.approx(expected)
    assert reranker.osrm_client.unavailable is True


def test_table_km_falls_back_on_null_cell_for_disconnected_pair():
    """OSRM is up, but the cross-metro pair has no route in the loaded graph
    -- /table returns `null` for that cell instead of erroring."""
    client = OSRMClient("yelp", profile="car")

    fake_response = MagicMock()
    fake_response.raise_for_status = lambda: None
    fake_response.json.return_value = {
        "code": "Ok",
        # Philadelphia<->Reno unreachable (null); both routes to/from
        # Indianapolis resolve fine.
        "distances": [
            [0, None, 1_200_000],
            [None, 0, 2_800_000],
            [1_200_000, 2_800_000, 0],
        ],
        "sources": [{"distance": 5.0}, {"distance": 12.0}, {"distance": 3.0}],
    }

    with patch.object(client.session, "get", return_value=fake_response):
        table = client.table_km(
            [PHILADELPHIA, RENO, INDIANAPOLIS], GeoReranker._haversine
        )

    expected_fallback = GeoReranker._haversine(PHILADELPHIA[1], PHILADELPHIA[2], RENO[1], RENO[2])
    assert table[(PHILADELPHIA[0], RENO[0])] == pytest.approx(expected_fallback)
    assert math.isfinite(table[(PHILADELPHIA[0], RENO[0])])
    # Reachable pairs use the OSRM-reported distance, not haversine.
    assert table[(PHILADELPHIA[0], INDIANAPOLIS[0])] == pytest.approx(1200.0)
    assert client.unavailable is False


def test_geo_select_does_not_crash_on_cross_metro_candidates():
    """End-to-end: a candidate set spanning disconnected metros should still
    produce a valid ranking, not raise or return non-finite distances."""
    reranker = GeoReranker(_coords_df(), min_distance_km=0.01, dataset="yelp", osrm_profile="car")
    candidates = [PHILADELPHIA[0], RENO[0], INDIANAPOLIS[0]]
    scores = {PHILADELPHIA[0]: 0.9, RENO[0]: 0.8, INDIANAPOLIS[0]: 0.7}

    selected = reranker._geo_select(candidates, scores, top_k=3)

    assert set(selected) == set(candidates)
