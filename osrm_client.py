"""
Thin client for a locally-hosted OSRM instance (see osrm/), used by
civic_reranker.py to replace haversine "as the crow flies" distance with
real travel distance.

Falls back to haversine (the function is passed in by the caller, so this
module has no dependency on civic_reranker's internals) whenever OSRM is
unreachable, a pair has no route, or a point doesn't snap to the road
network within OSRM_MAX_SNAP_DISTANCE_M -- so the reranker never hard-fails
just because OSRM isn't running.

Distances are cached to disk per (dataset, profile) under OSRM_CACHE_DIR,
since the same POI pairs recur heavily across users. Call save_cache()
after a batch of work to persist new entries.
"""

import json
import os
import time

import requests

from globals import (
    OSRM_CACHE_DIR,
    OSRM_HOST,
    OSRM_MAX_SNAP_DISTANCE_M,
    OSRM_PORTS,
    OSRM_REQUEST_TIMEOUT,
    OSRM_TABLE_MAX_COORDS,
    OSRM_TABLE_REQUEST_TIMEOUT,
)


def _chunk(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


class OSRMClient:
    def __init__(
        self,
        dataset,
        profile=None,
        host=OSRM_HOST,
        ports=OSRM_PORTS,
        timeout=OSRM_REQUEST_TIMEOUT,
        table_timeout=OSRM_TABLE_REQUEST_TIMEOUT,
        cache_dir=OSRM_CACHE_DIR,
        max_snap_distance_m=OSRM_MAX_SNAP_DISTANCE_M,
    ):
        from globals import OSRM_DEFAULT_PROFILE

        self.dataset = dataset
        self.profile = profile or OSRM_DEFAULT_PROFILE
        if dataset not in ports or self.profile not in ports[dataset]:
            raise ValueError(
                f"No OSRM port configured for dataset={dataset!r} profile={self.profile!r} "
                f"in globals.OSRM_PORTS. Run osrm/prepare_data.py first."
            )
        self.base_url = f"http://{host}:{ports[dataset][self.profile]}"
        self.timeout = timeout
        # /table cost grows ~quadratically with point count, so it needs a
        # much longer timeout than single-pair /route or /nearest calls.
        self.table_timeout = table_timeout
        self.max_snap_distance_m = max_snap_distance_m
        self.session = requests.Session()
        # Set on first connection failure; once true, skip network calls for
        # the rest of the process instead of retrying per-pair (which would
        # otherwise be extremely slow if OSRM is simply not running).
        self.unavailable = False

        self._cache_path = os.path.join(cache_dir, f"{dataset}_{self.profile}.json")
        self._cache = self._load_cache()
        self._dirty = False

    def _load_cache(self):
        if os.path.exists(self._cache_path):
            with open(self._cache_path) as f:
                return json.load(f)
        return {}

    def save_cache(self):
        if not self._dirty:
            return
        os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
        with open(self._cache_path, "w") as f:
            json.dump(self._cache, f)
        self._dirty = False

    @staticmethod
    def _pair_key(item_a, item_b):
        a, b = sorted((str(item_a), str(item_b)))
        return f"{a}|{b}"

    def _mark_unavailable(self):
        if not self.unavailable:
            print(
                f"[osrm_client] OSRM unreachable at {self.base_url} -- "
                "falling back to haversine for the rest of this run"
            )
        self.unavailable = True

    def wait_until_ready(self, timeout=120, poll_interval=2):
        """
        Polls OSRM until it responds or `timeout` seconds elapse. Call this
        once before a batch of work (e.g. at the top of civic_reranker.main())
        so a container that's still loading its graph into memory -- which
        can take a couple of minutes for a large region -- doesn't get
        mistaken for "not running" and permanently latch the haversine
        fallback via `unavailable`. Returns True if OSRM responded, False on
        timeout (in which case `unavailable` is left set to True).
        """
        url = f"{self.base_url}/nearest/v1/{self.profile}/0,0"
        start = time.monotonic()
        deadline = start + timeout
        attempt = 0
        last_error = None
        while time.monotonic() < deadline:
            attempt += 1
            try:
                self.session.get(url, timeout=self.timeout)
                self.unavailable = False
                if attempt > 1:
                    print(f"[osrm_client] OSRM at {self.base_url} ready after "
                          f"{time.monotonic() - start:.0f}s")
                return True
            except requests.exceptions.RequestException as e:
                last_error = e
                elapsed = time.monotonic() - start
                if attempt == 1 or elapsed // 15 > (elapsed - poll_interval) // 15:
                    print(f"[osrm_client] Waiting for OSRM at {self.base_url} to become ready "
                          f"({elapsed:.0f}s elapsed, last error: {e.__class__.__name__})...")
                time.sleep(poll_interval)
        print(f"[osrm_client] Gave up waiting for OSRM at {self.base_url} after "
              f"{timeout}s (last error: {last_error!r})")
        self._mark_unavailable()
        return False

    def distance_km(self, item_a, coords_a, item_b, coords_b, haversine_fn):
        """Pairwise distance with disk caching; used when a pair wasn't
        already warmed by table_km (e.g. the online single-user path)."""
        if item_a == item_b:
            return 0.0
        key = self._pair_key(item_a, item_b)
        if key in self._cache:
            cached = self._cache[key]
            return cached if cached is not None else haversine_fn(*coords_a, *coords_b)
        if self.unavailable:
            return haversine_fn(*coords_a, *coords_b)

        lat1, lon1 = coords_a
        lat2, lon2 = coords_b
        url = f"{self.base_url}/route/v1/{self.profile}/{lon1},{lat1};{lon2},{lat2}?overview=false"
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException:
            self._mark_unavailable()
            return haversine_fn(*coords_a, *coords_b)

        if data.get("code") != "Ok":
            self._cache[key] = None
            self._dirty = True
            return haversine_fn(*coords_a, *coords_b)

        dist_km = data["routes"][0]["distance"] / 1000.0
        self._cache[key] = dist_km
        self._dirty = True
        return dist_km

    def table_km(self, items_with_coords, haversine_fn):
        """
        items_with_coords: list of (item_id, lat, lon).
        Warms the cache and returns {(item_a, item_b): distance_km} for
        every pair. Falls back to haversine per-pair when a waypoint fails
        to snap within max_snap_distance_m, or the request fails outright.
        """
        results = {}
        if self.unavailable or len(items_with_coords) < 2:
            return self._haversine_matrix(items_with_coords, haversine_fn)

        for chunk in _chunk(items_with_coords, OSRM_TABLE_MAX_COORDS):
            results.update(self._table_chunk(chunk, haversine_fn))
        return results

    def _table_chunk(self, chunk, haversine_fn):
        # Skip the network call entirely if every pair in this chunk was
        # already resolved for a previous user -- POI candidate pools
        # overlap heavily across users, so this matters a lot over a batch run.
        cached = {}
        all_cached = True
        for i, (item_a, _, _) in enumerate(chunk):
            for j in range(i + 1, len(chunk)):
                item_b = chunk[j][0]
                key = self._pair_key(item_a, item_b)
                if key not in self._cache:
                    all_cached = False
                    break
                dist_km = self._cache[key]
                if dist_km is None:
                    lat_a, lon_a = chunk[i][1], chunk[i][2]
                    lat_b, lon_b = chunk[j][1], chunk[j][2]
                    dist_km = haversine_fn(lat_a, lon_a, lat_b, lon_b)
                cached[(item_a, item_b)] = dist_km
                cached[(item_b, item_a)] = dist_km
            if not all_cached:
                break
        if all_cached:
            return cached

        coords_param = ";".join(f"{lon},{lat}" for _, lat, lon in chunk)
        url = f"{self.base_url}/table/v1/{self.profile}/{coords_param}?annotations=distance"
        try:
            resp = self.session.get(url, timeout=self.table_timeout)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != "Ok":
                raise ValueError(data.get("code"))
        except requests.exceptions.RequestException:
            self._mark_unavailable()
            return self._haversine_matrix(chunk, haversine_fn)
        except ValueError:
            return self._haversine_matrix(chunk, haversine_fn)

        distances = data["distances"]
        sources = data.get("sources") or []
        well_snapped = (
            [(s.get("distance") or 0) <= self.max_snap_distance_m for s in sources]
            if sources
            else [True] * len(chunk)
        )

        results = {}
        for i, (item_a, lat_a, lon_a) in enumerate(chunk):
            for j in range(i + 1, len(chunk)):
                item_b, lat_b, lon_b = chunk[j]
                d = distances[i][j]
                if d is not None and well_snapped[i] and well_snapped[j]:
                    dist_km = d / 1000.0
                else:
                    dist_km = haversine_fn(lat_a, lon_a, lat_b, lon_b)
                self._cache[self._pair_key(item_a, item_b)] = dist_km
                self._dirty = True
                results[(item_a, item_b)] = dist_km
                results[(item_b, item_a)] = dist_km
        return results

    def table_block_km(self, sources, destinations, haversine_fn):
        """
        Rectangular block of the pairwise distance matrix between `sources`
        and `destinations` (each a list of (item_id, lat, lon)), via OSRM's
        /table sources+destinations params -- one request computes
        len(sources) x len(destinations) cells instead of the
        (len(sources)+len(destinations))^2 a flat table_km call over their
        union would do. Used by osrm/precompute_distance_matrix.py to tile a
        full catalog-wide distance matrix in the fewest requests. Skips the
        network call if every pair is already cached (resumable).
        len(sources) + len(destinations) must stay <= OSRM_TABLE_MAX_COORDS
        (== osrm-routed's --max-table-size, see osrm/docker-compose.yml).
        """
        all_cached = True
        cached = {}
        for item_a, _, _ in sources:
            for item_b, _, _ in destinations:
                if item_a == item_b:
                    continue
                key = self._pair_key(item_a, item_b)
                if key not in self._cache:
                    all_cached = False
                    break
                dist_km = self._cache[key]
                cached[(item_a, item_b)] = dist_km
                cached[(item_b, item_a)] = dist_km
            if not all_cached:
                break
        if all_cached:
            return cached
        if self.unavailable:
            return self._haversine_block(sources, destinations, haversine_fn)

        coords = sources + destinations
        n_sources = len(sources)
        coords_param = ";".join(f"{lon},{lat}" for _, lat, lon in coords)
        sources_param = ";".join(str(i) for i in range(n_sources))
        destinations_param = ";".join(str(i) for i in range(n_sources, len(coords)))
        url = (
            f"{self.base_url}/table/v1/{self.profile}/{coords_param}"
            f"?sources={sources_param}&destinations={destinations_param}&annotations=distance"
        )
        try:
            resp = self.session.get(url, timeout=self.table_timeout)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != "Ok":
                raise ValueError(data.get("code"))
        except requests.exceptions.RequestException:
            self._mark_unavailable()
            return self._haversine_block(sources, destinations, haversine_fn)
        except ValueError:
            return self._haversine_block(sources, destinations, haversine_fn)

        distances = data["distances"]
        src_meta = data.get("sources") or []
        dst_meta = data.get("destinations") or []
        src_snapped = (
            [(s.get("distance") or 0) <= self.max_snap_distance_m for s in src_meta]
            if src_meta
            else [True] * n_sources
        )
        dst_snapped = (
            [(d.get("distance") or 0) <= self.max_snap_distance_m for d in dst_meta]
            if dst_meta
            else [True] * len(destinations)
        )

        results = {}
        for i, (item_a, lat_a, lon_a) in enumerate(sources):
            for j, (item_b, lat_b, lon_b) in enumerate(destinations):
                if item_a == item_b:
                    continue
                d = distances[i][j]
                if d is not None and src_snapped[i] and dst_snapped[j]:
                    dist_km = d / 1000.0
                else:
                    dist_km = haversine_fn(lat_a, lon_a, lat_b, lon_b)
                self._cache[self._pair_key(item_a, item_b)] = dist_km
                self._dirty = True
                results[(item_a, item_b)] = dist_km
                results[(item_b, item_a)] = dist_km
        return results

    @staticmethod
    def _haversine_block(sources, destinations, haversine_fn):
        results = {}
        for item_a, lat_a, lon_a in sources:
            for item_b, lat_b, lon_b in destinations:
                if item_a == item_b:
                    continue
                d = haversine_fn(lat_a, lon_a, lat_b, lon_b)
                results[(item_a, item_b)] = d
                results[(item_b, item_a)] = d
        return results

    @staticmethod
    def _haversine_matrix(items_with_coords, haversine_fn):
        results = {}
        for i, (item_a, lat_a, lon_a) in enumerate(items_with_coords):
            for j in range(i + 1, len(items_with_coords)):
                item_b, lat_b, lon_b = items_with_coords[j]
                d = haversine_fn(lat_a, lon_a, lat_b, lon_b)
                results[(item_a, item_b)] = d
                results[(item_b, item_a)] = d
        return results

    def nearest(self, lat, lon):
        """Snap distance in meters to the routing network, or None on failure."""
        if self.unavailable:
            return None
        url = f"{self.base_url}/nearest/v1/{self.profile}/{lon},{lat}"
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException:
            self._mark_unavailable()
            return None
        if data.get("code") != "Ok":
            return None
        return data["waypoints"][0]["distance"]
