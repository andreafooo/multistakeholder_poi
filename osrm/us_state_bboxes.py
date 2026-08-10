"""
Approximate bounding boxes for US states/territories, used only to figure out
which Geofabrik state extracts a set of POI coordinates falls into (e.g. for
the yelp dataset, which spans several non-contiguous metros). Bounding boxes
are coarse rectangles (not real polygons), so a point can match more than one
state near a border -- that's fine here since we only use this to decide
which .osm.pbf files to download and merge, and downloading one extra
neighboring state is cheap compared to missing one entirely.

Each entry maps a state name to its Geofabrik "north-america/us/<slug>"
download slug and (min_lat, max_lat, min_lon, max_lon).
"""

STATE_BBOXES = {
    "alabama": ("alabama", 30.1, 35.1, -88.6, -84.7),
    "alaska": ("alaska", 51.0, 71.6, -179.9, -129.9),
    "arizona": ("arizona", 31.2, 37.1, -114.9, -108.9),
    "arkansas": ("arkansas", 32.9, 36.6, -94.7, -89.5),
    "california": ("california", 32.4, 42.1, -124.5, -114.0),
    "colorado": ("colorado", 36.9, 41.1, -109.2, -101.9),
    "connecticut": ("connecticut", 40.9, 42.1, -73.8, -71.7),
    "delaware": ("delaware", 38.4, 39.9, -75.9, -74.9),
    "florida": ("florida", 24.4, 31.1, -87.7, -79.9),
    "georgia": ("georgia", 30.3, 35.1, -85.7, -80.7),
    "hawaii": ("hawaii", 18.8, 22.4, -160.6, -154.7),
    "idaho": ("idaho", 41.9, 49.1, -117.3, -110.9),
    "illinois": ("illinois", 36.9, 42.6, -91.6, -87.0),
    "indiana": ("indiana", 37.7, 41.8, -88.1, -84.7),
    "iowa": ("iowa", 40.3, 43.6, -96.7, -90.0),
    "kansas": ("kansas", 36.9, 40.1, -102.1, -94.5),
    "kentucky": ("kentucky", 36.4, 39.2, -89.6, -81.9),
    "louisiana": ("louisiana", 28.8, 33.1, -94.1, -88.7),
    "maine": ("maine", 42.9, 47.5, -71.2, -66.8),
    "maryland": ("maryland", 37.8, 39.8, -79.5, -74.9),
    "massachusetts": ("massachusetts", 41.1, 43.0, -73.6, -69.8),
    "michigan": ("michigan", 41.6, 48.4, -90.5, -82.1),
    "minnesota": ("minnesota", 43.4, 49.4, -97.3, -89.4),
    "mississippi": ("mississippi", 30.1, 35.1, -91.7, -88.0),
    "missouri": ("missouri", 35.9, 40.7, -95.9, -89.0),
    "montana": ("montana", 44.3, 49.1, -116.2, -104.0),
    "nebraska": ("nebraska", 39.9, 43.1, -104.2, -95.2),
    "nevada": ("nevada", 34.9, 42.1, -120.1, -113.9),
    "new-hampshire": ("new-hampshire", 42.6, 45.4, -72.6, -70.6),
    "new-jersey": ("new-jersey", 38.7, 41.4, -75.7, -73.8),
    "new-mexico": ("new-mexico", 31.2, 37.1, -109.2, -102.9),
    "new-york": ("new-york", 40.4, 45.1, -79.9, -71.7),
    "north-carolina": ("north-carolina", 33.7, 36.7, -84.5, -75.3),
    "north-dakota": ("north-dakota", 45.9, 49.1, -104.2, -96.4),
    "ohio": ("ohio", 38.3, 42.0, -84.9, -80.4),
    "oklahoma": ("oklahoma", 33.5, 37.1, -103.1, -94.3),
    "oregon": ("oregon", 41.9, 46.4, -124.7, -116.4),
    "pennsylvania": ("pennsylvania", 39.6, 42.4, -80.6, -74.6),
    "rhode-island": ("rhode-island", 41.0, 42.1, -71.9, -71.0),
    "south-carolina": ("south-carolina", 32.0, 35.3, -83.5, -78.4),
    "south-dakota": ("south-dakota", 42.4, 46.1, -104.2, -96.3),
    "tennessee": ("tennessee", 34.9, 36.8, -90.4, -81.6),
    "texas": ("texas", 25.7, 36.6, -106.7, -93.4),
    "utah": ("utah", 36.9, 42.1, -114.2, -108.9),
    "vermont": ("vermont", 42.6, 45.1, -73.5, -71.4),
    "virginia": ("virginia", 36.5, 39.6, -83.7, -75.1),
    "washington": ("washington", 45.4, 49.1, -124.9, -116.9),
    "west-virginia": ("west-virginia", 37.1, 40.7, -82.7, -77.6),
    "wisconsin": ("wisconsin", 42.4, 47.1, -92.9, -86.6),
    "wyoming": ("wyoming", 40.9, 45.1, -111.2, -104.0),
    "district-of-columbia": ("district-of-columbia", 38.79, 39.0, -77.13, -76.9),
}

GEOFABRIK_US_BASE = "https://download.geofabrik.de/north-america/us"


def states_for_coords(coords_df, lat_col="lat:float", lon_col="lon:float"):
    """
    Given a DataFrame of POI coordinates, return the sorted list of Geofabrik
    US state slugs whose bounding box contains at least one point.
    """
    needed = set()
    for lat, lon in zip(coords_df[lat_col], coords_df[lon_col]):
        for slug, (_, min_lat, max_lat, min_lon, max_lon) in STATE_BBOXES.items():
            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                needed.add(slug)
    return sorted(needed)


def geofabrik_url(state_slug):
    return f"{GEOFABRIK_US_BASE}/{state_slug}-latest.osm.pbf"
