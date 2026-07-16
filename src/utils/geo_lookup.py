"""Static geographic lookups for the 51 So2Sat-LCZ42 cities.

City names match the ``JRC_NAME_MAIN`` column of
``data/so2sat_guppd_bounds.csv`` exactly (including the bracketed alternates and
the CJK Dongying entry). ``assign_city`` reproduces the nearest-centroid city
assignment used in ``notebooks/embedding_visualization.ipynb`` so the script and
notebook share one implementation.

Transcontinental / ambiguous cases resolved conventionally: Istanbul→Turkey/Asia
(Turkey is mostly Asian), Moscow→Russia/Europe (European Russia), Hong Kong→China,
San Jose→United States (So2Sat's San Jose is in California).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── City → country ────────────────────────────────────────────────────────────
CITY_TO_COUNTRY: dict[str, str] = {
    "Amsterdam": "Netherlands",
    "Beijing": "China",
    "Berlin": "Germany",
    "Bogota": "Colombia",
    "Buenos Aires": "Argentina",
    "Cairo": "Egypt",
    "Cape Town": "South Africa",
    "Caracas": "Venezuela",
    "Changsha": "China",
    "Chicago": "United States",
    "Cologne": "Germany",
    "Dhaka": "Bangladesh",
    "Guangzhou": "China",
    "Hong Kong": "China",
    "Istanbul": "Turkey",
    "Jakarta": "Indonesia",
    "Karachi": "Pakistan",
    "Lima": "Peru",
    "Lisbon": "Portugal",
    "London": "United Kingdom",
    "Los Angeles": "United States",
    "Madrid": "Spain",
    "Melbourne": "Australia",
    "Milan": "Italy",
    "Moscow": "Russia",
    "Mumbai": "India",
    "Munich": "Germany",
    "Nairobi": "Kenya",
    "Nanjing": "China",
    "New York": "United States",
    "Osaka [Kyoto]": "Japan",
    "Paris": "France",
    "Philadelphia": "United States",
    "Qingdao": "China",
    "Quezon City [Manila]": "Philippines",
    "Rawalpindi [Islamabad]": "Pakistan",
    "Rio De Janeiro": "Brazil",
    "Rome": "Italy",
    "Salvador": "Brazil",
    "San Jose": "United States",
    "Santiago": "Chile",
    "Shanghai": "China",
    "Sydney": "Australia",
    "São Paulo": "Brazil",
    "Tehran": "Iran",
    "Tokyo": "Japan",
    "Vancouver": "Canada",
    "Washington D.C.": "United States",
    "Wuhan": "China",
    "Zurich": "Switzerland",
    "东营区": "China",
}

# ── City → continent ──────────────────────────────────────────────────────────
CITY_TO_CONTINENT: dict[str, str] = {
    "Amsterdam": "Europe",
    "Beijing": "Asia",
    "Berlin": "Europe",
    "Bogota": "South America",
    "Buenos Aires": "South America",
    "Cairo": "Africa",
    "Cape Town": "Africa",
    "Caracas": "South America",
    "Changsha": "Asia",
    "Chicago": "North America",
    "Cologne": "Europe",
    "Dhaka": "Asia",
    "Guangzhou": "Asia",
    "Hong Kong": "Asia",
    "Istanbul": "Asia",
    "Jakarta": "Asia",
    "Karachi": "Asia",
    "Lima": "South America",
    "Lisbon": "Europe",
    "London": "Europe",
    "Los Angeles": "North America",
    "Madrid": "Europe",
    "Melbourne": "Oceania",
    "Milan": "Europe",
    "Moscow": "Europe",
    "Mumbai": "Asia",
    "Munich": "Europe",
    "Nairobi": "Africa",
    "Nanjing": "Asia",
    "New York": "North America",
    "Osaka [Kyoto]": "Asia",
    "Paris": "Europe",
    "Philadelphia": "North America",
    "Qingdao": "Asia",
    "Quezon City [Manila]": "Asia",
    "Rawalpindi [Islamabad]": "Asia",
    "Rio De Janeiro": "South America",
    "Rome": "Europe",
    "Salvador": "South America",
    "San Jose": "North America",
    "Santiago": "South America",
    "Shanghai": "Asia",
    "Sydney": "Oceania",
    "São Paulo": "South America",
    "Tehran": "Asia",
    "Tokyo": "Asia",
    "Vancouver": "North America",
    "Washington D.C.": "North America",
    "Wuhan": "Asia",
    "Zurich": "Europe",
    "东营区": "Asia",
}


def load_city_centers(bounds_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (city_names, centers) from the GUPPD bounds CSV.

    centers is an (N, 2) array of (lon, lat) bounding-box centroids.
    """
    df = pd.read_csv(bounds_csv)
    names = df["JRC_NAME_MAIN"].to_numpy()
    centers = np.column_stack([
        (df["minx"] + df["maxx"]) / 2.0,
        (df["miny"] + df["maxy"]) / 2.0,
    ])
    return names, centers


def assign_city(coords: np.ndarray, bounds_csv: Path) -> np.ndarray:
    """Assign each (lon, lat) coordinate to its nearest GUPPD city centroid.

    coords: (N, 2) array of (lon, lat). Returns an (N,) array of city-name
    strings. Always matches (nearest centroid), mirroring the notebook.
    """
    from sklearn.neighbors import NearestNeighbors

    names, centers = load_city_centers(bounds_csv)
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
    nn.fit(centers)
    _, idx = nn.kneighbors(coords)
    return names[idx.flatten()]
