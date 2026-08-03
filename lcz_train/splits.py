"""T1 — deterministic, region-stratified, city-held-out splits.

The unit of the split is the CITY, never a block/patch/pixel — blocks within
one city are near-duplicate morphology, so anything finer leaks. So2Sat
cities present in the label AOIs always land in **test**, never train, so
So2Sat remains an untouched external check on everything trained here.

Stratification uses 8 regions (the spec's required 7, plus a West/Central
Asia bucket split out of "Asia" so Istanbul/Tehran aren't lumped with East or
South/SE Asia) so no split is region-blind even at small AOI counts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REGIONS = [
    "Europe", "North America", "Latin America", "Africa",
    "South/SE Asia", "East Asia", "West/Central Asia", "Oceania",
]

# Country -> region. Covers every country in src.utils.geo_lookup.CITY_TO_COUNTRY
# (the 51 So2Sat cities) plus the M4 pilot roster's non-So2Sat countries.
# Extend when labelling AOIs in a country not yet listed here — city_regions()
# raises rather than silently misclassifying an unmapped country.
COUNTRY_TO_REGION: dict[str, str] = {
    # Europe
    "Netherlands": "Europe", "Germany": "Europe", "Portugal": "Europe",
    "United Kingdom": "Europe", "Spain": "Europe", "Italy": "Europe",
    "Russia": "Europe", "France": "Europe", "Switzerland": "Europe",
    # North America
    "United States": "North America", "Canada": "North America",
    # Latin America
    "Colombia": "Latin America", "Argentina": "Latin America",
    "Venezuela": "Latin America", "Peru": "Latin America",
    "Brazil": "Latin America", "Chile": "Latin America",
    # Africa
    "Egypt": "Africa", "South Africa": "Africa", "Kenya": "Africa",
    "Ghana": "Africa",
    # East Asia
    "China": "East Asia", "Japan": "East Asia", "South Korea": "East Asia",
    # South/SE Asia
    "Bangladesh": "South/SE Asia", "India": "South/SE Asia",
    "Pakistan": "South/SE Asia", "Indonesia": "South/SE Asia",
    "Philippines": "South/SE Asia", "Thailand": "South/SE Asia",
    # West/Central Asia
    "Turkey": "West/Central Asia", "Iran": "West/Central Asia",
    # Oceania
    "Australia": "Oceania", "New Zealand": "Oceania",
}


def city_regions(cities: list[str], bounds_csv: str | Path) -> dict[str, str]:
    """Region for each city: So2Sat cities via geo_lookup, else via bounds_csv.

    ``bounds_csv`` is the global bounds table (``data/guppd_bounds.csv``),
    columns ``JRC_NAME_MAIN, CNTRY_NAME``. Raises on any city whose country
    isn't in :data:`COUNTRY_TO_REGION` — extend the table rather than guess.
    """
    from utils.geo_lookup import CITY_TO_COUNTRY

    df = pd.read_csv(bounds_csv)
    global_country = dict(zip(df["JRC_NAME_MAIN"], df["CNTRY_NAME"]))

    out = {}
    for c in cities:
        country = CITY_TO_COUNTRY.get(c) or global_country.get(c)
        if country is None:
            raise ValueError(f"no country found for city {c!r} in geo_lookup or {bounds_csv}")
        region = COUNTRY_TO_REGION.get(country)
        if region is None:
            raise ValueError(
                f"country {country!r} (city {c!r}) has no region — add it to "
                "COUNTRY_TO_REGION in lcz_train/splits.py"
            )
        out[c] = region
    return out


def make_splits(
    cities: list[str],
    so2sat_cities: set[str],
    *,
    seed: int,
    version: str,
    bounds_csv: str | Path,
    train_frac: float = 0.8,
) -> dict:
    """Deterministic city-held-out, region-stratified splits.

    So2Sat cities in ``cities`` always land in ``test``. The remainder is
    shuffled per region with a seeded RNG and split ``train_frac``/rest into
    train/val (a region with a single non-So2Sat city goes entirely to train
    — there's nothing to hold out yet; val fills in as more cities per region
    are labelled).
    """
    regions = city_regions(cities, bounds_csv)
    so2sat_present = sorted(c for c in cities if c in so2sat_cities)
    trainable = sorted(c for c in cities if c not in so2sat_cities)

    rng = np.random.default_rng(seed)
    train, val = [], []
    by_region: dict[str, list[str]] = {}
    for c in trainable:
        by_region.setdefault(regions[c], []).append(c)
    for region in sorted(by_region):
        members = by_region[region]
        order = rng.permutation(len(members))
        shuffled = [members[i] for i in order]
        if len(shuffled) < 2:
            train.extend(shuffled)
            continue
        n_train = max(1, round(len(shuffled) * train_frac))
        n_train = min(n_train, len(shuffled) - 1)  # keep >=1 for val
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train:])

    splits = {
        "version": version,
        "seed": seed,
        "train_frac": train_frac,
        "region_of": regions,
        "train": sorted(train),
        "val": sorted(val),
        "test": so2sat_present,
    }
    assert_no_leakage(splits, so2sat_cities)
    return splits


def assert_no_leakage(splits: dict, so2sat_cities: set[str] | None = None) -> None:
    """City-disjointness + So2Sat-in-test guard; raises AssertionError."""
    train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])
    overlap = (train & val) | (train & test) | (val & test)
    if overlap:
        raise AssertionError(f"cities in more than one split: {sorted(overlap)}")
    if so2sat_cities is not None:
        present = so2sat_cities & (train | val | test)
        stray = present - test
        if stray:
            raise AssertionError(f"So2Sat cities outside test: {sorted(stray)}")
    all_cities = set(splits["region_of"])
    missing = all_cities - (train | val | test)
    if missing:
        raise AssertionError(f"cities missing from every split: {sorted(missing)}")


def save_splits(splits: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def load_splits(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
