"""Helpers for mapping geographic bounding boxes to MGRS 100km square codes."""

from __future__ import annotations

from pathlib import Path

import mgrs
import pandas as pd


_M = mgrs.MGRS()

SO2SAT_BOUNDS_CSV = Path(__file__).parents[2] / "data" / "so2sat_guppd_bounds.csv"


def bbox_to_mgrs_codes(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    step_deg: float = 0.05,
) -> list[str]:
    """Return all MGRS 100km square codes intersecting a WGS84 bbox.

    Samples the bbox on a grid with ``step_deg`` spacing and converts each
    point to its 5-character MGRS code (e.g. ``"30UYC"``).

    The default 0.05° (~5.5 km) is chosen to handle the narrowest possible
    MGRS squares: at UTM zone boundaries the edge column can be as thin as
    ~0.1° wide, so 0.4° or even 0.1° steps can miss them.

    Args:
        minx, miny, maxx, maxy: Bounding box in EPSG:4326 (lon/lat degrees).
        step_deg: Grid sampling step in degrees.

    Returns:
        Sorted list of unique MGRS 100km square codes.
    """
    lons = _inclusive_range(minx, maxx, step_deg)
    lats = _inclusive_range(miny, maxy, step_deg)

    codes: set[str] = set()
    for lat in lats:
        for lon in lons:
            try:
                code = _M.toMGRS(lat, lon, MGRSPrecision=0)
                if code:
                    codes.add(code)
            except Exception:
                pass

    return sorted(codes)


def so2sat_city_mgrs_codes(
    csv_path: Path | str = SO2SAT_BOUNDS_CSV,
    step_deg: float = 0.05,
) -> dict[str, list[str]]:
    """Return MGRS 100km square codes for every city in the So2Sat dataset.

    Args:
        csv_path: Path to ``so2sat_guppd_bounds.csv``.
        step_deg: Sampling step passed to :func:`bbox_to_mgrs_codes`.

    Returns:
        ``{city_name: [mgrs_code, ...]}`` for all 51 So2Sat cities.
    """
    df = pd.read_csv(csv_path)
    return {
        row.JRC_NAME_MAIN: bbox_to_mgrs_codes(
            row.minx, row.miny, row.maxx, row.maxy, step_deg
        )
        for _, row in df.iterrows()
    }


def _inclusive_range(start: float, stop: float, step: float) -> list[float]:
    """Float range that always includes both endpoints."""
    points: list[float] = []
    v = start
    while v < stop:
        points.append(v)
        v += step
    if not points or points[-1] < stop:
        points.append(stop)
    return points
