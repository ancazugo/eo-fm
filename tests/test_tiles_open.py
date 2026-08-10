"""Regression tests for datasets.tiles.open_tile's format dispatch.

A zarr store is a DIRECTORY, so the ``.zarr`` branch has to be tested before the
``path.is_dir()`` branch that routes Tessera global tiles to the NPY reader.
When the order was the other way round, every ``.zarr`` tile raised
"NPY files not found" and the ``.zarr`` branch was unreachable — which broke the
``alpha_earth`` (GEE) and ``tessera`` families end to end.

These run against the real tile stores and skip when the data is not mounted,
so they are safe in an offline checkout.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DATA_DIR = Path(os.getenv("DATA_DIR", "/maps/acz25/phd-thesis-data"))
GEE_ALPHAEARTH_DIR = DATA_DIR / "input" / "Google" / "AlphaEarth" / "2017"
TESSERA_GLOBAL_DIR = Path("/tessera/v1.1/global_0.1_degree_representation/2017")


def _first(directory: Path, pattern: str) -> Path | None:
    if not directory.is_dir():
        return None
    return next(iter(sorted(directory.glob(pattern))), None)


def test_open_tile_reads_a_zarr_store():
    """A .zarr directory must reach the zarr branch, not the NPY reader."""
    from datasets.tiles import open_tile

    tile = _first(GEE_ALPHAEARTH_DIR, "gse_*.zarr")
    if tile is None:
        pytest.skip(f"No GEE AlphaEarth .zarr tiles under {GEE_ALPHAEARTH_DIR}")

    assert tile.is_dir(), "precondition: a zarr store is a directory"
    da = open_tile(tile)

    assert da.dims == ("band", "y", "x")
    assert da.sizes["band"] == 64
    assert da.rio.crs is not None


def test_open_tile_reads_a_tessera_npy_directory():
    """The is_dir() branch must still route Tessera global tiles to the NPY reader."""
    from datasets.tiles import open_tile

    tile = _first(TESSERA_GLOBAL_DIR, "grid_*")
    if tile is None:
        pytest.skip(f"No Tessera global tiles under {TESSERA_GLOBAL_DIR}")

    da = open_tile(tile)

    assert da.dims == ("band", "y", "x")
    assert da.sizes["band"] == 128
    assert da.rio.crs is not None
