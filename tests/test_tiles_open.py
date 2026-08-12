"""Regression tests for datasets.tiles: open_tile's format dispatch and the
WGS84 tile footprints the spatial index is built from.

A zarr store is a DIRECTORY, so the ``.zarr`` branch has to be tested before the
``path.is_dir()`` branch that routes Tessera global tiles to the NPY reader.
When the order was the other way round, every ``.zarr`` tile raised
"NPY files not found" and the ``.zarr`` branch was unreachable — which broke the
``alpha_earth`` (GEE) and ``tessera`` families end to end.

The open_tile tests run against the real tile stores and skip when the data is
not mounted, so they are safe in an offline checkout. The footprint tests derive
everything from a tile name and need no data at all.
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


# ── Tile footprints (no data needed — derived from the tile name) ─────────────

# A near-equatorial tile and a high-latitude one. Reprojection distortion grows
# with latitude, so the second is where a bounding box over-claims most.
EQUATORIAL_TILE = "grid_36.85_-1.35"      # Nairobi
POLAR_TILE = "grid_16.55_78.05"           # Svalbard


@pytest.mark.parametrize("tile_name", [EQUATORIAL_TILE, POLAR_TILE])
def test_footprint_is_strictly_smaller_than_its_bounding_box(tile_name):
    """The whole point of exact_footprint_4326: the bbox claims land it has not got."""
    from shapely.geometry import box

    from datasets.tiles import tessera_grid_footprint_4326

    footprint = tessera_grid_footprint_4326(tile_name)
    bbox = box(*footprint.bounds)

    assert footprint.within(bbox)
    assert footprint.area < bbox.area


def test_a_patch_in_the_bbox_sliver_is_not_covered_by_the_exact_footprint():
    """The behaviour _fully_covered depends on, stated as a test.

    A point in the corner sliver passes a bounding-box coverage test and fails
    the exact one. Under the bbox the extraction accepted such patches and
    crop_patch returned a truncated array, which PatchDataset then stretched to
    the model's patch size — a silently distorted sample rather than a dropped
    one.
    """
    from shapely.geometry import box

    from datasets.tiles import tessera_grid_footprint_4326

    footprint = tessera_grid_footprint_4326(POLAR_TILE)
    bbox = box(*footprint.bounds)

    sliver = bbox.difference(footprint)
    assert not sliver.is_empty, "precondition: the bbox over-claims somewhere"

    probe = sliver.representative_point()
    assert probe.within(bbox)
    assert not probe.within(footprint)


def test_distortion_grows_with_latitude():
    """Ordering check: the correction is negligible at the equator, large at 78N.

    Pins the reason the fix exists at all — if these were the same magnitude the
    bounding box would have been good enough.
    """
    from shapely.geometry import box

    from datasets.tiles import tessera_grid_footprint_4326

    def over_claim(tile_name: str) -> float:
        fp = tessera_grid_footprint_4326(tile_name)
        return 1.0 - fp.area / box(*fp.bounds).area

    assert over_claim(EQUATORIAL_TILE) < 0.01
    assert over_claim(POLAR_TILE) > 0.05
    assert over_claim(POLAR_TILE) > 10 * over_claim(EQUATORIAL_TILE)


# ── tile_index_name ──────────────────────────────────────────────────────────
#
# build_tile_index returns the NPY *directory* for tesserav1.1_global, and
# Tessera names a tile after a fractional coordinate. Path therefore reads the
# trailing ".25" as a suffix, so `stem` truncates the latitude. All 8,108 v1.1
# global tiles are affected. It usually still yields the right UTM zone (set by
# longitude) but it flips the hemisphere just below the equator, and it can
# never match a tile referenced by its full name — which is how the corrupt-tile
# check in Task 2.0d silently measured nothing.


def test_stem_truncates_a_fractional_tile_name(tmp_path):
    """The precondition. If this ever fails, Path changed and the helper can go."""
    d = tmp_path / "grid_121.35_31.25"
    d.mkdir()
    assert d.stem == "grid_121.35_31"
    assert d.suffix == ".25"


def test_tile_index_name_keeps_the_full_name_for_a_tile_directory(tmp_path):
    from datasets.tiles import tile_index_name

    d = tmp_path / "grid_121.35_31.25"
    d.mkdir()
    assert tile_index_name(d) == "grid_121.35_31.25"


def test_tile_index_name_handles_the_npy_and_geoinfo_layouts(tmp_path):
    from datasets.tiles import tile_index_name

    d = tmp_path / "grid_121.35_31.25"
    d.mkdir()
    npy = d / "grid_121.35_31.25.npy"
    npy.touch()
    assert tile_index_name(npy) == "grid_121.35_31.25"

    tiff = tmp_path / "grid_0.15_52.05.tiff"
    tiff.touch()
    assert tile_index_name(tiff) == "grid_0.15_52.05"


def test_the_truncated_name_flips_the_hemisphere_just_below_the_equator(tmp_path):
    """The case that makes this a correctness bug and not just cosmetics:
    float("-0") >= 0 is True, so a southern tile is handed a northern EPSG."""
    from datasets.tiles import tessera_grid_geometry, tile_index_name

    d = tmp_path / "grid_-46.15_-0.95"
    d.mkdir()

    truncated = tessera_grid_geometry(d.stem)[0].to_epsg()
    correct = tessera_grid_geometry(tile_index_name(d))[0].to_epsg()

    assert truncated == 32623        # northern hemisphere, wrong
    assert correct == 32723          # southern hemisphere, right
