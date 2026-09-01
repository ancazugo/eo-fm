"""G2 -- patch identity and edge erosion in the segmentation label burn.

Offline: rasterisation is a pure function of shapely geometries and a tile
footprint, so none of this needs the data mounts.

Two properties carry the weight.

**The label raster and the patch-UID raster must cover exactly the same
pixels.** Patch-level aggregation pools per-pixel softmax over the UID raster
and scores it against the label; if the two disagree even at the margins, the
headline kappa is computed over a slightly different pixel set than the labels
it is compared against, and nothing downstream would notice.

**Erosion must shrink, never delete.** So2Sat patch edges carry digitisation
slop that WUDAPT explicitly tolerates, so trimming them is right -- but a patch
small enough for erosion to erase has to survive un-eroded, because silently
dropping the smallest patches would bias the class distribution.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets.grid_tiles import (  # noqa: E402
    rasterize_patch_uids,
    rasterize_polys,
)

# 1280 m tile at 10 m/px = 128 px, matching create_city_grids --sub-tile-size.
TILE = box(0, 0, 1280, 1280)
SHAPE = (128, 128)
PATCHES = [(box(100, 100, 420, 420), 3, 0), (box(600, 600, 920, 920), 12, 1)]


def test_labels_use_the_project_wide_class_convention():
    lab = rasterize_polys(TILE, PATCHES, SHAPE)
    # Raw burn is 1-17 with 0 as nodata; the -1 shift happens in the dataset.
    assert set(np.unique(lab).tolist()) == {0, 3, 12}


def test_patch_uids_burn_with_minus_one_as_nodata():
    uid = rasterize_patch_uids(TILE, PATCHES, SHAPE)
    assert set(np.unique(uid).tolist()) == {-1, 0, 1}
    assert uid.dtype == np.int32


def test_uid_zero_survives_the_nodata_convention():
    """UIDs are burned as uid+1 and shifted back, so uid 0 is not swallowed by
    rasterio's fill=0."""
    single = [(box(100, 100, 420, 420), 3, 0)]
    uid = rasterize_patch_uids(TILE, single, SHAPE)
    assert (uid == 0).sum() > 0
    assert (uid == -1).sum() > 0


def test_the_label_and_uid_rasters_cover_identical_pixels():
    for erode in (0.0, 2.0, 4.0):
        lab = rasterize_polys(TILE, PATCHES, SHAPE, erode_px=erode)
        uid = rasterize_patch_uids(TILE, PATCHES, SHAPE, erode_px=erode)
        assert ((lab > 0) == (uid >= 0)).all(), f"mismatch at erode_px={erode}"


def test_erosion_removes_exactly_a_ring_of_pixels():
    """A 320 m patch is 32 px; eroding 2 px a side leaves 28x28."""
    lab = rasterize_polys(TILE, PATCHES, SHAPE, erode_px=0.0)
    eroded = rasterize_polys(TILE, PATCHES, SHAPE, erode_px=2.0)
    assert (lab > 0).sum() == 2 * 32 * 32
    assert (eroded > 0).sum() == 2 * 28 * 28


def test_erosion_keeps_a_patch_it_would_otherwise_erase():
    """Dropping the smallest patches would bias the class distribution."""
    tiny = [(box(100, 100, 110, 110), 7, 0)]      # 1 px across
    eroded = rasterize_polys(TILE, tiny, SHAPE, erode_px=8.0)
    assert (eroded > 0).sum() > 0
    assert set(np.unique(eroded).tolist()) == {0, 7}


def test_two_tuple_polys_still_work_for_the_label_burn():
    """The grid split mode and the pseudo-raster path both predate UIDs."""
    lab = rasterize_polys(TILE, [(box(0, 0, 640, 640), 5)], SHAPE)
    assert set(np.unique(lab).tolist()) == {0, 5}


def test_two_tuple_polys_yield_an_empty_uid_raster():
    """Rather than mis-indexing the class as an identity."""
    uid = rasterize_patch_uids(TILE, [(box(0, 0, 640, 640), 5)], SHAPE)
    assert (uid == -1).all()


def test_an_empty_tile_is_all_nodata():
    assert (rasterize_polys(TILE, [], SHAPE) == 0).all()
    assert (rasterize_patch_uids(TILE, [], SHAPE) == -1).all()


def test_erosion_is_disabled_at_zero():
    a = rasterize_polys(TILE, PATCHES, SHAPE, erode_px=0.0)
    b = rasterize_polys(TILE, PATCHES, SHAPE)
    assert (a == b).all()


# ── City names that are not glob-safe ─────────────────────────────────────────

def test_a_bracketed_city_name_is_not_read_as_a_glob_pattern(tmp_path):
    """Three So2Sat cities carry brackets: Osaka_[Kyoto],
    Quezon_City_[Manila], Rawalpindi_[Islamabad]. Matching their tiles with
    glob(f"{city}_*.npy") reads "[Kyoto]" as a character class, so the pattern
    matches nothing and the city vanishes from the run without a warning —
    which is exactly the kind of loss that never shows up as an error.
    """
    import geopandas as gpd
    from shapely.geometry import box as _box

    from datasets.grid_tiles import build_city_tile_items

    city = "Osaka_[Kyoto]"
    cdir = tmp_path / city
    (cdir / "AlphaEarthCoop" / "2017" / "train").mkdir(parents=True)
    for gid in range(3):
        np.save(cdir / "AlphaEarthCoop" / "2017" / "train" / f"{city}_{gid:02d}.npy",
                np.zeros((4, 8, 8), dtype=np.float32))

    cells = gpd.GeoDataFrame(
        {"grid_id": [0, 1, 2], "is_valid": [True] * 3},
        geometry=[_box(i * 1280, 0, (i + 1) * 1280, 1280) for i in range(3)],
        crs="EPSG:32653",
    )
    cells.to_file(cdir / f"{city}_grid.gpkg", driver="GPKG")

    patches = gpd.GeoDataFrame(
        {"patch_id": ["000001"], "dataset": ["training"],
         "LCZ_class": [3], "grid_id": [0], "split": ["train"]},
        geometry=[_box(100, 100, 420, 420)], crs="EPSG:32653",
    )
    patches.to_file(cdir / f"patches_reference_{city}_split.gpkg", driver="GPKG")

    items, split_map = build_city_tile_items(
        cdir, "AlphaEarthCoop", "2017", "gpkg", "LCZ_class")
    assert len(items) == 3, "bracketed city silently dropped"
    assert len(split_map) == 3
