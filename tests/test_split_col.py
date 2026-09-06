"""``--split-col``: custom train/val/test assignments on the global split.

Offline: writes a tiny GeoPackage and a fake patch index, so no data mounts.

The property under test is a *separation*. The global GPKG's 'dataset' column
means two things at once -- which of training/validation/testing subdirectory
holds a patch's npy, and which split it belongs to -- and patch_ids restart at
000000 in each of those directories. So expressing a custom split by rewriting
'dataset' does not merely mislabel a patch: it repoints it at a different file
that happens to share its id. That is the same failure mode as the patch_id
collision bug of 2026-06, which silently poisoned every run up to ~252.
``split_col`` exists so the split can be restated without touching the column
that resolves the path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets.so2sat import build_global_items  # noqa: E402

# patch_id "000000" deliberately exists in BOTH directories, as it does on disk.
PATCH_INDEX = {
    "training":   {"000000": Path("/fake/training/patch_000000.npy"),
                   "000001": Path("/fake/training/patch_000001.npy")},
    "validation": {"000000": Path("/fake/validation/patch_000000.npy")},
    "testing":    {"000000": Path("/fake/testing/patch_000000.npy")},
}


def _write_gpkg(tmp_path: Path, rows: list[dict]) -> Path:
    gdf = gpd.GeoDataFrame(rows, geometry=[Point(0, i) for i in range(len(rows))],
                           crs="EPSG:4326")
    path = tmp_path / "custom_split.gpkg"
    gdf.to_file(path, driver="GPKG")
    return path


def _rows() -> list[dict]:
    # 'fold' restates the split WITHOUT moving any patch between directories.
    return [
        {"patch_id": "000000", "dataset": "training",   "LCZ_class": 1, "fold": "train"},
        {"patch_id": "000001", "dataset": "training",   "LCZ_class": 2, "fold": "test"},
        {"patch_id": "000000", "dataset": "validation", "LCZ_class": 3, "fold": "val"},
        {"patch_id": "000000", "dataset": "testing",    "LCZ_class": 4, "fold": "train"},
    ]


def test_split_col_reassigns_splits_without_moving_files(tmp_path):
    """The point of the flag: split follows 'fold', path still follows 'dataset'."""
    gpkg = _write_gpkg(tmp_path, _rows())
    items = build_global_items(gpkg, PATCH_INDEX, split_col="fold")

    by_path = {str(it.path): it.split for it in items}
    assert by_path == {
        "/fake/training/patch_000000.npy":   "train",
        "/fake/training/patch_000001.npy":   "test",   # moved split, same dir
        "/fake/validation/patch_000000.npy": "val",
        "/fake/testing/patch_000000.npy":    "train",  # moved split, same dir
    }


def test_dataset_column_still_resolves_the_path(tmp_path):
    """The collision guard.

    Three rows share patch_id "000000" and differ only by 'dataset'. If the
    path lookup ever started keying on the split instead, they would collapse
    onto one file -- silently, with plausible-looking labels.
    """
    gpkg = _write_gpkg(tmp_path, _rows())
    items = build_global_items(gpkg, PATCH_INDEX, split_col="fold")

    zero_paths = {str(it.path) for it in items if "000000" in str(it.path)}
    assert zero_paths == {
        "/fake/training/patch_000000.npy",
        "/fake/validation/patch_000000.npy",
        "/fake/testing/patch_000000.npy",
    }


def test_default_is_unchanged(tmp_path):
    """Without the flag, 'dataset' drives the split exactly as before."""
    gpkg = _write_gpkg(tmp_path, _rows())
    items = build_global_items(gpkg, PATCH_INDEX)

    assert sorted(it.split for it in items) == ["test", "train", "train", "val"]
    train = {str(it.path) for it in items if it.split == "train"}
    assert train == {"/fake/training/patch_000000.npy",
                     "/fake/training/patch_000001.npy"}


def test_long_and_short_split_names_both_work(tmp_path):
    rows = _rows()
    for r, name in zip(rows, ["training", "testing", "validation", "training"]):
        r["fold"] = name
    gpkg = _write_gpkg(tmp_path, rows)
    items = build_global_items(gpkg, PATCH_INDEX, split_col="fold")
    assert sorted(it.split for it in items) == ["test", "train", "train", "val"]


def test_missing_column_raises(tmp_path):
    gpkg = _write_gpkg(tmp_path, _rows())
    with pytest.raises(ValueError, match="not a column"):
        build_global_items(gpkg, PATCH_INDEX, split_col="nope")


def test_empty_split_raises_rather_than_training_on_a_truncated_set(tmp_path):
    """A typo'd or unmapped column must fail loudly.

    Silently dropping every test patch would still train, still log, and still
    report a metric -- computed on whatever survived.
    """
    rows = _rows()
    for r in rows:
        r["fold"] = "train"
    gpkg = _write_gpkg(tmp_path, rows)
    with pytest.raises(ValueError, match="produced no val/test"):
        build_global_items(gpkg, PATCH_INDEX, split_col="fold")
