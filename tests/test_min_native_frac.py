"""Rev A's native-fraction coverage filter (``--min-native-frac``).

Offline: the filter is a pure function over an item list and the Task 2.0
manifest, so none of this needs the data mounts.

Same shape as ``test_max_invalid_frac.py``, and for the same reason — the
property that matters most is the *default*. Every Task 2.1 anchor and every
Arm A run leaves this at 0.0, and if that were not an exact no-op the whole
phase would be comparing against a silently different population. The sign is
the one difference worth watching: this threshold is a floor, not a ceiling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets.so2sat import (  # noqa: E402
    PatchItem,
    filter_by_native_fraction,
    patch_key,
)

COOP = "alpha_earth_coop"
TESSERA = "tesserav1.1_global"

# (split, patch_id, native fraction for Tessera) — a whole tile, a mild edge
# crop, and the 2-px-per-side truncation Rev A found near tile boundaries.
FIXTURE = [
    ("training", "000000", 1.0),
    ("training", "000001", 0.75),
    ("training", "000002", 0.05),
    ("validation", "000000", 1.0),
    ("validation", "000001", 0.40),
    ("testing", "000000", 0.50),
    ("testing", "000001", 1.0),
]

_SPLIT_MAP = {"training": "train", "validation": "val", "testing": "test"}


def _path(tmp_path: Path, split: str, patch_id: str) -> Path:
    """The real on-disk shape: {split}/{output_name}/{year}/patch_{id}.npy."""
    return tmp_path / split / "GeoTessera_v1.1_global" / "2017" / f"patch_{patch_id}.npy"


@pytest.fixture
def items(tmp_path):
    return [
        PatchItem(_path(tmp_path, split, pid), label=0, split=_SPLIT_MAP[split])
        for split, pid, _ in FIXTURE
    ]


@pytest.fixture
def manifest(tmp_path):
    """Manifest-shaped: one row per patch, one native_frac column per family.
    Coop is whole everywhere, so only the Tessera column should ever bite."""
    rows = [
        {"dataset": split, "patch_id": pid,
         f"native_frac_{TESSERA}": frac,
         f"native_frac_{COOP}": 1.0,
         "in_manifest": True}
        for split, pid, frac in FIXTURE
    ]
    path = tmp_path / "patch_manifest_v1.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


# ── The default must be a no-op ──────────────────────────────────────────────

def test_default_threshold_returns_the_identical_list(items, manifest):
    out = filter_by_native_fraction(items, 0.0, [TESSERA], manifest)
    assert out is items


def test_default_threshold_never_reads_the_manifest(items, tmp_path):
    """No manifest, no filtering, no error — the unfiltered path must not
    acquire a new data dependency."""
    missing = tmp_path / "does_not_exist.parquet"
    assert filter_by_native_fraction(items, 0.0, [TESSERA], missing) is items


def test_a_negative_threshold_is_also_a_no_op(items, manifest):
    assert filter_by_native_fraction(items, -1.0, [TESSERA], manifest) is items


# ── Filtering ────────────────────────────────────────────────────────────────

def test_threshold_drops_exactly_the_patches_below_it(items, manifest):
    kept = filter_by_native_fraction(items, 0.5, [TESSERA], manifest)
    assert {patch_key(it.path) for it in kept} == {
        ("training", "000000"),      # 1.00
        ("training", "000001"),      # 0.75
        ("validation", "000000"),    # 1.00
        ("testing", "000000"),       # 0.50, exactly at the threshold
        ("testing", "000001"),       # 1.00
    }


def test_the_threshold_is_inclusive(items, manifest):
    """A patch exactly at the threshold is kept — 0.50 survives 0.50, matching
    the manifest's own `native_frac >= min_native_frac` rule."""
    kept = filter_by_native_fraction(items, 0.50, [TESSERA], manifest)
    assert ("testing", "000000") in {patch_key(it.path) for it in kept}


def test_filtering_applies_to_evaluation_as_well_as_training(items, manifest):
    """Filtering only the training split would report accuracy on data the model
    was never allowed to learn from."""
    kept = filter_by_native_fraction(items, 0.5, [TESSERA], manifest)
    assert {it.split for it in kept} == {"train", "val", "test"}
    assert not any(patch_key(it.path) == ("validation", "000001") for it in kept)
    assert not any(patch_key(it.path) == ("training", "000002") for it in kept)


def test_order_is_preserved(items, manifest):
    kept = filter_by_native_fraction(items, 0.1, [TESSERA], manifest)
    assert [patch_key(it.path) for it in kept] == [
        patch_key(it.path) for it in items if patch_key(it.path) != ("training", "000002")
    ]


def test_a_fused_run_takes_the_worst_source(items, manifest):
    """Coop is whole everywhere in the fixture, so selecting both families must
    still drop what Tessera alone would drop — the minimum decides."""
    both = filter_by_native_fraction(items, 0.5, [TESSERA, COOP], manifest)
    tessera_only = filter_by_native_fraction(items, 0.5, [TESSERA], manifest)
    assert [patch_key(i.path) for i in both] == [patch_key(i.path) for i in tessera_only]


def test_a_whole_family_alone_drops_nothing(items, manifest):
    kept = filter_by_native_fraction(items, 0.5, [COOP], manifest)
    assert len(kept) == len(items)


def test_patches_absent_from_the_manifest_are_kept(items, manifest):
    """The filter must not become a second, accidental coverage restriction."""
    extra = PatchItem(
        Path("/nowhere/unlabeled/GeoTessera_v1.1_global/2017/patch_999999.npy"),
        label=0, split="train",
    )
    kept = filter_by_native_fraction([*items, extra], 0.5, [TESSERA], manifest)
    assert extra in kept


# ── Failure modes ────────────────────────────────────────────────────────────

def test_a_missing_manifest_names_the_command_that_writes_it(items, tmp_path):
    missing = tmp_path / "not_there.parquet"
    with pytest.raises(SystemExit):
        filter_by_native_fraction(items, 0.5, [TESSERA], missing)


def test_a_family_without_a_column_is_an_error_not_a_silent_pass(items, manifest):
    """Selecting a family the manifest never measured would filter against no
    data and keep everything, which looks exactly like a threshold that did
    nothing."""
    with pytest.raises(SystemExit):
        filter_by_native_fraction(items, 0.5, ["seamless"], manifest)


def test_no_families_is_an_error_not_a_silent_pass(items, manifest):
    with pytest.raises(SystemExit):
        filter_by_native_fraction(items, 0.5, None, manifest)
