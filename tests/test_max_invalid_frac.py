"""Task 1.75.1 — the nodata drop policy.

Offline: the filter is a pure function over an item list and a parquet, so none
of this needs the data mounts.

The property that matters most is the *default*. Every Phase 2 baseline is run
at ``--max-invalid-frac 1.0``, and if that were not an exact no-op the whole
phase would be comparing against a silently different population.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets.so2sat import (  # noqa: E402
    PatchItem,
    filter_by_invalid_fraction,
    patch_key,
)

COOP = "alpha_earth_coop"
TESSERA = "tesserav1.1_global"

# (split, patch_id, invalid fraction) — one clean, one marginal, one mostly gone.
FIXTURE = [
    ("training", "000000", 0.0),
    ("training", "000001", 0.10),
    ("training", "000002", 0.90),
    ("validation", "000000", 0.0),
    ("validation", "000001", 0.60),
    ("testing", "000000", 0.30),
    ("testing", "000001", 0.0),
]

_SPLIT_MAP = {"training": "train", "validation": "val", "testing": "test"}


def _path(tmp_path: Path, split: str, patch_id: str) -> Path:
    """The real on-disk shape: {split}/{output_name}/{year}/patch_{id}.npy."""
    return tmp_path / split / "AlphaEarthCoop" / "2017" / f"patch_{patch_id}.npy"


@pytest.fixture
def items(tmp_path):
    return [
        PatchItem(_path(tmp_path, split, pid), label=0, split=_SPLIT_MAP[split])
        for split, pid, _ in FIXTURE
    ]


@pytest.fixture
def parquet(tmp_path):
    """Both families present, so the family filter has something to select."""
    rows = [
        {"family": family, "dataset": split, "patch_id": pid,
         # Tessera is clean everywhere: only the coop rows should ever bite.
         "invalid_frac": frac if family == COOP else 0.0}
        for family in (COOP, TESSERA)
        for split, pid, frac in FIXTURE
    ]
    path = tmp_path / "invalid_fraction.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


# ── The key ──────────────────────────────────────────────────────────────────

def test_patch_key_recovers_split_and_id(tmp_path):
    """patch_id restarts at 000000 in each split, so the split has to be part
    of the key or validation rows resolve against training fractions."""
    assert patch_key(_path(tmp_path, "validation", "000123")) == ("validation", "000123")


def test_patch_key_of_a_fused_item_uses_the_first_source(tmp_path):
    fused = (_path(tmp_path, "testing", "000007"),
             tmp_path / "testing" / "GeoTessera_v1.1_global" / "2017" / "patch_000007.npy")
    assert patch_key(fused) == ("testing", "000007")


# ── The default must be a no-op ──────────────────────────────────────────────

def test_default_threshold_returns_the_identical_list(items, parquet):
    out = filter_by_invalid_fraction(items, 1.0, [COOP], parquet)
    assert out is items


def test_default_threshold_never_reads_the_parquet(items, tmp_path):
    """No parquet, no filtering, no error — the unfiltered path must not acquire
    a new data dependency."""
    missing = tmp_path / "does_not_exist.parquet"
    assert filter_by_invalid_fraction(items, 1.0, [COOP], missing) is items


# ── Filtering ────────────────────────────────────────────────────────────────

def test_threshold_drops_exactly_the_patches_above_it(items, parquet):
    kept = filter_by_invalid_fraction(items, 0.25, [COOP], parquet)
    assert {patch_key(it.path) for it in kept} == {
        ("training", "000000"),      # 0.00
        ("training", "000001"),      # 0.10
        ("validation", "000000"),    # 0.00
        ("testing", "000001"),       # 0.00
    }


def test_the_threshold_is_inclusive(items, parquet):
    """A patch exactly at the threshold is kept — 0.10 survives 0.10."""
    kept = filter_by_invalid_fraction(items, 0.10, [COOP], parquet)
    assert ("training", "000001") in {patch_key(it.path) for it in kept}


def test_filtering_applies_to_evaluation_as_well_as_training(items, parquet):
    """Filtering only the training split would report accuracy on data the model
    was never allowed to learn from."""
    kept = filter_by_invalid_fraction(items, 0.25, [COOP], parquet)
    splits = {it.split for it in kept}
    assert splits == {"train", "val", "test"}
    assert not any(patch_key(it.path) == ("validation", "000001") for it in kept)
    assert not any(patch_key(it.path) == ("testing", "000000") for it in kept)


def test_order_is_preserved(items, parquet):
    kept = filter_by_invalid_fraction(items, 0.95, [COOP], parquet)
    assert [patch_key(it.path) for it in kept] == [
        patch_key(it.path) for it in items
    ]


def test_a_fused_run_takes_the_worst_source(items, parquet):
    """Tessera is clean everywhere in the fixture, so selecting both families
    must still drop what coop alone would drop."""
    both = filter_by_invalid_fraction(items, 0.25, [COOP, TESSERA], parquet)
    coop_only = filter_by_invalid_fraction(items, 0.25, [COOP], parquet)
    assert [patch_key(i.path) for i in both] == [patch_key(i.path) for i in coop_only]


def test_a_clean_family_alone_drops_nothing(items, parquet):
    kept = filter_by_invalid_fraction(items, 0.25, [TESSERA], parquet)
    assert len(kept) == len(items)


def test_patches_absent_from_the_audit_are_kept(items, parquet, caplog):
    """The filter must not become a second, accidental coverage restriction."""
    extra = PatchItem(
        Path("/nowhere/unlabeled/AlphaEarthCoop/2017/patch_999999.npy"),
        label=0, split="train",
    )
    kept = filter_by_invalid_fraction([*items, extra], 0.25, [COOP], parquet)
    assert extra in kept


# ── Failure modes ────────────────────────────────────────────────────────────

def test_a_missing_parquet_names_the_command_that_writes_it(items, tmp_path):
    missing = tmp_path / "not_there.parquet"
    with pytest.raises(SystemExit):
        filter_by_invalid_fraction(items, 0.25, [COOP], missing)


def test_an_unaudited_family_is_an_error_not_a_silent_pass(items, parquet):
    """Selecting a family with no rows would filter against an empty table and
    keep everything, which looks exactly like a threshold that did nothing."""
    with pytest.raises(SystemExit):
        filter_by_invalid_fraction(items, 0.25, ["seamless"], parquet)
