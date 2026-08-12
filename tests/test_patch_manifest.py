"""Task 2.0 — the common patch manifest.

Offline: the filter is a pure function over an item list and a parquet, and the
native-fraction rule is arithmetic, so none of this needs the data mounts.

Two properties carry the most weight. The **default** must be an exact no-op,
because Task 2.1's anchor and every Arm B run are defined as "no manifest" and a
default that quietly filtered would make them a different experiment. And the
native-fraction rule must be **relative**, because `seamless` is legitimately
11-13 px where the 10 m families are 33, and an absolute threshold erases it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets.so2sat import (  # noqa: E402
    PatchItem,
    filter_by_manifest,
    manifest_sha256,
    patch_key,
)
from diagnostics.patch_manifest import build_manifest, native_frac  # noqa: E402

COOP = "alpha_earth_coop"
TESSERA = "tesserav1.1_global"
SEAMLESS = "seamless"

# (split, patch_id, in_manifest)
FIXTURE = [
    ("training", "000000", True),
    ("training", "000001", False),
    ("training", "000002", True),
    ("validation", "000000", False),
    ("validation", "000001", True),
    ("testing", "000000", True),
    ("testing", "000001", False),
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
def manifest(tmp_path):
    path = tmp_path / "patch_manifest_v1.parquet"
    pd.DataFrame(
        [{"dataset": s, "patch_id": p, "in_manifest": m} for s, p, m in FIXTURE]
    ).to_parquet(path, index=False)
    return path


# ── The default must be a no-op ──────────────────────────────────────────────

def test_no_manifest_returns_the_identical_list(items):
    assert filter_by_manifest(items, None) is items


def test_no_manifest_never_touches_the_filesystem(items):
    """Passing None must not acquire a data dependency the anchor runs lack."""
    assert filter_by_manifest(items, None) is items


# ── Filtering ────────────────────────────────────────────────────────────────

def test_drops_exactly_the_non_members(items, manifest):
    kept = filter_by_manifest(items, manifest)
    assert {patch_key(it.path) for it in kept} == {
        ("training", "000000"),
        ("training", "000002"),
        ("validation", "000001"),
        ("testing", "000000"),
    }


def test_filtering_applies_to_evaluation_as_well_as_training(items, manifest):
    """Restricting only the training split would evaluate on patches the model
    was never allowed to learn from — a different experiment from Arm A."""
    kept = filter_by_manifest(items, manifest)
    assert {it.split for it in kept} == {"train", "val", "test"}
    assert not any(patch_key(it.path) == ("validation", "000000") for it in kept)
    assert not any(patch_key(it.path) == ("testing", "000001") for it in kept)


def test_order_is_preserved(items, manifest):
    kept = filter_by_manifest(items, manifest)
    survivors = {patch_key(it.path) for it in kept}
    assert [patch_key(it.path) for it in kept] == [
        patch_key(it.path) for it in items if patch_key(it.path) in survivors
    ]


def test_patch_id_alone_is_not_the_key(items, manifest):
    """patch_id restarts at 000000 in every split. training/000000 is a member
    and validation/000000 is not, so a filter keyed on the id alone would keep
    both or drop both."""
    kept = {patch_key(it.path) for it in filter_by_manifest(items, manifest)}
    assert ("training", "000000") in kept
    assert ("validation", "000000") not in kept


def test_patches_outside_the_manifest_universe_are_kept(items, manifest):
    """Unlabeled and pseudo-labelled pools are not in the manifest's universe;
    dropping them would make this a silent second coverage restriction."""
    extra = PatchItem(
        Path("/nowhere/unlabeled/AlphaEarthCoop/2017/patch_999999.npy"),
        label=0, split="train",
    )
    assert extra in filter_by_manifest([*items, extra], manifest)


# ── Failure modes ────────────────────────────────────────────────────────────

def test_a_missing_manifest_is_an_error_not_a_silent_pass(items, tmp_path):
    with pytest.raises(SystemExit):
        filter_by_manifest(items, tmp_path / "not_there.parquet")


# ── Hash ─────────────────────────────────────────────────────────────────────

def test_sha256_is_none_without_a_manifest():
    assert manifest_sha256(None) is None


def test_sha256_changes_when_the_manifest_does(tmp_path, manifest):
    first = manifest_sha256(manifest)
    pd.DataFrame([{"dataset": "training", "patch_id": "000000",
                   "in_manifest": False}]).to_parquet(manifest, index=False)
    assert manifest_sha256(manifest) != first


# ── native_frac is relative, not absolute ────────────────────────────────────

def test_native_frac_is_relative_to_each_family(tmp_path):
    """A 12x12 seamless crop is full size; a 12x12 Tessera crop has lost 87% of
    its extent. An absolute pixel threshold cannot tell those apart."""
    df = pd.DataFrame([
        {"family": SEAMLESS, "h": 12, "w": 12},
        {"family": TESSERA, "h": 12, "w": 12},
    ])
    frac = native_frac(df, {SEAMLESS: 144.0, TESSERA: 1122.0})
    assert frac.iloc[0] == pytest.approx(1.0)
    assert frac.iloc[1] < 0.15


def test_a_truncated_seamless_crop_still_fails(tmp_path):
    """3x11 = 33 px against a 144 median is 0.23 — the relative rule must still
    catch ESD's own truncated tail, not just wave every small crop through."""
    df = pd.DataFrame([{"family": SEAMLESS, "h": 3, "w": 11}])
    assert native_frac(df, {SEAMLESS: 144.0}).iloc[0] < 0.5


# ── build_manifest ───────────────────────────────────────────────────────────

def _scan_row(family, dataset, pid, frac=0.0, h=33, w=34, city="Nairobi", lcz=6.0):
    return {"family": family, "dataset": dataset, "patch_id": pid,
            "invalid_frac": frac, "h": h, "w": w, "city": city, "lcz": lcz}


def test_membership_requires_every_family():
    """Present in two of three is not a member: the whole point is that all
    three train on the same patches."""
    rows = [
        # a: everywhere and clean       b: absent from tessera
        _scan_row(COOP, "training", "a"), _scan_row(TESSERA, "training", "a"),
        _scan_row(SEAMLESS, "training", "a", h=12, w=12),
        _scan_row(COOP, "training", "b"),
        _scan_row(SEAMLESS, "training", "b", h=12, w=12),
    ]
    man, meta = build_manifest(pd.DataFrame(rows), [COOP, TESSERA, SEAMLESS], 0.25, 0.5)
    got = dict(zip(man["patch_id"], man["in_manifest"]))
    assert got == {"a": True, "b": False}
    assert meta["n_members"] == 1
    assert man.loc[man["patch_id"] == "b", f"present_{TESSERA}"].item() is False


def test_a_single_family_failing_a_criterion_excludes_the_patch():
    rows = [
        _scan_row(COOP, "training", "a", frac=0.9),      # too much nodata
        _scan_row(TESSERA, "training", "a"),
        _scan_row(SEAMLESS, "training", "a", h=12, w=12),
        _scan_row(COOP, "training", "b"),
        _scan_row(TESSERA, "training", "b", h=4, w=4),   # truncated
        _scan_row(SEAMLESS, "training", "b", h=12, w=12),
    ]
    man, _ = build_manifest(pd.DataFrame(rows), [COOP, TESSERA, SEAMLESS], 0.25, 0.5)
    assert not man["in_manifest"].any()
    # ...and the flags say which family was responsible, for the audit.
    assert man.loc[man["patch_id"] == "a", f"ok_{COOP}"].item() is False
    assert man.loc[man["patch_id"] == "a", f"ok_{TESSERA}"].item() is True
    assert man.loc[man["patch_id"] == "b", f"ok_{TESSERA}"].item() is False


def test_non_members_stay_in_the_table():
    """The loader has to tell 'excluded' from 'outside the universe', which only
    an explicit in_manifest=False row supports."""
    rows = [
        _scan_row(COOP, "training", "a", frac=0.9),
        _scan_row(TESSERA, "training", "a"),
        _scan_row(SEAMLESS, "training", "a", h=12, w=12),
    ]
    man, meta = build_manifest(pd.DataFrame(rows), [COOP, TESSERA, SEAMLESS], 0.25, 0.5)
    assert len(man) == 1 and meta["n_reference"] == 1 and meta["n_members"] == 0


def test_the_threshold_is_inclusive_on_both_criteria():
    """A patch exactly at max_invalid_frac, or exactly at min_native_frac,
    is kept — the boundary belongs to the members."""
    rows = [
        _scan_row(COOP, "training", "a", frac=0.25),
        _scan_row(TESSERA, "training", "a", h=33, w=17),   # 561/1122 = 0.500
        _scan_row(SEAMLESS, "training", "a", h=12, w=12),
        # Two more full-size patches, so the median native area is 1122 rather
        # than being dragged down by the one truncated crop under test.
        _scan_row(COOP, "training", "b"), _scan_row(TESSERA, "training", "b"),
        _scan_row(SEAMLESS, "training", "b", h=12, w=12),
        _scan_row(COOP, "training", "c"), _scan_row(TESSERA, "training", "c"),
        _scan_row(SEAMLESS, "training", "c", h=12, w=12),
    ]
    man, meta = build_manifest(pd.DataFrame(rows), [COOP, TESSERA, SEAMLESS], 0.25, 0.5)
    assert meta["expected_native_area"][TESSERA] == pytest.approx(1122.0)
    assert man.loc[man["patch_id"] == "a", "in_manifest"].item() is True


def test_denominator_comes_from_the_training_split_only():
    """Deriving it per split would let a small or odd evaluation split move the
    rule under the training data."""
    rows = [
        _scan_row(COOP, "training", "a"), _scan_row(TESSERA, "training", "a"),
        _scan_row(SEAMLESS, "training", "a", h=12, w=12),
        # A degenerate testing split that would halve the denominator if used
        _scan_row(COOP, "testing", "a", h=8, w=8),
        _scan_row(TESSERA, "testing", "a", h=8, w=8),
        _scan_row(SEAMLESS, "testing", "a", h=6, w=6),
    ]
    _, meta = build_manifest(pd.DataFrame(rows), [COOP, TESSERA, SEAMLESS], 0.25, 0.5)
    assert meta["expected_native_area"][TESSERA] == pytest.approx(33 * 34)
    assert meta["denominator_split"] == "training"


def test_a_family_with_no_training_rows_is_an_error():
    """Without a training split there is no denominator, and silently falling
    back to another family's would compare ESD against a 10 m expectation."""
    rows = [
        _scan_row(COOP, "training", "a"), _scan_row(TESSERA, "training", "a"),
        _scan_row(SEAMLESS, "testing", "a", h=12, w=12),
    ]
    with pytest.raises(SystemExit):
        build_manifest(pd.DataFrame(rows), [COOP, TESSERA, SEAMLESS], 0.25, 0.5)
