"""Decision-table + confidence tests (offline, synthetic UCP rows)."""

import pytest

from lcz_labels.classify import classify_patch
from lcz_labels.config import LczLabelConfig

CFG = LczLabelConfig()


def row(**kw):
    """A UCP row with all evidence zeroed unless overridden."""
    base = dict(
        bsf=0.0, h_mean=float("nan"), h_max=0.0, n_tower=0,
        height_evidence_frac=1.0, height_none_frac=0.0, mean_footprint_area=0.0,
        large_lowrise_frac=0.0, built_area_ml_frac=0.0,
        f_water=0.0, f_trees=0.0, f_lowplants=0.0, f_shrub=0.0, f_sand=0.0,
        f_bare_rock=0.0, f_paved_infra=0.0, f_industrial_lu=0.0,
        n_heavy_industry_poi=0, ghs_built_s=0.5,
    )
    base.update(kw)
    return base


def lcz(**kw):
    return classify_patch(row(**kw), CFG)["lcz"]


# ── Height x density matrix: every cell ──────────────────────────────────────

@pytest.mark.parametrize("bsf,h,expected", [
    (0.50, 30, 1),   # high compact
    (0.30, 30, 4),   # high open
    (0.10, 30, None),  # high sparse -> unlabelled
    (0.50, 15, 2),   # mid compact
    (0.30, 15, 5),   # mid open
    (0.10, 15, None),  # mid sparse -> unlabelled
    (0.30, 6, 6),    # low open
    # NB low+compact is NOT here — it goes through the Stage 5b router (test_router.py)
])
def test_matrix_cells(bsf, h, expected):
    assert lcz(bsf=bsf, h_mean=h, ghs_built_s=0.5) == expected


def test_lcz9_sparse_low_requires_vegetation():
    # low + sparse with vegetation -> LCZ 9, without -> unlabelled
    assert lcz(bsf=0.10, h_mean=6, f_lowplants=0.5, ghs_built_s=0.15) == 9
    assert lcz(bsf=0.10, h_mean=6, f_lowplants=0.0, ghs_built_s=0.15) is None


def test_high_via_towers_on_hmax():
    # h_mean mid but many towers with tall h_max -> high
    assert lcz(bsf=0.50, h_mean=15, h_max=40, n_tower=3, ghs_built_s=0.5) == 1


# ── Density boundary behaviour ───────────────────────────────────────────────

@pytest.mark.parametrize("bsf,expected", [
    (0.40, 1),   # exactly compact floor -> compact
    (0.20, 4),   # exactly open floor (high) -> open
    (0.05, None),  # exactly sparse floor, high -> unlabelled
])
def test_density_thresholds_inclusive(bsf, expected):
    assert lcz(bsf=bsf, h_mean=30, ghs_built_s=0.5) == expected


def test_boundary_penalty_applied():
    d = classify_patch(row(bsf=0.40, h_mean=30, ghs_built_s=0.5), CFG)
    assert d["lcz"] == 1 and d["boundary_flag"] is True
    assert d["confidence"] == pytest.approx(CFG.confidence.boundary_penalty)


# ── Natural / non-built classes ──────────────────────────────────────────────

@pytest.mark.parametrize("field,val,expected", [
    ("f_water", 0.80, 17),      # G
    ("f_trees", 0.80, 11),      # A dense
    ("f_shrub", 0.70, 13),      # C
    ("f_lowplants", 0.80, 14),  # D
    ("f_bare_rock", 0.70, 15),  # E rock
    ("f_paved_infra", 0.70, 15),  # E paved
    ("f_sand", 0.70, 16),       # F
])
def test_natural_classes(field, val, expected):
    assert lcz(bsf=0.0, ghs_built_s=0.0, **{field: val}) == expected


def test_scattered_trees_needs_veg_sum():
    # B: 0.35<=trees<0.75 and lowplants+trees>=0.75
    assert lcz(bsf=0.0, ghs_built_s=0.0, f_trees=0.5, f_lowplants=0.3) == 12
    # trees in range but not enough total vegetation -> not B
    assert lcz(bsf=0.0, ghs_built_s=0.0, f_trees=0.5, f_lowplants=0.0) != 12


def test_absence_is_not_evidence():
    # Empty patch (no built, no land cover) must NOT become a natural/sparse label
    assert lcz(bsf=0.0, ghs_built_s=0.0) is None


# ── Special built classes ────────────────────────────────────────────────────

def test_heavy_industry():
    assert lcz(bsf=0.30, h_mean=8, f_industrial_lu=0.6, n_heavy_industry_poi=2,
               ghs_built_s=0.5) == 10


def test_large_lowrise():
    assert lcz(bsf=0.30, h_mean=8, mean_footprint_area=1200, large_lowrise_frac=0.8,
               ghs_built_s=0.5) == 8


# ── LCZ 7 only ever comes from the compact-low router ────────────────────────

def test_lcz7_only_from_router():
    """LCZ 7 must never be emitted outside the compact+low cell (which routes)."""
    import itertools
    for bsf, h, veg in itertools.product(
        [0.0, 0.05, 0.1, 0.25, 0.6],   # exclude compact (>=0.4) low → that's the router
        [float("nan"), 12, 20, 30],    # exclude low height → router only fires at low
        [0.0, 0.4, 0.8],
    ):
        assert lcz(bsf=bsf, h_mean=h, f_lowplants=veg, ghs_built_s=0.3) != 7
