"""Confidence / purity filter tests, incl. the completeness (informal) trap."""

import pytest

from lcz_labels.classify import classify_patch
from lcz_labels.config import LczLabelConfig
from lcz_labels.tests.test_classify import row

CFG = LczLabelConfig()


def test_full_height_evidence_full_confidence():
    d = classify_patch(row(bsf=0.5, h_mean=30, height_evidence_frac=0.9, ghs_built_s=0.5), CFG)
    assert d["lcz"] == 1 and d["confidence"] == pytest.approx(1.0)


def test_raster_only_heights_capped():
    d = classify_patch(row(bsf=0.5, h_mean=30, height_evidence_frac=0.0,
                           height_none_frac=0.0, ghs_built_s=0.5), CFG)
    assert d["lcz"] == 1 and d["confidence"] == pytest.approx(CFG.confidence.raster_only_cap)


def test_no_height_tier_dropped():
    d = classify_patch(row(bsf=0.5, h_mean=30, height_none_frac=0.9, ghs_built_s=0.5), CFG)
    assert d["lcz"] is None and d["reject_reason"] == "insufficient_height_evidence"


def test_completeness_trap():
    # Positive natural evidence but GHS-BUILT-S says built -> under-mapped -> drop
    d = classify_patch(row(bsf=0.02, f_trees=0.9, ghs_built_s=0.30), CFG)
    assert d["lcz"] is None and d["reject_reason"] == "completeness_trap"


def test_acceptance_criterion_3():
    # No patch labelled sparse/natural where ghs_built_s>0.15 and bsf<0.05
    for f in ("f_water", "f_trees", "f_lowplants", "f_shrub", "f_sand", "f_bare_rock"):
        d = classify_patch(row(bsf=0.02, ghs_built_s=0.2, **{f: 0.9}), CFG)
        assert d["lcz"] is None


def test_suspect_ml_penalty():
    d = classify_patch(row(bsf=0.3, h_mean=15, height_evidence_frac=0.9,
                           ghs_built_s=0.0, built_area_ml_frac=1.0), CFG)
    assert d["lcz"] == 5 and d["suspect_ml"] is True
    assert d["confidence"] == pytest.approx(CFG.confidence.suspect_ml_penalty)


def test_config_hash_stable_and_yaml_roundtrip(tmp_path):
    c1 = LczLabelConfig()
    assert c1.config_hash == LczLabelConfig().config_hash
    p = c1.to_yaml(tmp_path / "cfg.yaml")
    c2 = LczLabelConfig.from_yaml(p)
    assert c2.config_hash == c1.config_hash
    # A threshold change must change the hash
    c3 = LczLabelConfig()
    c3.classification.bsf_compact_min = 0.5
    assert c3.config_hash != c1.config_hash
