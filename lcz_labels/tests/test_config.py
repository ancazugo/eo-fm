"""Extraction-hash vs config-hash split (M0).

The Overture download cache keys on ``extraction_hash(aoi)`` so that
classification/confidence/block threshold changes never invalidate the
(hundreds of MB per city) raw extraction, while ``config_hash`` still keys
every derivation-stage cache.
"""

from lcz_labels.config import AOI, LczLabelConfig


def test_threshold_change_keeps_extraction_hash():
    base = LczLabelConfig()
    tweaked = LczLabelConfig()
    tweaked.classification.bsf_compact_min = 0.45
    assert base.config_hash != tweaked.config_hash
    assert base.extraction_hash("Nairobi") == tweaked.extraction_hash("Nairobi")


def test_block_param_change_keeps_extraction_hash():
    base = LczLabelConfig()
    tweaked = LczLabelConfig()
    tweaked.blocks.max_block_area_km2 = 1.0
    assert base.config_hash != tweaked.config_hash
    assert base.extraction_hash("Nairobi") == tweaked.extraction_hash("Nairobi")


def test_release_change_invalidates_both():
    base = LczLabelConfig()
    repinned = LczLabelConfig(overture_release="2026-07-22.0")
    assert base.config_hash != repinned.config_hash
    assert base.extraction_hash("Nairobi") != repinned.extraction_hash("Nairobi")


def test_extraction_hash_is_per_aoi():
    cfg = LczLabelConfig()
    assert cfg.extraction_hash("Nairobi") != cfg.extraction_hash("Paris")


def test_aoi_bbox_change_invalidates_extraction():
    named = LczLabelConfig(aoi_list=[AOI(name="X", bbox=(0.0, 0.0, 1.0, 1.0))])
    moved = LczLabelConfig(aoi_list=[AOI(name="X", bbox=(0.0, 0.0, 1.5, 1.0))])
    assert named.extraction_hash("X") != moved.extraction_hash("X")


def test_new_submodels_serialise_round_trip(tmp_path):
    cfg = LczLabelConfig()
    cfg.zones.min_zone_area_ha = 20.0
    cfg.export.erosion_px = 2
    cfg.change.built_delta_max = 0.05
    p = cfg.to_yaml(tmp_path / "cfg.yaml")
    back = LczLabelConfig.from_yaml(p)
    assert back.config_hash == cfg.config_hash
    assert back.zones.min_zone_area_ha == 20.0
    assert back.export.erosion_px == 2
    assert back.change.built_delta_max == 0.05
