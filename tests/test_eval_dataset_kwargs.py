"""Evaluators must rebuild a checkpoint's own input pipeline.

ensemble_eval / tta_city_adapt / generate_pseudo_labels / eval_seg_on_patches
used to build their PatchDataset with no normalisation and no nodata handling,
so every checkpoint trained since 2026-08-11 (channel-normalised by default)
was scored, adapted or used as a teacher on raw embeddings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets.registry import provenance  # noqa: E402
from utils.runtime import eval_dataset_kwargs  # noqa: E402


def _ckpt(**meta):
    return {"model_state_dict": {}, **meta}


def test_a_pre_2026_08_checkpoint_gets_the_raw_pipeline():
    assert eval_dataset_kwargs(_ckpt(), "tesserav1.1_global") == {}
    assert eval_dataset_kwargs({"fc.weight": torch.zeros(1)}, "seamless") == {}  # bare state dict


def test_a_normalised_patch_checkpoint_gets_its_stats_and_masking():
    kw = eval_dataset_kwargs(_ckpt(normalize="channel",
                                   channel_mean=torch.ones(3), channel_std=torch.full((3,), 2.0)),
                             "alpha_earth_coop")
    assert kw["normalize"] == "channel"
    assert np.allclose(kw["channel_mean"], 1) and np.allclose(kw["channel_std"], 2)
    assert kw["nodata_mode"] == "mask" and callable(kw["nodata_predicate"])


def test_a_segmentation_checkpoint_masks_only_when_it_says_so():
    base = dict(normalize="channel", channel_mean=torch.zeros(2), channel_std=torch.ones(2))
    assert "nodata_mode" not in eval_dataset_kwargs(_ckpt(**base), "tesserav2",
                                                     pipeline="segmentation")
    kw = eval_dataset_kwargs(_ckpt(**base, nodata_mode="mask"), "tesserav2",
                             pipeline="segmentation")
    assert kw["nodata_mode"] == "mask"


def test_fused_sources_get_one_predicate_each():
    kw = eval_dataset_kwargs(_ckpt(nodata_mode="mask"), ["tesserav1.1_global", "aux_struct"])
    assert isinstance(kw["nodata_predicate"], list) and len(kw["nodata_predicate"]) == 2


def test_a_checkpoint_from_another_product_is_refused():
    with pytest.raises(ValueError, match="provenance"):
        eval_dataset_kwargs(_ckpt(**provenance("tesserav1.1")), "tesserav1.1_global")
