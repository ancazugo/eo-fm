"""T7 — ladder orchestration on synthetic data: build_model, one A rung, one B
rung, B3's graceful skip (no torch_geometric here), and the markdown table."""

import numpy as np
import pandas as pd
import pytest
import torch

from lcz_train.config import TrainConfig
from lcz_train.datasets import BlockDataset, PixelWindowDataset
from lcz_train.experiments import (
    build_model,
    render_ladder_table,
    run_block_experiment,
    run_dense_experiment,
)
from lcz_train.models import A1Linear, A3DilatedConv, MultiScaleWrapper


@pytest.mark.parametrize("exp_id,expect", [
    ("A1", A1Linear), ("A1_MS", MultiScaleWrapper), ("A2_MS", MultiScaleWrapper),
    ("A3", A3DilatedConv),
])
def test_build_model_dense_rungs(exp_id, expect):
    model = build_model(exp_id, in_channels=16)
    assert isinstance(model, expect)


@pytest.mark.parametrize("exp_id", ["B1", "B2"])
def test_build_model_block_rungs(exp_id):
    model = build_model(exp_id, in_channels=16, extra_features=3)
    x = torch.randn(2, 16 + 3) if exp_id == "B1" else None
    assert model is not None


def test_build_model_b3_raises_without_gnn_extra():
    with pytest.raises(ImportError, match="torch_geometric"):
        build_model("B3", in_channels=16)


def test_build_model_unknown_raises():
    with pytest.raises(ValueError):
        build_model("Z9", in_channels=16)


def _dense_raster_set(h=200, w=200, c=6, seed=0):
    rng = np.random.default_rng(seed)
    mid_h, mid_w = h // 2, w // 2
    bitmask = np.zeros((h, w), dtype=np.uint32)
    bitmask[:mid_h, :mid_w] = 1 << 2
    bitmask[:mid_h, mid_w:] = 1 << 5
    conf = np.where(bitmask != 0, 90, 0).astype(np.uint8)
    block_idx = np.zeros((h, w), dtype=np.uint32)
    block_idx[:mid_h, :mid_w] = 1
    block_idx[:mid_h, mid_w:] = 2
    block_idx[mid_h:, :] = 3
    mosaic = 0.05 * rng.normal(size=(c, h, w)).astype(np.float32)
    mosaic[0, :, :mid_w] += 1.0
    mosaic[0, :, mid_w:] -= 1.0
    return mosaic, bitmask, conf, block_idx


def test_run_dense_experiment_end_to_end():
    mosaic, bitmask, conf, block_idx = _dense_raster_set()
    train_ds = PixelWindowDataset(mosaic, bitmask, conf, block_idx, window_px=96,
                                  erosion_px=1, seed=0)
    gt = pd.DataFrame({
        "label_type": ["hard", "hard", "unlabelled"],
        "lcz": [3, 6, None], "lcz_set": [[3], [6], []],
        "block_kind": ["enclosure"] * 3,
    })
    cfg = TrainConfig(exp_id="A1", steps=40, batch_size=8, lr=5e-2, log_every=40, seed=0)
    result = run_dense_experiment("A1", train_ds, mosaic, block_idx, gt, cfg, in_channels=6,
                                  device=torch.device("cpu"))
    assert result["exp_id"] == "A1" and result["family"] == "A"
    assert result["n_params"] > 0
    assert result["metrics"]["n"] == 2   # only the 2 labelled blocks scored
    assert result["metrics"]["oa"] > 0.5   # the signal is trivially learnable


def _blocks_df(n=6, seed=0):
    # Geometric extras kept on a scale comparable to the embedding signal —
    # this test checks orchestration wiring, not feature normalisation.
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "block_id": [f"b{i}" for i in range(n)],
        "area_m2": rng.uniform(0, 1, n),
        "compactness": rng.uniform(0, 1, n),
        "elongation": rng.uniform(0, 1, n),
        "bitmask": [1 << 2] * (n // 2) + [1 << 5] * (n - n // 2),
        "confidence": rng.uniform(0.5, 1.0, n),
    })


def test_run_block_experiment_end_to_end():
    rng = np.random.default_rng(0)
    blocks_df = _blocks_df(n=8)
    pooled = np.zeros((8, 4), dtype=np.float32)
    pooled[:4, 0] = 3.0     # class-3 blocks: strong signal on channel 0
    pooled[4:, 0] = -3.0    # class-6 blocks
    pooled += 0.05 * rng.normal(size=pooled.shape)
    train_ds = BlockDataset(pooled, blocks_df)
    gt = pd.DataFrame({
        "label_type": ["hard"] * 8,
        "lcz": [3] * 4 + [6] * 4,
        "lcz_set": [[3]] * 4 + [[6]] * 4,
        "block_kind": ["enclosure"] * 8,
    })
    cfg = TrainConfig(exp_id="B1", steps=60, batch_size=4, lr=1e-2, log_every=60, seed=0)
    result = run_block_experiment("B1", train_ds, train_ds, gt, cfg, in_channels=4,
                                  extra_features=3, device=torch.device("cpu"))
    assert result is not None
    assert result["exp_id"] == "B1" and result["family"] == "B"
    assert result["metrics"]["n"] == 8
    assert result["metrics"]["oa"] > 0.5


def test_run_block_experiment_b3_returns_none_gracefully():
    blocks_df = _blocks_df(n=4)
    pooled = np.zeros((4, 4), dtype=np.float32)
    train_ds = BlockDataset(pooled, blocks_df)
    gt = pd.DataFrame({"label_type": ["hard"] * 4, "lcz": [3, 3, 6, 6],
                      "lcz_set": [[3], [3], [6], [6]], "block_kind": ["enclosure"] * 4})
    cfg = TrainConfig(exp_id="B3", steps=5, batch_size=4, log_every=5)
    result = run_block_experiment("B3", train_ds, train_ds, gt, cfg, in_channels=4)
    assert result is None


def test_render_ladder_table_handles_skipped_rungs():
    results = [
        {"exp_id": "A1", "family": "A", "n_params": 100, "train_time_s": 1.5,
         "metrics": {"n": 2, "oa": 0.8, "macro_f1": 0.75, "coarse_oa": 0.5,
                    "per_class": {3: {"f1": 0.9}, 6: {"f1": 0.6}}}},
        {"exp_id": "B1", "family": "B", "n_params": 50, "train_time_s": 0.5,
         "metrics": {"n": 2, "oa": 0.9, "macro_f1": 0.85, "coarse_oa": float("nan"),
                    "per_class": {3: {"f1": 0.95}, 6: {"f1": 0.75}}}},
        None,   # B3 skipped
    ]
    table = render_ladder_table(results)
    assert "A1" in table and "B1" in table
    assert "skipped" in table
    assert "LCZ3: +0.050" in table   # 0.95 - 0.9
    assert "LCZ6: +0.150" in table   # 0.75 - 0.6
