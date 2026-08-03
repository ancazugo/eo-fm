"""T5 — dense majority-vote, block scoring, confusions, block_kind stratification."""

import numpy as np
import pandas as pd
import torch

from lcz_train.eval import (
    dense_predict_full,
    evaluate_blocks,
    majority_vote_per_block,
    patch_agreement,
)
from lcz_train.models import A1Linear


def test_dense_predict_full_chunked_matches_whole_image():
    torch.manual_seed(0)
    model = A1Linear(in_channels=4)
    mosaic = np.random.default_rng(0).normal(size=(4, 20, 20)).astype(np.float32)
    device = torch.device("cpu")
    full = dense_predict_full(model, mosaic, device, chunk=1000)
    chunked = dense_predict_full(model, mosaic, device, chunk=7)  # forces multiple tiles
    np.testing.assert_array_equal(full, chunked)
    assert full.shape == (20, 20)


def test_majority_vote_per_block_basic():
    pred = np.array([[0, 0, 5], [0, 2, 5]])   # 0-indexed classes
    block_idx = np.array([[1, 1, 2], [1, 2, 2]], dtype=np.uint32)
    out = majority_vote_per_block(pred, block_idx, n_blocks=2)
    assert out[0] == 1     # block 1: classes [0,0,0] -> majority class 0 -> LCZ 1
    assert out[1] == 6     # block 2: classes [5,5,2] -> majority class 5 -> LCZ 6


def test_majority_vote_per_block_no_votes_is_zero():
    pred = np.zeros((2, 2), dtype=np.int64)
    block_idx = np.zeros((2, 2), dtype=np.uint32)   # block 3 has no pixels here
    out = majority_vote_per_block(pred, block_idx, n_blocks=3)
    assert out[2] == 0


def _gt_frame():
    return pd.DataFrame({
        "label_type": ["hard", "hard", "hard", "coarse", "hard", "unlabelled"],
        "lcz": [3, 3, 7, None, 2, None],
        "lcz_set": [[3], [3], [7], [3, 7], [2], []],
        "block_kind": ["enclosure", "enclosure", "grid_fallback", "enclosure", "mn", "enclosure"],
    })


def test_evaluate_blocks_oa_and_hard_metrics():
    gt = _gt_frame()
    pred = np.array([3, 6, 7, 3, 2, 5])   # row1 wrong (3->6), row3 coarse hit, rest correct
    result = evaluate_blocks(pred, gt)
    assert result["n"] == 5   # unlabelled row (pred=5 but label_type=unlabelled) excluded
    assert result["oa"] == 4 / 5
    assert result["confusions"]["3->6"] == 1
    assert result["confusions"]["7->3"] == 0


def test_evaluate_blocks_coarse_reported_separately():
    gt = _gt_frame()
    pred = np.array([3, 3, 7, 7, 2, 1])   # coarse row predicted 7 (in {3,7}) -> correct
    result = evaluate_blocks(pred, gt)
    assert result["n_coarse"] == 1
    assert result["coarse_oa"] == 1.0
    assert result["hard_oa"] == 1.0    # all 4 hard rows correct


def test_evaluate_blocks_per_class_f1():
    gt = pd.DataFrame({
        "label_type": ["hard"] * 4,
        "lcz": [1, 1, 2, 2],
        "lcz_set": [[1], [1], [2], [2]],
        "block_kind": ["enclosure"] * 4,
    })
    pred = np.array([1, 2, 2, 2])   # class1: 1 tp,1 fn; class2: 2 tp, 1 fp
    result = evaluate_blocks(pred, gt)
    assert result["per_class"][1]["recall"] == 0.5
    assert result["per_class"][1]["precision"] == 1.0
    assert result["per_class"][2]["recall"] == 1.0
    assert result["per_class"][2]["precision"] == 2 / 3


def test_evaluate_blocks_stratified_by_block_kind():
    gt = _gt_frame()
    pred = np.array([3, 3, 3, 3, 2, 1])   # grid_fallback row (idx2, GT=7) wrong
    result = evaluate_blocks(pred, gt)
    strat = result["by_block_kind"]
    assert strat["grid_fallback"]["oa"] == 0.0
    assert strat["enclosure"]["oa"] == 1.0
    assert strat["mn"]["oa"] == 1.0


def test_evaluate_blocks_unpredicted_rows_excluded():
    gt = _gt_frame()
    pred = np.array([3, 3, 0, 3, 2, 0])  # block 3 got no votes (pred=0)
    result = evaluate_blocks(pred, gt)
    assert result["n"] == 4   # the pred=0 hard row is dropped, not scored wrong


def test_evaluate_blocks_empty_returns_n_zero():
    gt = pd.DataFrame({"label_type": ["unlabelled"], "lcz": [None], "lcz_set": [[]],
                      "block_kind": ["enclosure"]})
    result = evaluate_blocks(np.array([3]), gt)
    assert result == {"n": 0}


def test_patch_agreement_basic():
    df = pd.DataFrame({
        "dominant_frac": [0.9, 0.5, 0.8],
        "dominant_set": [[3], [3], [7]],
        "so2sat_lcz": [3.0, 3.0, 3.0],
    })
    result = patch_agreement(df, min_dominant=0.75)
    assert result["n"] == 2       # the 0.5-dominant row is excluded
    assert result["oa"] == 0.5    # row0 correct, row2 (7) wrong


def test_patch_agreement_no_ground_truth_returns_empty():
    df = pd.DataFrame({"dominant_frac": [0.9], "dominant_set": [[3]]})
    assert patch_agreement(df) == {}
