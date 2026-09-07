"""restrict_to_dataset in evaluate_segmentation_as_patches.

Offline: a fake task and a hand-built batch, no data mounts or a real model.

An eval-only loader (no tile purity, see test_grid_tile_labels.py's
eval_only tests) can put patches of different `dataset` values inside the
same batch. `restrict_to_dataset=True` must narrow the scored population
down to `dataset_filter`; the default `False` must be a complete no-op,
because a --split-mode grid test tile can legitimately mix in
dataset=="training" patches (grid mode has no culture-10 concept), and
silently filtering those out under the default dataset_filter="testing"
would gut that mode's existing metric.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.evaluate import evaluate_segmentation_as_patches  # noqa: E402

# num_classes=2 (not 17) so _lcz_suite short-circuits and this stays offline.
_NUM_CLASSES = 2


class _FakeTask:
    """Always predicts class 0, regardless of input."""

    def eval(self) -> None:
        pass

    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        b, _, h, w = imgs.shape
        logits = torch.full((b, _NUM_CLASSES, h, w), -10.0)
        logits[:, 0] = 10.0
        return logits


def _batch(uid_map: np.ndarray) -> dict:
    h, w = uid_map.shape
    return {
        "image": torch.zeros(1, 3, h, w),
        "patch_uid": torch.from_numpy(uid_map).long().unsqueeze(0),
    }


def _make_loader():
    # A 4x4 tile split in half: uid 0 (left) is a testing patch, uid 1
    # (right) is a training patch that happens to share the tile — exactly
    # the situation --split-mode eval_only produces and --split-mode global
    # never would (its tiles are pure by construction).
    uid_map = np.array([[0, 0, 1, 1]] * 4, dtype=np.int64)
    return [_batch(uid_map)]


def test_restrict_to_dataset_false_scores_every_patch_in_the_loader(tmp_path):
    uid_to_label = {0: 0, 1: 0}
    uid_to_key = {0: ("Nairobi", "testing", "000001"),
                  1: ("Nairobi", "training", "000002")}
    results = evaluate_segmentation_as_patches(
        _FakeTask(), _make_loader(), torch.device("cpu"), _NUM_CLASSES,
        tmp_path, "test", use_wandb=False,
        uid_to_label=uid_to_label, uid_to_key=uid_to_key, save_probs=False,
    )
    assert results["test_n_patch"] == 2


def test_restrict_to_dataset_true_narrows_to_the_target_dataset(tmp_path):
    uid_to_label = {0: 0, 1: 0}
    uid_to_key = {0: ("Nairobi", "testing", "000001"),
                  1: ("Nairobi", "training", "000002")}
    results = evaluate_segmentation_as_patches(
        _FakeTask(), _make_loader(), torch.device("cpu"), _NUM_CLASSES,
        tmp_path, "test", use_wandb=False,
        uid_to_label=uid_to_label, uid_to_key=uid_to_key, save_probs=False,
        dataset_filter="testing", restrict_to_dataset=True,
    )
    assert results["test_n_patch"] == 1


def test_restrict_to_dataset_true_requires_uid_to_key(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        evaluate_segmentation_as_patches(
            _FakeTask(), _make_loader(), torch.device("cpu"), _NUM_CLASSES,
            tmp_path, "test", use_wandb=False,
            uid_to_label={0: 0, 1: 0}, uid_to_key=None, save_probs=False,
            restrict_to_dataset=True,
        )
