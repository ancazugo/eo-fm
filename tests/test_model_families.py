"""Every registered family must build and return the shape its pipeline promises.

Offline: random tensors, no data mounts, no pretrained weights.

`ModelFamily.pipeline` is a contract the training loop trusts blindly --
classification wrappers feed (B, num_classes) into CrossEntropyLoss, segmentation
wrappers index masks with the logits' own (H, W). A family that silently returns
the wrong rank, or the right rank at the wrong resolution, only fails deep inside
a training run. The odd-size case matters because segmentation batches are padded
to the largest square in the batch, so edge tiles arrive one pixel short.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import build_model, families_for, get_family  # noqa: E402

IN_CHANNELS = 128   # Tessera
NUM_CLASSES = 17    # LCZ 1-17 → 0-16
PRESET_NAMES = {"nano", "small", "base", "medium", "large"}


@pytest.mark.parametrize("family", families_for("classification"))
def test_classification_output_shape(family):
    # A superset of kwargs, exactly as patch_classification.py passes them:
    # build_model drops the ones a given builder does not declare. img_size is
    # what keeps vit valid at 32 px.
    model = build_model(
        family,
        "nano",
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        head_dropout=0.0,
        img_size=32,
    ).eval()

    with torch.no_grad():
        out = model(torch.randn(2, IN_CHANNELS, 32, 32))

    assert out.shape == (2, NUM_CLASSES)


@pytest.mark.parametrize("family", families_for("segmentation"))
@pytest.mark.parametrize("size", [64, 63])
def test_segmentation_output_shape(family, size):
    model = build_model(
        family,
        "nano",
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        bottleneck_dropout=0.3,
    ).eval()

    with torch.no_grad():
        out = model(torch.randn(2, IN_CHANNELS, size, size))

    assert out.shape == (2, NUM_CLASSES, size, size)


@pytest.mark.parametrize("family", ["shallow_cnn", "fcn8"])
def test_new_families_define_every_preset(family):
    # utils.cli hardcodes the five preset names and defaults to "large", so a
    # family missing one is unusable from the CLI without --preset.
    assert set(get_family(family).presets) == PRESET_NAMES


def test_shallow_cnn_arch_override():
    # patch_classification.py --arch bypasses the presets entirely.
    model = build_model(
        "shallow_cnn", None, "scnn_8-16-32", in_channels=64, num_classes=NUM_CLASSES
    ).eval()

    with torch.no_grad():
        out = model(torch.randn(2, 64, 32, 32))

    assert out.shape == (2, NUM_CLASSES)
