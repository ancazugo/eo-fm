"""Dihedral test-time augmentation must map dense outputs back to the input frame.

A 1x1 convolution is exactly equivariant to rotations and flips, so averaging
its predictions over the dihedral group has to return its plain prediction.
The previous segmentation TTA summed the rotated/flipped output maps without
inverting the transforms, which mixed predictions for eight different ground
locations into every pixel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.evaluate import dihedral_tta  # noqa: E402


def test_spatial_tta_of_a_pointwise_model_is_the_identity():
    torch.manual_seed(0)
    model = torch.nn.Conv2d(5, 3, kernel_size=1)
    x = torch.randn(2, 5, 8, 8)
    with torch.no_grad():
        assert torch.allclose(dihedral_tta(model, x, spatial=True), model(x), atol=1e-6)


def test_spatial_tta_output_is_aligned_with_a_non_symmetric_input():
    """Pin the frame: a single hot pixel must stay where it was."""
    model = torch.nn.Identity()
    x = torch.zeros(1, 1, 6, 6)
    x[0, 0, 1, 4] = 1.0
    out = dihedral_tta(model, x, spatial=True)
    assert out[0, 0, 1, 4] == 1.0 and out.sum() == 1.0


def test_classification_tta_averages_frame_free_logits():
    """Global-pooled logits are invariant, so the average equals the plain output."""
    model = torch.nn.Sequential(torch.nn.Conv2d(4, 3, 1),
                                torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
    x = torch.randn(3, 4, 8, 8)
    with torch.no_grad():
        assert torch.allclose(dihedral_tta(model, x), model(x), atol=1e-6)
