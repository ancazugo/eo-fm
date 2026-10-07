"""infer_roi must rebuild a classifier at the size it was trained at.

The timm stem adaptation relaxes strides/pooling until the final map is >= 4x4
*for the given input size*. A model trained at 32 px and rebuilt at 64 px
therefore has different strides or pooling but identical parameter shapes, so
load_state_dict succeeds on a network that is not the one that was trained.
infer_roi used to default --patch-size to 64 for every family.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from infer_roi import default_patch_size, load_model_and_normalize  # noqa: E402
from models import build_model  # noqa: E402


def _ckpt(tmp_path, patch_size=None):
    model = build_model("mobilenet", "nano", in_channels=128, num_classes=17, img_size=32)
    payload = {"model_state_dict": model.state_dict(), "normalize": "none"}
    if patch_size is not None:
        payload["patch_size"] = patch_size
    path = tmp_path / "m.pt"
    torch.save(payload, path)
    return path


def test_a_rebuild_at_another_size_loads_but_differs():
    """The trap itself: same parameters, different strides."""
    a = build_model("mobilenet", "nano", in_channels=128, num_classes=17, img_size=32)
    b = build_model("mobilenet", "nano", in_channels=128, num_classes=17, img_size=64)
    b.load_state_dict(a.state_dict())          # no error ...
    strides = lambda m: [x.stride for x in m.modules() if isinstance(x, torch.nn.Conv2d)]  # noqa: E731
    assert strides(a) != strides(b)            # ... yet a different network


def test_default_patch_size_reads_the_checkpoint(tmp_path):
    assert default_patch_size(_ckpt(tmp_path, 48), "mobilenet") == 48
    assert default_patch_size(_ckpt(tmp_path), "mobilenet") == 32   # legacy default
    assert default_patch_size(_ckpt(tmp_path, 48), "unet") == 64     # seg: window size


def test_a_conflicting_patch_size_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="trained at patch size 32"):
        load_model_and_normalize(_ckpt(tmp_path, 32), "mobilenet", "tesserav1.1_global",
                                 torch.device("cpu"), preset="nano", patch_size=64)


def test_the_matching_patch_size_loads(tmp_path):
    model, norm = load_model_and_normalize(
        _ckpt(tmp_path, 32), "mobilenet", "tesserav1.1_global",
        torch.device("cpu"), preset="nano", patch_size=32)
    assert norm is None and model(torch.zeros(1, 128, 32, 32)).shape == (1, 17)
