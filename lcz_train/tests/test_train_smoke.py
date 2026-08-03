"""T5 training smoke test — A1, a few steps, synthetic embeddings.

Fully offline: a trivially-learnable synthetic problem (embedding channel 0's
sign perfectly predicts hard-3 vs hard-6) so a linear probe reliably drives
the loss down in a handful of steps without flaking.
"""

import numpy as np
import torch

from lcz_train.config import TrainConfig
from lcz_train.datasets import PixelWindowDataset
from lcz_train.models import A1Linear
from lcz_train.train import dense_forward_and_loss, seed_all, train_loop


def _trivially_learnable_rasters(h=200, w=200, c=8, seed=0):
    rng = np.random.default_rng(seed)
    mid_h, mid_w = h // 2, w // 2
    bitmask = np.zeros((h, w), dtype=np.uint32)
    bitmask[:mid_h, :mid_w] = 1 << 2      # hard 3, left
    bitmask[:mid_h, mid_w:] = 1 << 5      # hard 6, right
    conf = np.where(bitmask != 0, 90, 0).astype(np.uint8)
    block_idx = np.zeros((h, w), dtype=np.uint32)
    block_idx[:mid_h, :mid_w] = 1
    block_idx[:mid_h, mid_w:] = 2
    block_idx[mid_h:, :] = 3

    mosaic = 0.05 * rng.normal(size=(c, h, w)).astype(np.float32)
    mosaic[0, :, :mid_w] += 1.0    # channel 0: +1 on the hard-3 side
    mosaic[0, :, mid_w:] -= 1.0    # -1 on the hard-6 side
    return mosaic, bitmask, conf, block_idx


def test_seed_all_is_reproducible():
    seed_all(123)
    a = torch.randn(4)
    seed_all(123)
    b = torch.randn(4)
    torch.testing.assert_close(a, b)


def test_a1_training_smoke_loss_decreases_and_checkpoint_written(tmp_path):
    mosaic, bitmask, conf, block_idx = _trivially_learnable_rasters()
    dataset = PixelWindowDataset(mosaic, bitmask, conf, block_idx, window_px=96,
                                 erosion_px=1, seed=0)

    cfg = TrainConfig(exp_id="A1", steps=60, batch_size=8, lr=5e-2, log_every=10, seed=0)
    model = A1Linear(in_channels=mosaic.shape[0])
    ckpt = tmp_path / "a1-best.pt"

    result = train_loop(model, dataset, dense_forward_and_loss, cfg, ckpt_path=ckpt)

    assert result["final_loss"] < result["first_loss"]
    assert result["final_loss"] < 0.5 * result["first_loss"]   # substantial, not noise
    assert np.isfinite(result["losses"]).all()

    assert ckpt.exists()
    saved = torch.load(ckpt, weights_only=False)
    assert set(saved) == {"model_state_dict", "step", "config_hash", "metrics"}
    assert saved["step"] == 60
    assert saved["config_hash"] == cfg.config_hash

    # the checkpoint actually reloads into a fresh model
    reloaded = A1Linear(in_channels=mosaic.shape[0])
    reloaded.load_state_dict(saved["model_state_dict"])


def test_train_loop_runs_deterministically_given_seed():
    # Correct usage: seed before constructing the model — its random init
    # draws from the ambient torch RNG at construction time, which train_loop
    # cannot retroactively fix by reseeding only once training starts.
    mosaic, bitmask, conf, block_idx = _trivially_learnable_rasters(seed=1)
    cfg = TrainConfig(exp_id="A1", steps=10, batch_size=4, lr=1e-2, log_every=5, seed=7)

    def _run():
        seed_all(cfg.seed)
        ds = PixelWindowDataset(mosaic, bitmask, conf, block_idx, window_px=96,
                                erosion_px=1, seed=0)
        model = A1Linear(in_channels=mosaic.shape[0])
        return train_loop(model, ds, dense_forward_and_loss, cfg)

    r1, r2 = _run(), _run()
    np.testing.assert_allclose(r1["losses"], r2["losses"])
