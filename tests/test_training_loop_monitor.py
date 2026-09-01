"""The monitored metric must not fail silently when it goes NaN.

Offline: a tiny module and two-batch loaders, no data mounts.

`run_training_loop` maximises `task_module.monitor` starting from -inf. NaN
compares greater than nothing, so a monitor that goes NaN checkpoints nothing,
`best_ckpt_path` stays None, and the caller carries on and evaluates a model
that was never selected -- silently. This bit during the segmentation work when
`--monitor val_kappa` met a degenerate val split: MulticlassCohenKappa returns
NaN when the val pixels hold a single class, because expected agreement is 1
and the chance correction divides by zero.

The point of the test is not that NaN is handled gracefully -- it is that the
run says so out loud, because the failure otherwise looks like a completed run
with plausible test numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.loop import run_training_loop  # noqa: E402


class _Tiny(nn.Module):
    def __init__(self, monitor_values):
        super().__init__()
        self.model = nn.Linear(4, 2)
        self.monitor = "val_metric"
        self.lr, self.weight_decay = 1e-3, 0.0
        self._values = list(monitor_values)
        self._epoch = 0

    def reset_train_metrics(self):
        pass

    def train_step(self, batch, device):
        return self.model(batch["image"]).sum()

    def compute_train_logs(self):
        return {"train_loss": 0.0}

    def reset_val_metrics(self):
        pass

    def val_step(self, batch, device):
        pass

    def compute_val_logs(self):
        v = self._values[min(self._epoch, len(self._values) - 1)]
        self._epoch += 1
        return {"val_metric": v, "val_acc": 0.5}


class _DM:
    def setup(self):
        pass

    def _loader(self):
        return [{"image": torch.randn(2, 4)}]

    train_dataloader = _loader
    val_dataloader = _loader


def _run(tmp_path, values, epochs=2):
    task = _Tiny(values)
    return run_training_loop(
        task_module=task, datamodule=_DM(), device=torch.device("cpu"),
        max_epochs=epochs, early_stopping_patience=99,
        run_dir=tmp_path, model_name="tiny",
    )


def test_a_nan_monitor_is_reported_and_leaves_no_checkpoint(tmp_path):
    # loguru does not route through pytest's caplog, so capture it directly.
    from loguru import logger

    seen: list[str] = []
    sink = logger.add(seen.append, level="WARNING")
    try:
        _task, ckpt = _run(tmp_path, [float("nan"), float("nan")])
    finally:
        logger.remove(sink)

    assert ckpt is None
    text = " ".join(seen).lower()
    assert "nan" in text
    assert "no checkpoint was ever saved" in text
    assert "final weights" in text


def test_a_healthy_monitor_still_checkpoints_normally(tmp_path):
    _task, ckpt = _run(tmp_path, [0.1, 0.4])
    assert ckpt is not None and ckpt.exists()
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert saved["val_metric"] == 0.4 and saved["epoch"] == 2


def test_one_good_epoch_survives_a_later_nan(tmp_path):
    """The NaN must not overwrite or invalidate a real best."""
    _task, ckpt = _run(tmp_path, [0.3, float("nan")])
    assert ckpt is not None and ckpt.exists()
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert saved["val_metric"] == 0.3 and saved["epoch"] == 1
