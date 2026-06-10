"""Linear probe classification family: pooled features → BatchNorm → Linear.

The arch payload is the pooling mode: ``"gap"`` (global average, D = C) or
``"mean_std"`` (mean+std concat, D = 2C). All presets default to ``"gap"``;
pass ``--arch mean_std`` to override.

Feature normalisation is a ``BatchNorm1d(affine=False)`` layer (the "BN +
linear" probe from the MAE paper): it tracks training-set mean/var as running
buffers, so the checkpoint is self-contained — unlike the old standalone
linear_probe.py, no external ``*_stats.npz`` file is needed at inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.registry import ModelFamily, register

LINEAR_PROBE_PRESETS: dict[str, str] = {
    "nano":   "gap",
    "small":  "gap",
    "base":   "gap",
    "medium": "gap",
    "large":  "gap",
}


class LinearProbeModel(nn.Module):
    """Pooling + BatchNorm1d(affine=False) + Linear. Accepts (B, C, H, W) or (B, D).

    If input is 4-D, spatial pooling is applied first ("gap" = channel means;
    "mean_std" = channel means + stds concatenated). 2-D input is treated as
    pre-pooled features.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        pooling: str = "gap",
    ) -> None:
        super().__init__()
        if pooling not in ("gap", "mean_std"):
            raise ValueError(f"pooling must be 'gap' or 'mean_std', got {pooling!r}")
        self.pooling = pooling
        feature_dim = in_channels * (2 if pooling == "mean_std" else 1)
        self.norm = nn.BatchNorm1d(feature_dim, affine=False)
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            mean = x.mean(dim=(-2, -1))
            if self.pooling == "gap":
                x = mean
            else:
                # unbiased=False matches numpy's default ddof=0 used previously
                x = torch.cat([mean, x.std(dim=(-2, -1), unbiased=False)], dim=1)
        return self.fc(self.norm(x))


def load_legacy_linear_probe(
    state: dict,
    stats_file,
    in_channels: int,
    num_classes: int,
) -> LinearProbeModel:
    """Convert a legacy linear-probe checkpoint into a LinearProbeModel.

    The retired standalone linear_probe.py saved only ``fc.weight``/``fc.bias``
    and z-scored features with externally stored train-set stats
    (``(x - mean) / (std + 1e-8)``). This writes those stats into the
    BatchNorm running buffers (eps=0, var=(std+1e-8)²) so the converted model
    is numerically identical in eval mode — and runs through the standard
    classification inference path with no stats file beyond this load.

    Pooling is inferred from the checkpoint's feature dimension
    (D == C → gap, D == 2C → mean_std).
    """
    import numpy as np

    feature_dim = state["fc.weight"].shape[1]
    if feature_dim == in_channels:
        pooling = "gap"
    elif feature_dim == 2 * in_channels:
        pooling = "mean_std"
    else:
        raise ValueError(
            f"Checkpoint feature_dim={feature_dim} matches neither gap ({in_channels}) "
            f"nor mean_std ({2 * in_channels}) for in_channels={in_channels}"
        )

    stats = np.load(stats_file)
    mean = torch.from_numpy(stats["mean"]).float()
    std = torch.from_numpy(stats["std"]).float()
    if mean.numel() != feature_dim:
        raise ValueError(
            f"Stats dim {mean.numel()} does not match checkpoint feature_dim {feature_dim}"
        )

    model = LinearProbeModel(in_channels, num_classes, pooling=pooling)
    # BatchNorm divides by sqrt(var + eps); choose var so that equals std + 1e-8
    # exactly (torch requires eps > 0, so fold it into var and clamp at 0).
    model.norm.eps = 1e-12
    with torch.no_grad():
        model.norm.running_mean.copy_(mean)
        model.norm.running_var.copy_(((std + 1e-8) ** 2 - model.norm.eps).clamp(min=0.0))
        model.fc.weight.copy_(state["fc.weight"])
        model.fc.bias.copy_(state["fc.bias"])
    return model


def build_linear_probe(
    arch: str,
    in_channels: int,
    num_classes: int,
) -> nn.Module:
    """``arch`` is the pooling mode: 'gap' or 'mean_std'."""
    return LinearProbeModel(
        in_channels=in_channels,
        num_classes=num_classes,
        pooling=arch,
    )


register(ModelFamily(
    name="linear_probe",
    pipeline="classification",
    presets=LINEAR_PROBE_PRESETS,
    build=build_linear_probe,
))
