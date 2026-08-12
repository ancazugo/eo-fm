"""GAP + MLP classification family.

Arch spec string: ``"mlp_<h1>-<h2>-..."`` — hyphen-separated hidden sizes.
``"mlp_"`` (no hidden layers) is a linear probe equivalent.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.pooling import pool_mean_std
from models.registry import ModelFamily, register

MLP_PRESETS: dict[str, str] = {
    "nano":   "mlp_",            # no hidden layers — linear probe equivalent
    "small":  "mlp_256",
    "base":   "mlp_512-256",
    "medium": "mlp_512-256-128",
    "large":  "mlp_1024-512-256",
}


class MLPModel(nn.Module):
    """GAP + configurable MLP classifier. Accepts (B, C, H, W) or (B, C).

    If input is 4-D, global average pool collapses spatial dims first.
    Empty hidden_sizes produces a linear probe equivalent (single FC layer).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_sizes: list[int],
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_channels
        for h in hidden_sizes:
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(prev, num_classes))
        self.mlp = nn.Sequential(*layers)

    accepts_valid_mask = True

    def forward(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() == 4:
            x, _ = pool_mean_std(x, valid)   # GAP: (B, C, H, W) → (B, C)
        return self.mlp(x)


def build_mlp(
    arch: str,
    in_channels: int,
    num_classes: int,
    head_dropout: float = 0.0,
) -> nn.Module:
    """Construct an MLPModel from an arch string like ``'mlp_512-256'``."""
    suffix = arch[len("mlp_"):]                          # "" | "256" | "512-256-128"
    hidden_sizes = [int(s) for s in suffix.split("-") if s]
    return MLPModel(
        in_channels=in_channels,
        hidden_sizes=hidden_sizes,
        num_classes=num_classes,
        dropout=head_dropout,
    )


register(ModelFamily(
    name="mlp",
    pipeline="classification",
    presets=MLP_PRESETS,
    build=build_mlp,
))
