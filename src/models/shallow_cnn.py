"""Shallow CNN classification family.

Two convolutional blocks (Conv3x3 → BatchNorm → ReLU → MaxPool), global average
pooling, and a linear classifier — the minimal architecture that still models
spatial structure, sitting between the pooling probes (``linear_probe``, ``mlp``,
no spatial modelling) and the deep ImageNet backbones.

Arch spec string: ``"scnn_<c1>-<c2>-..."`` — hyphen-separated channel widths,
one conv block per entry.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.registry import ModelFamily, register

SHALLOW_CNN_PRESETS: dict[str, str] = {
    "nano":   "scnn_16-32",
    "small":  "scnn_32-64",
    "base":   "scnn_64-128",
    "medium": "scnn_96-192",
    "large":  "scnn_128-256",
}


class ShallowCNN(nn.Module):
    """Conv blocks + GAP + Linear classifier. Accepts (B, C, H, W).

    Each block halves the spatial resolution, so the default two blocks take a
    32×32 embedding patch to 8×8 before pooling. Global average pooling makes
    the model input-size agnostic, and ``ceil_mode=True`` keeps the pooling
    valid for small patches or long channel specs.

    Args:
        in_channels: Number of embedding input channels.
        channels: Output width of each conv block; its length is the number of
            blocks (2 for every built-in preset).
        num_classes: Number of output classes.
        dropout: Dropout probability before the final Linear (0 = off).
    """

    def __init__(
        self,
        in_channels: int,
        channels: list[int],
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("channels must list at least one conv block")

        layers: list[nn.Module] = []
        prev = in_channels
        for c in channels:
            layers += [
                nn.Conv2d(prev, c, 3, padding=1, padding_mode="reflect", bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, ceil_mode=True),
            ]
            prev = c
        self.features = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d(1)
        head: list[nn.Module] = []
        if dropout > 0.0:
            head.append(nn.Dropout(p=dropout))
        head.append(nn.Linear(prev, num_classes))
        self.head = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)     # (B, C, H, W) → (B, C)
        return self.head(x)


def build_shallow_cnn(
    arch: str,
    in_channels: int,
    num_classes: int,
    head_dropout: float = 0.0,
) -> nn.Module:
    """Construct a ShallowCNN from an arch string like ``'scnn_32-64'``."""
    suffix = arch[len("scnn_"):]                         # "16-32" | "32-64-128"
    channels = [int(s) for s in suffix.split("-") if s]
    return ShallowCNN(
        in_channels=in_channels,
        channels=channels,
        num_classes=num_classes,
        dropout=head_dropout,
    )


register(ModelFamily(
    name="shallow_cnn",
    pipeline="classification",
    presets=SHALLOW_CNN_PRESETS,
    build=build_shallow_cnn,
))
