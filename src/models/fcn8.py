"""Miniature FCN-8 segmentation family.

A fully-convolutional network in the original sense: a plain CNN encoder whose
dense layers are replaced by convolutions, with per-stage 1x1 "score" heads
fused coarse-to-fine and upsampled back to the input resolution.

Compared with the standard FCN-8s (VGG, pool3/4/5 at strides 8/16/32) this is a
miniature built for embedding tiles: 2-3 pooling stages, so the deepest stride
is 4 or 8 rather than 32. Preset payloads are ``(depth, base_features)`` tuples,
matching :data:`models.unet.UNET_PRESETS`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import ModelFamily, register
from models.unet import DoubleConv

FCN8_PRESETS: dict[str, tuple[int, int]] = {
    "nano":   (2,  8),
    "small":  (3, 16),
    "base":   (3, 32),
    "medium": (3, 48),
    "large":  (3, 64),
}


class FCN8(nn.Module):
    """Fully-convolutional network with coarse-to-fine score fusion.

    Encoder: DoubleConv → MaxPool2d (×depth); the pooled output of every stage
    is a fusion source (strides 2, 4, ..., 2**depth).
    Classifier: a 3×3 conv block standing in for the VGG fc6/fc7 dense layers.
    Fusion: 1×1 score conv per stage; the coarsest score is upsampled ×2 with a
    ConvTranspose2d and *added* to the next finer score, repeated down the
    pyramid, then resized to the input resolution.

    ``depth`` is capped at 3 by the presets: FCN-8s fuses three scales, and
    deeper pyramids lose too much detail on 128-px grid tiles. The larger
    presets widen instead.

    Built-in presets (depth, base_features):
        nano:   (2,  8)  — 2 stages (strides 2/4), minimal params
        small:  (3, 16)  — 3 stages (strides 2/4/8), lightweight
        base:   (3, 32)  — baseline width
        medium: (3, 48)  — wider
        large:  (3, 64)  — widest recommended preset

    Args:
        in_channels: Number of embedding input channels.
        num_classes: Number of segmentation output classes.
        depth: Number of encoder stages (= number of fused scales).
        base_features: Feature maps at the first encoder stage;
            doubles at each subsequent stage.
        bottleneck_dropout: Dropout2d probability in the classifier block.
    """

    PRESETS = FCN8_PRESETS

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        depth: int = 3,
        base_features: int = 32,
        bottleneck_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.depth = depth

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        stage_channels: list[int] = []
        ch = in_channels
        for i in range(depth):
            out_ch = base_features * (2 ** i)
            self.encoders.append(DoubleConv(ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            stage_channels.append(out_ch)
            ch = out_ch

        # ── Classifier (the "dense layers as convolutions" of FCN) ───────────
        layers: list[nn.Module] = [
            nn.Conv2d(ch, ch, 3, padding=1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        ]
        if bottleneck_dropout > 0.0:
            layers.append(nn.Dropout2d(bottleneck_dropout))
        self.classifier = nn.Sequential(*layers)

        # ── Score heads, one per fused scale (coarsest last) ─────────────────
        self.scores = nn.ModuleList(
            nn.Conv2d(c, num_classes, kernel_size=1) for c in stage_channels
        )

        # ── ×2 upsamples between consecutive scores ──────────────────────────
        self.upsamples = nn.ModuleList(
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2)
            for _ in range(depth - 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        pooled: list[torch.Tensor] = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            pooled.append(pool(x))

        # Coarsest stage goes through the conv classifier before scoring.
        out = self.scores[-1](self.classifier(pooled[-1]))

        # Fuse coarse → fine: upsample ×2, add the finer stage's score.
        for up, score, feat in zip(
            self.upsamples, reversed(self.scores[:-1]), reversed(pooled[:-1])
        ):
            out = up(out)
            skip = score(feat)
            # Correct for odd-sized inputs (bilinear resize if needed)
            if out.shape[-2:] != skip.shape[-2:]:
                out = F.interpolate(out, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            out = out + skip

        return F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)


def build_fcn8(
    arch: tuple[int, int],
    in_channels: int,
    num_classes: int,
    bottleneck_dropout: float = 0.3,
) -> nn.Module:
    """``arch`` is the ``(depth, base_features)`` preset payload."""
    depth, base_features = arch
    return FCN8(
        in_channels=in_channels,
        num_classes=num_classes,
        depth=depth,
        base_features=base_features,
        bottleneck_dropout=bottleneck_dropout,
    )


register(ModelFamily(
    name="fcn8",
    pipeline="segmentation",
    presets=FCN8_PRESETS,
    build=build_fcn8,
))
