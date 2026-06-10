"""U-Net segmentation family.

Preset payloads are ``(depth, base_features)`` tuples.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import ModelFamily, register

UNET_PRESETS: dict[str, tuple[int, int]] = {
    "nano":   (2,  8),
    "small":  (3, 32),
    "base":   (3, 48),
    "medium": (4, 32),
    "large":  (4, 48),
}


class DoubleConv(nn.Module):
    """Two consecutive Conv2d(3×3) → BatchNorm → ReLU blocks.

    Args:
        in_ch: Number of input channels.
        out_ch: Number of output channels.
        dropout: Dropout2d probability applied after the second ReLU (0 = off).
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """U-Net with configurable depth and feature width.

    Encoder: DoubleConv → MaxPool2d (×depth).
    Bottleneck: DoubleConv (with dropout).
    Decoder: ConvTranspose2d → skip-cat → DoubleConv (×depth).
    Head: Conv2d(1×1) → num_classes logits.

    Built-in presets (depth, base_features):
        nano:   (2,  8)  — fast experiments, minimal params
        small:  (3, 32)  — lightweight baseline
        base:   (3, 48)  — wider baseline (mirrors tessera-cnn-example "base")
        medium: (4, 32)  — deeper with moderate width
        large:  (4, 48)  — deepest + widest recommended preset

    Args:
        in_channels: Number of embedding input channels.
        num_classes: Number of segmentation output classes.
        depth: Number of encoder/decoder stages.
        base_features: Feature maps at the first encoder stage;
            doubles at each subsequent stage.
        bottleneck_dropout: Dropout2d probability at the bottleneck.
    """

    PRESETS = UNET_PRESETS  # backward-compat alias (UNet.PRESETS)

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        depth: int = 3,
        base_features: int = 32,
        bottleneck_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.depth = depth

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        enc_channels: list[int] = []
        ch = in_channels
        for i in range(depth):
            out_ch = base_features * (2 ** i)
            self.encoders.append(DoubleConv(ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            enc_channels.append(out_ch)
            ch = out_ch

        # ── Bottleneck ───────────────────────────────────────────────────────
        bottleneck_ch = base_features * (2 ** depth)
        self.bottleneck = DoubleConv(ch, bottleneck_ch, dropout=bottleneck_dropout)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = bottleneck_ch
        for i in reversed(range(depth)):
            skip_ch = enc_channels[i]
            self.upsamples.append(nn.ConvTranspose2d(ch, skip_ch, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(skip_ch * 2, skip_ch))
            ch = skip_ch

        self.head = nn.Conv2d(ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []

        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = up(x)
            # Correct for odd-sized inputs (bilinear resize if needed)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.head(x)


def build_unet(
    arch: tuple[int, int],
    in_channels: int,
    num_classes: int,
    bottleneck_dropout: float = 0.3,
) -> nn.Module:
    """``arch`` is the ``(depth, base_features)`` preset payload."""
    depth, base_features = arch
    return UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        depth=depth,
        base_features=base_features,
        bottleneck_dropout=bottleneck_dropout,
    )


register(ModelFamily(
    name="unet",
    pipeline="segmentation",
    presets=UNET_PRESETS,
    build=build_unet,
))
