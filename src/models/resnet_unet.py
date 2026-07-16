"""ResNet-backbone U-Net segmentation family.

Preset payloads are timm ResNet backbone names (nano→resnet18, small→resnet34,
base→resnet50).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from models.registry import ModelFamily, register
from models.unet import DoubleConv

RESNET_UNET_PRESETS: dict[str, str] = {
    "nano":  "resnet18",
    "small": "resnet34",
    "base":  "resnet50",
}

_SUPPORTED_BACKBONES = ("resnet18", "resnet34", "resnet50")


class ResNetUNet(nn.Module):
    """U-Net with a timm ResNet encoder.

    The ResNet stem is adapted for small spatial inputs by replacing the
    standard 7×7 stride-2 conv and stride-2 maxpool with a 3×3 stride-1 conv.
    This reduces the encoder total stride from 32 to 8, making the architecture
    suitable for patches as small as 32×32 px.

    Skip connections are taken from the outputs of ResNet layer1–4.  The decoder
    mirrors the standard U-Net pattern: ConvTranspose2d upsample → concat skip
    → DoubleConv, repeated three times (layer4→layer3→layer2→layer1), ending at
    the full input resolution.

    Args:
        in_channels: Embedding input channels (e.g. 128 for Tessera, 64 for AlphaEarth).
        num_classes: Number of segmentation output classes.
        backbone: timm model name; one of ``_SUPPORTED_BACKBONES``.
        bottleneck_dropout: Dropout2d probability applied to the deepest encoder
            features before decoding (0 = off).
        pretrained: Whether to load ImageNet-pretrained weights.  Only meaningful
            when ``in_channels == 3``; ignored (set to False) otherwise since
            embedding inputs differ fundamentally from RGB.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        backbone: str = "resnet18",
        bottleneck_dropout: float = 0.3,
        pretrained: bool = False,
    ) -> None:
        import timm

        super().__init__()

        if backbone not in _SUPPORTED_BACKBONES:
            raise ValueError(
                f"backbone='{backbone}' not supported. "
                f"Choose from: {list(_SUPPORTED_BACKBONES)}"
            )

        # Pretrained weights are only sensible for 3-channel RGB input.
        use_pretrained = pretrained and (in_channels == 3)
        if pretrained and not use_pretrained:
            logger.warning(
                f"pretrained=True ignored for in_channels={in_channels} "
                "(embedding inputs are not RGB — ImageNet weights are irrelevant)."
            )

        # ── Encoder (timm ResNet, feature extraction at each layer group) ────
        # timm feature index 0 is the stem (act1); layer1-4 are indices 1-4.
        self.encoder = timm.create_model(
            backbone,
            pretrained=use_pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=(1, 2, 3, 4),  # after layer1, layer2, layer3, layer4
        )

        # Adapt stem for small spatial inputs.
        # Original stem: Conv2d(in_ch, 64, 7×7, stride=2) + MaxPool(stride=2) → stride 4.
        # Modified stem: Conv2d(in_ch, 64, 3×3, stride=1) + Identity → stride 1.
        # After this change the four encoder feature strides become ≈ [1, 2, 4, 8]
        # instead of [4, 8, 16, 32], so a 64-px patch yields an 8-px bottleneck.
        stem_out_ch = self.encoder.conv1.out_channels
        self.encoder.conv1 = nn.Conv2d(
            in_channels, stem_out_ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.encoder.maxpool = nn.Identity()

        enc_ch = self.encoder.feature_info.channels()  # [c1, c2, c3, c4] deepest last

        # ── Bottleneck dropout ───────────────────────────────────────────────
        self.bottleneck_drop = (
            nn.Dropout2d(bottleneck_dropout) if bottleneck_dropout > 0 else nn.Identity()
        )

        # ── Decoder: one up-block per skip connection (layer3 → layer2 → layer1) ─
        # enc_ch[-1] is the bottleneck; enc_ch[:-1] are the skip sources (reversed).
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = enc_ch[-1]
        for skip_ch in reversed(enc_ch[:-1]):
            self.upsamples.append(nn.ConvTranspose2d(ch, skip_ch, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(skip_ch * 2, skip_ch))
            ch = skip_ch

        self.head = nn.Conv2d(ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)  # [layer1, layer2, layer3, layer4] finest→coarsest

        bottleneck = self.bottleneck_drop(features[-1])
        skips = features[:-1]  # [layer1, layer2, layer3]

        out = bottleneck
        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            out = up(out)
            if out.shape[-2:] != skip.shape[-2:]:
                out = F.interpolate(out, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            out = torch.cat([skip, out], dim=1)
            out = dec(out)

        return self.head(out)


def build_resnet_unet(
    arch: str,
    in_channels: int,
    num_classes: int,
    bottleneck_dropout: float = 0.3,
    pretrained: bool = False,
) -> nn.Module:
    """``arch`` is the timm ResNet backbone name (e.g. 'resnet50')."""
    return ResNetUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        backbone=arch,
        bottleneck_dropout=bottleneck_dropout,
        pretrained=pretrained,
    )


register(ModelFamily(
    name="resnet_unet",
    pipeline="segmentation",
    presets=RESNET_UNET_PRESETS,
    build=build_resnet_unet,
))
