"""Light ASPP (Atrous Spatial Pyramid Pooling) classification family.

Multi-dilation context aggregation over the embedding patch, followed by GAP
and a linear classifier. Presets control the per-path channel width.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import ModelFamily, register

ASPP_PRESETS: dict[str, str] = {
    "nano":   "aspp_nano",
    "small":  "aspp_small",
    "base":   "aspp_base",
    "medium": "aspp_medium",
    "large":  "aspp_large",
}

_ASPP_OUT_CHANNELS: dict[str, int] = {
    "nano":   32,
    "small":  64,
    "base":   128,
    "medium": 192,
    "large":  256,
}


class LightASPPHead(nn.Module):
    def __init__(self, in_channels=128, out_channels=64, num_classes=17):
        super().__init__()

        # 1x1 Conv path
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        # 3x3 Conv with dilation=2 (Small neighborhood context)
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        # 3x3 Conv with dilation=4 (Broader urban context)
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # Global context path
        self.global_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # Final linear projection
        # 4 paths concatenated = out_channels * 4
        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.classifier = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        size = x.shape[-2:]

        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)

        # Global pooling path requires upsampling back to match spatial size
        x4 = self.global_avg(x)
        x4 = F.interpolate(x4, size=size, mode='bilinear', align_corners=False)

        # Concatenate all scale features
        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.bottleneck(out)

        # Collapse remaining spatial structure to classify
        out = F.adaptive_avg_pool2d(out, (1, 1)).flatten(1)
        return self.classifier(out)


def build_aspp(arch: str, in_channels: int, num_classes: int) -> nn.Module:
    out_channels = _ASPP_OUT_CHANNELS[arch.replace("aspp_", "")]
    return LightASPPHead(in_channels=in_channels, out_channels=out_channels, num_classes=num_classes)


register(ModelFamily(
    name="aspp",
    pipeline="classification",
    presets=ASPP_PRESETS,
    build=build_aspp,
))
