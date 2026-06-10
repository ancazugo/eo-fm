"""timm-backed classification families: resnet, efficientnet, convnext,
densenet, mobilenet, vit.

All families share one builder (:func:`build_timm`). Family-specific input
adaptations for small embedding patches (32×32 px):

- resnet: stem surgery — 7×7 stride-2 conv + maxpool replaced by a 3×3
  stride-1 conv + Identity, so feature maps stay large.
- vit: patch_size overridden to 2 → (img_size / 2)² tokens (e.g. 256 tokens
  for a 32×32 input).
"""

from __future__ import annotations

import torch.nn as nn

from models.registry import ModelFamily, register

TIMM_PRESETS: dict[str, dict[str, str]] = {
    "resnet": {
        "nano":   "resnet18",
        "small":  "resnet34",
        "base":   "resnet50",
        "medium": "resnet101",
        "large":  "resnet152",
    },
    "efficientnet": {
        "nano":   "efficientnet_b0",
        "small":  "efficientnet_b1",
        "base":   "efficientnet_b3",
        "medium": "efficientnet_b5",
        "large":  "efficientnet_b7",
    },
    "convnext": {
        "nano":   "convnext_nano",
        "small":  "convnext_tiny",
        "base":   "convnext_small",
        "medium": "convnext_base",
        "large":  "convnext_large",
    },
    "densenet": {
        "nano":   "densenet121",
        "small":  "densenet161",
        "base":   "densenet169",
        "medium": "densenet201",
        "large":  "densenet264d",
    },
    "mobilenet": {
        "nano":   "mobilenetv3_small_050",
        "small":  "mobilenetv3_small_100",
        "base":   "mobilenetv3_large_100",
        "medium": "mobilenetv3_large_150d",
        "large":  "mobilenetv4_conv_large",
    },
    "vit": {
        "nano":   "vit_tiny_patch16_224",
        "small":  "vit_small_patch16_224",
        "base":   "vit_small_patch8_224",
        "medium": "vit_base_patch16_224",
        "large":  "vit_base_patch8_224",
    },
}


def build_timm(
    arch: str,
    in_channels: int,
    num_classes: int,
    family: str = "resnet",
    head_dropout: float = 0.0,
    img_size: int | None = None,
) -> nn.Module:
    """Build a timm classification model with embedding-input adaptations."""
    import timm

    kwargs: dict = dict(
        in_chans=in_channels,
        num_classes=num_classes,
        pretrained=False,
        drop_rate=head_dropout,
    )

    if family == "vit":
        # img_size is required for ViT (timm CNNs reject the kwarg);
        # patch_size=2 → (img_size / 2)² spatial tokens (256 for 32×32 input)
        if img_size is not None:
            kwargs["img_size"] = img_size
        return timm.create_model(arch, **kwargs, patch_size=2)

    model = timm.create_model(arch, **kwargs)

    if family == "resnet":
        out_ch = model.conv1.out_channels  # preserve original width (64 for all resnet variants)
        model.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        model.maxpool = nn.Identity()

    return model


def _make_build(family_name: str):
    def build(
        arch: str,
        in_channels: int,
        num_classes: int,
        head_dropout: float = 0.0,
        img_size: int | None = None,
    ) -> nn.Module:
        return build_timm(
            arch,
            in_channels=in_channels,
            num_classes=num_classes,
            family=family_name,
            head_dropout=head_dropout,
            img_size=img_size,
        )
    return build


for _name, _presets in TIMM_PRESETS.items():
    register(ModelFamily(
        name=_name,
        pipeline="classification",
        presets=_presets,
        build=_make_build(_name),
    ))
