"""timm-backed classification families: resnet, efficientnet, convnext,
densenet, mobilenet, vit.

All families share one builder (:func:`build_timm`). Input adaptations for
small embedding patches (32×32 px):

- conv families: :func:`adapt_stem_for_small_inputs` relaxes early downsampling
  until the deepest feature map is at least 4×4. Untouched, all of them end at
  1×1 on a 32 px patch — they are built for 224 px and downsample by 32×. Until
  2026-09 only resnet got this (hardcoded), which is why the other families
  looked like weak architectures when they were really being handed a single
  pixel to classify from.
- vit: patch_size overridden to 2 → (img_size / 2)² tokens (e.g. 256 tokens
  for a 32×32 input).
"""

from __future__ import annotations

import torch.nn as nn

from models.registry import ModelFamily, register

# Patch size the stem adaptation is calibrated for when a caller does not say.
# 32 px is the So2Sat patch (320 m at 10 m/px) and the --patch-size default.
DEFAULT_IMG_SIZE = 32

# Spatial size the deepest feature map should keep. 4 is what the historical
# resnet-only surgery produced, so resnet's structure — and every resnet
# checkpoint ever written — is unchanged by generalising it.
MIN_FINAL_MAP = 4


def _final_map_size(model: nn.Module, in_channels: int, img_size: int) -> int:
    """Spatial size of the feature map entering the classifier, or 0 if not 4-D."""
    import torch

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            feats = model.forward_features(torch.zeros(2, in_channels, img_size, img_size))
    finally:
        model.train(was_training)
    return feats.shape[-1] if feats.dim() == 4 else 0


def _replace(model: nn.Module, path: str, new: nn.Module) -> None:
    parts = path.split(".")
    parent = model
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new
    else:
        setattr(parent, last, new)


def adapt_stem_for_small_inputs(
    model: nn.Module,
    in_channels: int,
    img_size: int = DEFAULT_IMG_SIZE,
    min_final_map: int = MIN_FINAL_MAP,
) -> list[str]:
    """Relax early downsampling so a small input keeps a usable feature map.

    Every timm classification model here is designed for 224 px ImageNet images
    and downsamples by 32×. On a 32 px embedding patch that leaves a 1×1 map at
    the classifier — all spatial structure inside the patch is gone before the
    first block finishes. Measured on the cultural split, the families that end
    at 1×1 cluster at kappa 0.599–0.601 regardless of whether they carry 604k or
    12.9M parameters, while resnet (4×4, via the surgery this generalises) gets
    0.6145 and a purpose-built 2-conv net (8×8) gets 0.6330.

    Walks the modules in definition order — which is forward order for these
    architectures — and neutralises each stride>1 op until ``forward_features``
    returns at least ``min_final_map``. A change that does not increase the map
    is reverted, which is what skips squeeze-excite branches: their pooling is a
    side path, so relaxing it moves nothing.

    Convs with kernel ≥4 (the 7×7 ResNet/DenseNet stems, ConvNeXt's 4×4
    patchify) are replaced by 3×3 stride-1; smaller convs keep their kernel and
    just lose the stride; pooling layers become Identity.

    For resnet this reproduces the previous hardcoded surgery exactly — 3×3
    stride-1 conv1 plus Identity maxpool — so resnet checkpoints still load.
    Checkpoints for the other families written before this change will not:
    their shapes genuinely differ, and ``load_state_dict`` says so loudly.

    Returns:
        The module paths that were relaxed, for logging.
    """
    relaxed: list[str] = []
    for path, mod in list(model.named_modules()):
        if _final_map_size(model, in_channels, img_size) >= min_final_map:
            break
        if not isinstance(mod, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
            continue
        stride = mod.stride if isinstance(mod.stride, tuple) else (mod.stride, mod.stride)
        if stride[0] <= 1:
            continue

        before = _final_map_size(model, in_channels, img_size)
        if isinstance(mod, nn.Conv2d) and mod.kernel_size[0] >= 4:
            _replace(model, path, nn.Conv2d(
                mod.in_channels, mod.out_channels, kernel_size=3, stride=1,
                padding=1, bias=mod.bias is not None,
            ))
            undo = lambda p=path, m=mod: _replace(model, p, m)  # noqa: E731
        elif isinstance(mod, nn.Conv2d):
            mod.stride = (1, 1)
            undo = lambda m=mod, s=stride: setattr(m, "stride", s)  # noqa: E731
        else:
            _replace(model, path, nn.Identity())
            undo = lambda p=path, m=mod: _replace(model, p, m)  # noqa: E731

        if _final_map_size(model, in_channels, img_size) <= before:
            undo()
        else:
            relaxed.append(path)
    return relaxed


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
    adapt_stem_for_small_inputs(model, in_channels, img_size or DEFAULT_IMG_SIZE)
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
