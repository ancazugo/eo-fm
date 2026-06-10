"""Model family registry.

Each architecture family registers itself here at import time (see the family
modules in this package). The two training pipelines query the registry:

- ``patch_classification.py`` uses ``families_for("classification")``
- ``semantic_segmentation.py`` uses ``families_for("segmentation")``

so adding a new model = one new module in ``models/`` + one ``register()`` call
(+ an import in ``models/__init__.py``), and it automatically appears in the
``--family`` CLI choices of the matching pipeline.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch.nn as nn


@dataclass(frozen=True)
class ModelFamily:
    """One architecture family (e.g. resnet, unet, mlp).

    Attributes:
        name: Family key used in CLI ``--family`` / ``--model-type``.
        pipeline: Which pipeline the family plugs into.
            "classification" → (B, C, H, W) → (B, num_classes)
            "segmentation"   → (B, C, H, W) → (B, num_classes, H, W)
        presets: preset name → arch payload. The payload type is family-specific
            (timm model name str, "mlp_512-256" spec str, (depth, base_features)
            tuple for unet, backbone str for resnet_unet).
        build: ``build(arch, in_channels, num_classes, **kwargs) -> nn.Module``.
            Extra kwargs not accepted by the builder are filtered out by
            :func:`build_model`, so callers can pass a superset.
        default_preset: Used when no preset is given.
    """

    name: str
    pipeline: Literal["classification", "segmentation"]
    presets: dict[str, Any] = field(default_factory=dict)
    build: Callable[..., nn.Module] = None
    default_preset: str = "base"


MODEL_REGISTRY: dict[str, ModelFamily] = {}


def register(family: ModelFamily) -> None:
    if family.name in MODEL_REGISTRY:
        raise ValueError(f"Model family '{family.name}' is already registered")
    MODEL_REGISTRY[family.name] = family


def get_family(name: str) -> ModelFamily:
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model family '{name}'. Choose from: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name]


def families_for(pipeline: str) -> list[str]:
    """Family names supporting a pipeline — feeds argparse ``choices``."""
    return sorted(f.name for f in MODEL_REGISTRY.values() if f.pipeline == pipeline)


def resolve_arch(family: str, preset: str | None = None, arch: Any | None = None) -> Any:
    """Resolve the arch payload: explicit override > preset > default preset."""
    fam = get_family(family)
    if arch is not None:
        return arch
    key = preset or fam.default_preset
    if key not in fam.presets:
        raise ValueError(
            f"Unknown preset '{key}' for family '{family}'. "
            f"Choose from: {sorted(fam.presets)}"
        )
    return fam.presets[key]


def build_model(
    family: str,
    preset: str | None = None,
    arch: Any | None = None,
    *,
    in_channels: int,
    num_classes: int,
    **kwargs: Any,
) -> nn.Module:
    """Build a model from the registry.

    ``kwargs`` may be a superset of what the family's builder accepts
    (e.g. head_dropout, img_size, bottleneck_dropout) — irrelevant ones are
    dropped, so the entry scripts don't need per-family branching.
    """
    fam = get_family(family)
    payload = resolve_arch(family, preset, arch)

    sig = inspect.signature(fam.build)
    accepts_var_kw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if not accepts_var_kw:
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    return fam.build(payload, in_channels=in_channels, num_classes=num_classes, **kwargs)
