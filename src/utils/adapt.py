"""Warm-starting, layer freezing and shot budgets for per-city adaptation.

The protocol these support: train a model on So2Sat, then give it more
geographic context from WUDAPT -- globally, and then per city with a small
number of labels. Three pieces were missing from the stack.

**Warm starting.** ``--checkpoint`` loads weights and *skips training entirely*
(patch_classification.py, semantic_segmentation.py), which is right for
inference and useless for adaptation. ``--init-checkpoint`` loads the same file
and then trains. The two are deliberately separate flags rather than a mode
switch on one, because silently changing what ``--checkpoint`` does would
reinterpret every existing command in the campaign's history.

**Freezing.** ``run_training_loop`` optimises ``task_module.parameters()``
unconditionally, so setting ``requires_grad = False`` alone would still let Adam
update frozen weights through weight decay and momentum. :func:`apply_freeze`
sets the flags and ``loop.py`` now filters the optimiser to match.

**Shot budgets.** An N-shot curve needs a seeded, class-balanced subsample of
the training items, which nothing in the dataset layer offered.

Checkpoints store only ``model_state_dict`` (loop.py), so warm starting is a
plain ``load_state_dict`` -- no optimiser state to reconcile.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from loguru import logger
from torch import nn

__all__ = [
    "add_adaptation_args",
    "apply_freeze",
    "find_head_modules",
    "limit_shots_per_class",
    "limit_train_items",
    "load_init_weights",
    "trainable_parameters",
]

FREEZE_MODES = ("none", "backbone", "backbone_keep_bn")


def add_adaptation_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Add the Adaptation group, shared by both pipelines."""
    g = parser.add_argument_group("Adaptation")
    g.add_argument(
        "--init-checkpoint", type=Path, default=None,
        help="Warm-start model weights from this .pt file and THEN train. "
             "Distinct from --checkpoint, which loads weights and skips training.",
    )
    g.add_argument(
        "--freeze", choices=FREEZE_MODES, default="none",
        help="Which parameters to freeze. 'backbone' trains only the output head "
             "(head-only refit); 'backbone_keep_bn' also leaves BatchNorm affine "
             "parameters trainable so feature statistics can adapt (default: none).",
    )
    g.add_argument(
        "--shots-per-class", type=int, default=None,
        help="Shot budget for the per-city arm. Classification: at most N training "
             "samples per class per city. Segmentation: at most N training tiles per "
             "city, since tiles are inherently multi-class. None = use everything.",
    )
    return g


def load_init_weights(model: nn.Module, checkpoint: Path, device) -> dict:
    """Warm-start ``model`` from a training checkpoint. Returns a load report.

    Tolerant of a changed output head (``strict=False``) so a 17-class So2Sat
    checkpoint can seed a run with a different class count, but every missing or
    unexpected key is logged: a silently-mismatched load produces a model that
    trains fine and means nothing, which is the worst possible failure mode here.
    """
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)

    # strict=False forgives missing and unexpected *keys*, but a shape mismatch
    # still raises — so seeding a 5-class run from a 17-class So2Sat checkpoint
    # would die on fc.weight. Drop the mismatched tensors explicitly and report
    # them; that is what "tolerant of a changed head" has to mean in practice.
    own = model.state_dict()
    mismatched = {
        k: (tuple(v.shape), tuple(own[k].shape))
        for k, v in state.items()
        if k in own and hasattr(v, "shape") and v.shape != own[k].shape
    }
    usable = {k: v for k, v in state.items() if k not in mismatched}
    result = model.load_state_dict(usable, strict=False)

    missing, unexpected = list(result.missing_keys), list(result.unexpected_keys)
    n_loaded = len(usable) - len(unexpected)
    logger.info(
        f"Warm start from {checkpoint}: loaded {n_loaded}/{len(state)} tensors"
    )
    if mismatched:
        logger.warning(
            f"  {len(mismatched)} params re-initialised on a shape change "
            f"(usually the output head): "
            + ", ".join(f"{k} {a}->{b}" for k, (a, b) in list(mismatched.items())[:4])
        )
    if missing:
        logger.warning(f"  {len(missing)} params NOT in the checkpoint (kept random): {missing[:6]}")
    if unexpected:
        logger.warning(f"  {len(unexpected)} checkpoint params unused by this model: {unexpected[:6]}")
    if n_loaded == 0:
        raise ValueError(
            f"Warm start loaded zero tensors from {checkpoint}. The checkpoint almost "
            "certainly belongs to a different model family or preset."
        )
    return {
        "loaded": n_loaded,
        "missing": len(missing),
        "unexpected": len(unexpected),
        "reinit": len(mismatched),
    }


def find_head_modules(model: nn.Module, num_classes: int) -> list[str]:
    """Names of the modules that form the model's output head.

    Identified structurally rather than by hard-coded attribute names, because
    the head lives somewhere different in every family: ``fc`` on timm ResNets,
    ``classifier`` on EfficientNet/ConvNeXt, ``head`` on ViT, a final 1x1
    ``Conv2d`` on UNet and FCN8, a bare ``Linear`` on linear_probe. Any Linear or
    Conv2d whose output width equals ``num_classes`` is a head; there is normally
    exactly one, and taking all of them is correct for the fused/auxiliary heads
    that FCN8 and AttentionUNet carry.
    """
    names = []
    for name, mod in model.named_modules():
        width = None
        if isinstance(mod, nn.Linear):
            width = mod.out_features
        elif isinstance(mod, nn.Conv2d):
            width = mod.out_channels
        if width == num_classes:
            names.append(name)
    return names


def apply_freeze(model: nn.Module, mode: str, num_classes: int) -> dict:
    """Freeze parameters according to ``mode``. Returns a summary.

    ``none``
        Everything trainable.
    ``backbone``
        Only the output head trains -- the frozen-backbone arm. Equivalent to a
        linear probe on the pretrained features, but reusing the real model so
        the comparison against full fine-tuning is like for like.
    ``backbone_keep_bn``
        As above, plus BatchNorm affine parameters. Lets the network rescale
        features to a new city's statistics without changing what they detect,
        which is the cheap middle ground between the two arms.
    """
    if mode not in FREEZE_MODES:
        raise ValueError(f"unknown freeze mode {mode!r}; expected one of {FREEZE_MODES}")
    if mode == "none":
        for p in model.parameters():
            p.requires_grad_(True)
        n = sum(p.numel() for p in model.parameters())
        return {"mode": mode, "trainable": n, "frozen": 0, "head_modules": []}

    heads = find_head_modules(model, num_classes)
    if not heads:
        raise ValueError(
            f"--freeze {mode} found no output head with width {num_classes}. "
            "Freezing the whole model would train nothing, so this is refused "
            "rather than silently producing an untrainable run."
        )
    head_params = set()
    for name in heads:
        mod = model.get_submodule(name)
        head_params.update(id(p) for p in mod.parameters())

    for p in model.parameters():
        p.requires_grad_(id(p) in head_params)

    if mode == "backbone_keep_bn":
        for mod in model.modules():
            if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                for p in mod.parameters():
                    p.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info(
        f"freeze={mode}: {trainable:,} trainable / {frozen:,} frozen "
        f"({100 * trainable / max(trainable + frozen, 1):.2f}% trainable), head={heads}"
    )
    return {"mode": mode, "trainable": trainable, "frozen": frozen, "head_modules": heads}


def trainable_parameters(model: nn.Module) -> list:
    """Parameters an optimiser should actually see.

    ``requires_grad = False`` alone does not stop Adam: weight decay and
    momentum still move a parameter that is in the optimiser's param group, so
    the frozen weights drift and the "frozen backbone" arm quietly is not one.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError(
            "No trainable parameters remain. Check --freeze; training would be a no-op."
        )
    return params


def limit_shots_per_class(items: list, n: int | None, seed: int = 42) -> list:
    """Seeded, class-balanced subsample of training items: at most ``n`` per class.

    Grouped by ``(city, label)`` where items carry a city, so a 5-shot run means
    five examples per class *per city* rather than five worldwide -- the former
    is what the few-shot literature means and the latter would make the per-city
    arm depend on which cities happen to sort first.

    Only ``split == "train"`` items are capped; val and test pass through
    untouched, otherwise the shot budget would also shrink the yardstick.
    """
    if n is None:
        return items
    import random

    rng = random.Random(seed)
    buckets: dict[tuple, list] = defaultdict(list)
    out = []
    for it in items:
        if getattr(it, "split", None) != "train":
            out.append(it)
            continue
        buckets[(getattr(it, "city", None), it.label)].append(it)

    for key in sorted(buckets, key=lambda k: (str(k[0]), k[1])):
        group = buckets[key]
        rng.shuffle(group)
        out.extend(group[:n])

    n_train = sum(1 for it in out if getattr(it, "split", None) == "train")
    logger.info(f"--shots-per-class {n}: train items {len(items)} -> {n_train}")
    return out


def limit_train_items(items: list, split_map: dict, n: int | None, seed: int = 42) -> list:
    """Cap the number of training tiles, for the segmentation shot budget.

    Segmentation tiles are multi-class by nature, so "N per class" is not
    meaningful; the budget is N tiles per city, which is the closest honest
    analogue of a per-city label budget.
    """
    if n is None:
        return items
    import random
    import re

    # Mirror semantic_segmentation._key: under channel fusion it[0] is a tuple
    # of per-embedding paths, and split_map is keyed on the first of them.
    def _key(it):
        return it[0][0] if isinstance(it[0], tuple) else it[0]

    # The city is the filename stem, not a directory level: paths look like
    # {city_dir}/{output_name}/{year}/{split}/{city}_{grid_id}.npy, and the same
    # regex is used by build_city_tile_items because three So2Sat city names
    # contain brackets that glob would misread.
    name_re = re.compile(r"^(.+)_(\d+)\.npy$")

    rng = random.Random(seed)
    buckets: dict[str, list] = defaultdict(list)
    out = []
    for it in items:
        path = _key(it)
        if split_map.get(path) != "train":
            out.append(it)
            continue
        m = name_re.match(Path(path).name)
        buckets[m.group(1) if m else Path(path).parent.name].append(it)

    for city in sorted(buckets):
        group = buckets[city]
        rng.shuffle(group)
        out.extend(group[:n])

    n_train = sum(1 for it in out if split_map.get(_key(it)) == "train")
    logger.info(f"tile budget {n}: train tiles {len(items)} -> {n_train}")
    return out
