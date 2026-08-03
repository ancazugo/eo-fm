"""T5 — a simple step-based training loop, shared by A and B experiments.

Not ``src/training/loop.py``: that loop duck-types an epoch-oriented task/
datamodule with integer class targets, and its checkpoint/monitor plumbing
assumes that shape. Dense bitmask windows (class-balanced streaming sampler,
no natural "epoch") and the marginalised-set loss don't fit it, and adapting
it would cost more than this ~80-line loop replaces. Embeddings are frozen
throughout — every model here is a light head trained from scratch.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

from .config import TrainConfig
from .losses import bitmask_to_target, marginalized_ce


def seed_all(seed: int) -> None:
    """Seed random/numpy/torch(+CUDA) and force deterministic cuDNN kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dense_forward_and_loss(model, batch, cfg: TrainConfig) -> torch.Tensor:
    """A-family step: per-pixel logits -> marginalised CE over the bitmask."""
    logits = model(batch["image"])
    target = bitmask_to_target(batch["bitmask"]).permute(0, 3, 1, 2)
    weights = torch.tensor(cfg.class_weights) if cfg.class_weights else None
    return marginalized_ce(
        logits, target, valid=batch["valid"], class_weights=weights,
        smoothing_eps=cfg.smoothing_eps, gamma=cfg.conf_gamma,
    )


def block_forward_and_loss(model, batch, cfg: TrainConfig) -> torch.Tensor:
    """B1-family step: concat(pooled embedding, extra) -> marginalised CE.

    Matches ``B1MeanPoolMLP``'s flat ``in_features = in_channels +
    extra_features`` contract. B2 (attention pooling over a raw per-block
    pixel set) and B3 (graph batches) need different batch shapes than
    ``BlockDataset`` produces here and are wired directly in
    ``experiments.py`` rather than through this helper.
    """
    x = batch["embedding"]
    extra = batch.get("extra")
    if extra is not None and extra.numel() > 0:
        x = torch.cat([x, extra], dim=-1)
    logits = model(x)
    target = bitmask_to_target(batch["bitmask"])
    weights = torch.tensor(cfg.class_weights) if cfg.class_weights else None
    conf = batch.get("confidence")
    return marginalized_ce(
        logits, target, class_weights=weights,
        smoothing_eps=cfg.smoothing_eps, conf=conf, gamma=cfg.conf_gamma,
    )


def _lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def train_loop(
    model: torch.nn.Module,
    dataset,
    forward_and_loss,
    cfg: TrainConfig,
    *,
    device: torch.device | None = None,
    ckpt_path: str | Path | None = None,
) -> dict:
    """Frozen-embedding step-based loop: AdamW + cosine-with-warmup.

    ``forward_and_loss(model, batch, cfg) -> scalar loss`` decouples the loop
    from A vs B model shapes (:func:`dense_forward_and_loss` /
    :func:`block_forward_and_loss` cover both). Checkpoint (when
    ``ckpt_path`` given): ``{"model_state_dict", "step", "config_hash",
    "metrics"}``.

    Call :func:`seed_all` yourself before constructing ``model`` if you need
    the run fully reproducible end-to-end — its random init draws from the
    ambient torch RNG at construction time, so seeding only here (after
    ``model`` already exists) cannot retroactively fix that draw. This
    function still reseeds internally for its own batch-order randomness.
    """
    seed_all(cfg.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: _lr_lambda(s, cfg.warmup_steps, cfg.steps)
    )

    losses: list[float] = []
    it = iter(loader)
    for step in range(cfg.steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        loss = forward_and_loss(model, batch, cfg)
        opt.zero_grad()
        loss.backward()
        if cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        opt.step()
        sched.step()
        losses.append(float(loss.detach().cpu()))

        if (step + 1) % max(cfg.log_every, 1) == 0 or step == cfg.steps - 1:
            window = losses[-cfg.log_every:]
            logger.info(f"[{cfg.exp_id}] step {step + 1}/{cfg.steps} "
                        f"loss={np.mean(window):.4f}")
            if cfg.wandb:
                import wandb
                if wandb.run:
                    wandb.log({"loss": np.mean(window), "step": step + 1})

    metrics = {
        "final_loss": float(np.mean(losses[-min(len(losses), cfg.log_every):])),
        "first_loss": losses[0] if losses else float("nan"),
    }
    if ckpt_path is not None:
        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "step": cfg.steps,
            "config_hash": cfg.config_hash,
            "metrics": metrics,
        }, ckpt_path)
        logger.info(f"[{cfg.exp_id}] checkpoint -> {ckpt_path}")
    return {"losses": losses, **metrics}
