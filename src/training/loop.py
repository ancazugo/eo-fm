"""Generic training loop shared by both pipelines.

Replaces the near-identical ``_run_resnet_training_loop`` and
``_run_unet_training_loop``: Adam + CosineAnnealingLR, per-epoch WandB logging,
early stopping and checkpointing on the task's monitored metric (``val_f1``
for classification, ``val_miou`` for segmentation).

Checkpoint format (unchanged from the previous loops):
    {"model_state_dict": ..., "epoch": ..., <task.monitor>: <best value>}
"""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger


def run_training_loop(
    task_module,
    datamodule,
    device: torch.device,
    max_epochs: int,
    early_stopping_patience: int,
    run_dir: Path,
    model_name: str,
    warmup_epochs: int = 0,
):
    """Train ``task_module`` and return ``(task_with_best_weights, best_ckpt_path)``.

    Args:
        task_module: LCZResNetModule or LCZUNetModule (moved to device inside).
        datamodule: Object with setup(), train_dataloader(), val_dataloader().
        device: Device to train on.
        max_epochs: Maximum number of epochs.
        early_stopping_patience: Stop after this many epochs without improvement
            of the task's monitored metric.
        run_dir: Directory to save checkpoints.
        model_name: Stem for the checkpoint filename.
        warmup_epochs: Linear LR warmup epochs before cosine decay (0 = off).
    """
    import wandb

    task_module = task_module.to(device)
    opt = torch.optim.Adam(
        task_module.parameters(), lr=task_module.lr, weight_decay=task_module.weight_decay
    )
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.01, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, max_epochs - warmup_epochs)
        )
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    monitor = task_module.monitor
    best_value = -1.0
    patience_counter = 0
    best_ckpt_path: Path | None = None

    for epoch in range(max_epochs):
        # ── Train ─────────────────────────────────────────────────────────
        task_module.train()
        task_module.reset_train_metrics()
        for batch in train_loader:
            opt.zero_grad()
            loss = task_module.train_step(batch, device)
            if loss is None:
                continue
            loss.backward()
            opt.step()
        train_logs = task_module.compute_train_logs()

        # ── Validate ───────────────────────────────────────────────────────
        task_module.eval()
        task_module.reset_val_metrics()
        with torch.no_grad():
            for batch in val_loader:
                task_module.val_step(batch, device)
        val_logs = task_module.compute_val_logs()
        sched.step()

        if wandb.run:
            wandb.log({**train_logs, **val_logs, "epoch": epoch + 1})
        logger.info(
            f"Epoch {epoch+1}/{max_epochs}  "
            f"loss={train_logs['train_loss']:.4f}  "
            f"{monitor}={val_logs[monitor]:.4f}  val_acc={val_logs['val_acc']:.4f}"
        )

        value = val_logs[monitor]
        if value > best_value:
            best_value = value
            patience_counter = 0
            best_ckpt_path = run_dir / f"{model_name}-best.pt"
            torch.save(
                {
                    "model_state_dict": task_module.model.state_dict(),
                    "epoch": epoch + 1,
                    monitor: value,
                },
                best_ckpt_path,
            )
            logger.info(f"  → New best ({monitor}={value:.4f}), checkpoint saved")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    if best_ckpt_path and best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        task_module.model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            f"Loaded best model ({monitor}={ckpt[monitor]:.4f}) from {best_ckpt_path}"
        )

    return task_module, best_ckpt_path
