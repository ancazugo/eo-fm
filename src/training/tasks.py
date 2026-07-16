"""Task modules: model wrapper + loss + metrics for each pipeline.

Both tasks expose a uniform interface consumed by
:func:`training.loop.run_training_loop`:

- ``monitor``: name of the validation metric used for checkpointing/early stop
- ``reset_train_metrics()`` / ``train_step(batch, device) -> loss | None``
  / ``compute_train_logs() -> dict``
- ``reset_val_metrics()`` / ``val_step(batch, device)``
  / ``compute_val_logs() -> dict``

The wrapped architecture always lives in ``self.model`` and checkpoints store
``model.state_dict()`` only, so checkpoints are interchangeable across task
wrappers and remain compatible with previously trained models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, JaccardIndex
from torchmetrics.classification import MulticlassCohenKappa, MulticlassF1Score


# ─── Classification ───────────────────────────────────────────────────────────

class LCZResNetModule(nn.Module):
    """Wraps a classification model for multiclass LCZ patch classification.

    Loss: CrossEntropyLoss with ignore_index=-1 (skips all-nodata patches),
    optionally class-weighted, label-smoothed and mixup-regularized.
    Monitored metric: val_f1 (macro F1) by default, or val_kappa.

    Args:
        model: Any (B, C, H, W) → (B, num_classes) module.
        num_classes: Number of classification classes.
        lr: Adam learning rate.
        weight_decay: Adam L2 regularization.
        max_epochs: Total training epochs (used for CosineAnnealingLR T_max).
        class_weights: Optional (num_classes,) tensor of per-class CE weights.
        label_smoothing: CE label smoothing (default 0.0).
        mixup_alpha: Beta(alpha, alpha) mixup on training batches (0 = off).
        monitor: Validation metric for checkpointing/early stopping
            ("val_f1" or "val_kappa").
        logit_adjustment_tau: Logit-adjusted CE (Menon et al. 2021): the
            training loss sees logits + tau*log(prior); val/test use raw
            logits, which shifts decisions toward rare classes (0 = off).
        class_priors: (num_classes,) train-frequency priors; required when
            logit_adjustment_tau > 0.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 50,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        mixup_alpha: float = 0.0,
        monitor: str = "val_f1",
        logit_adjustment_tau: float = 0.0,
        class_priors: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.mixup_alpha = mixup_alpha
        self.monitor = monitor
        self.logit_adjustment_tau = logit_adjustment_tau
        if logit_adjustment_tau > 0:
            if class_priors is None:
                raise ValueError("logit_adjustment_tau > 0 requires class_priors")
            self.register_buffer(
                "log_prior", torch.log(class_priors.clamp_min(1e-12))
            )

        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=-1, weight=class_weights, label_smoothing=label_smoothing
        )
        # Per-sample variant for weighted (pseudo-labeled) batches. Normalizing
        # by Σ sample_w · class_w[target] makes an all-weights-1 batch match
        # ce_loss's "mean" reduction exactly.
        self.ce_loss_none = nn.CrossEntropyLoss(
            ignore_index=-1, weight=class_weights,
            label_smoothing=label_smoothing, reduction="none",
        )
        self.register_buffer(
            "_ce_class_w",
            class_weights.clone() if class_weights is not None
            else torch.ones(num_classes),
        )

        metric_kw = dict(task="multiclass", num_classes=num_classes, ignore_index=-1)
        f1_kw = dict(num_classes=num_classes, ignore_index=-1)

        self.train_acc        = Accuracy(**metric_kw)
        self.train_acc_macro  = Accuracy(**metric_kw, average="macro")
        self.train_f1_macro   = MulticlassF1Score(**f1_kw, average="macro")
        self.train_f1_micro   = MulticlassF1Score(**f1_kw, average="micro")
        self.train_kappa      = MulticlassCohenKappa(num_classes=num_classes, ignore_index=-1)

        self.val_acc          = Accuracy(**metric_kw)
        self.val_acc_macro    = Accuracy(**metric_kw, average="macro")
        self.val_f1           = MulticlassF1Score(**f1_kw, average="macro")
        self.val_f1_micro     = MulticlassF1Score(**f1_kw, average="micro")
        self.val_kappa        = MulticlassCohenKappa(num_classes=num_classes, ignore_index=-1)

        self._train_loss_sum = 0.0
        self._n_train = 0
        self._val_loss_sum = 0.0
        self._n_val = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.float())

    def _adjust_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Train-time logit adjustment; identity when disabled."""
        if self.logit_adjustment_tau > 0:
            return logits + self.logit_adjustment_tau * self.log_prior
        return logits

    def _weighted_ce(
        self, logits: torch.Tensor, labels: torch.Tensor, sample_w: torch.Tensor
    ) -> torch.Tensor:
        """Per-sample weighted CE (ignore_index=-1 samples contribute nothing)."""
        per_sample = self.ce_loss_none(logits, labels)   # includes class weight
        valid = labels != -1
        cw = torch.where(
            valid, self._ce_class_w[labels.clamp_min(0)],
            torch.zeros_like(sample_w),
        )
        denom = (sample_w * cw).sum().clamp_min(1e-8)
        return (sample_w * per_sample).sum() / denom

    # ── Loop interface ────────────────────────────────────────────────────────

    def reset_train_metrics(self) -> None:
        for m in (self.train_acc, self.train_acc_macro, self.train_f1_macro,
                  self.train_f1_micro, self.train_kappa):
            m.reset()
        self._train_loss_sum = 0.0
        self._n_train = 0

    def train_step(self, batch: dict, device: torch.device) -> torch.Tensor | None:
        """Forward + loss + metric update. Returns the loss tensor, or None
        for skipped batches (all-nodata labels or NaN loss)."""
        images = batch["image"].to(device).float()
        labels = batch["label"].to(device)
        sample_w = batch.get("weight")
        if sample_w is not None:
            sample_w = sample_w.to(device).float()
        if (labels != -1).sum() == 0:
            return None

        def _ce(logits: torch.Tensor, targets: torch.Tensor,
                w: torch.Tensor | None) -> torch.Tensor:
            if w is None:
                return self.ce_loss(logits, targets)
            return self._weighted_ce(logits, targets, w)

        if self.mixup_alpha > 0 and self.training:
            lam = float(torch.distributions.Beta(
                self.mixup_alpha, self.mixup_alpha).sample())
            perm = torch.randperm(images.size(0), device=device)
            logits = self.model(lam * images + (1 - lam) * images[perm])
            loss_logits = self._adjust_logits(logits)
            loss = lam * _ce(loss_logits, labels, sample_w) + \
                (1 - lam) * _ce(loss_logits, labels[perm],
                                sample_w[perm] if sample_w is not None else None)
        else:
            logits = self.model(images)
            loss = _ce(self._adjust_logits(logits), labels, sample_w)
        if torch.isnan(loss):
            return None
        # Train metrics are computed against the dominant (unpermuted) labels;
        # under mixup they are an approximation, useful only as a trend.
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            self.train_acc(preds, labels)
            self.train_acc_macro(preds, labels)
            self.train_f1_macro(preds, labels)
            self.train_f1_micro(preds, labels)
            self.train_kappa(preds, labels)
        self._train_loss_sum += loss.item()
        self._n_train += 1
        return loss

    def compute_train_logs(self) -> dict[str, float]:
        return {
            "train_loss":      self._train_loss_sum / max(1, self._n_train),
            "train_oa":        self.train_acc.compute().item(),
            "train_acc_macro": self.train_acc_macro.compute().item(),
            "train_f1_macro":  self.train_f1_macro.compute().item(),
            "train_f1_micro":  self.train_f1_micro.compute().item(),
            "train_kappa":     self.train_kappa.compute().item(),
        }

    def reset_val_metrics(self) -> None:
        for m in (self.val_acc, self.val_acc_macro, self.val_f1,
                  self.val_f1_micro, self.val_kappa):
            m.reset()
        self._val_loss_sum = 0.0
        self._n_val = 0

    def val_step(self, batch: dict, device: torch.device) -> None:
        images = batch["image"].to(device).float()
        labels = batch["label"].to(device)
        if (labels != -1).sum() == 0:
            return
        logits = self.model(images)
        loss = self.ce_loss(logits, labels)
        preds = logits.argmax(dim=1)
        self.val_acc(preds, labels)
        self.val_acc_macro(preds, labels)
        self.val_f1(preds, labels)
        self.val_f1_micro(preds, labels)
        self.val_kappa(preds, labels)
        self._val_loss_sum += loss.item()
        self._n_val += 1

    def compute_val_logs(self) -> dict[str, float]:
        return {
            "val_loss":      self._val_loss_sum / max(1, self._n_val),
            "val_acc":       self.val_acc.compute().item(),
            "val_acc_macro": self.val_acc_macro.compute().item(),
            "val_f1":        self.val_f1.compute().item(),
            "val_f1_micro":  self.val_f1_micro.compute().item(),
            "val_kappa":     self.val_kappa.compute().item(),
        }


# ─── Segmentation ─────────────────────────────────────────────────────────────

class MulticlassDiceLoss(nn.Module):
    """Soft multiclass Dice loss, macro-averaged over present classes.

    For each class, computes the soft Dice coefficient over valid pixels
    (those not equal to ``ignore_index``), then averages over classes that
    have at least one positive target pixel. Returns 0 if no valid class exists.

    Args:
        num_classes: Number of segmentation classes.
        ignore_index: Label value to exclude from loss computation.
        smooth: Laplace smoothing term to avoid division by zero.
    """

    def __init__(
        self, num_classes: int, ignore_index: int = -1, smooth: float = 1.0
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (N, C, H, W) — unnormalized class logits.
            targets: (N, H, W)    — integer class indices; ignore_index excluded.
        """
        probs = F.softmax(logits, dim=1)          # (N, C, H, W)
        valid = targets != self.ignore_index       # (N, H, W) bool

        dice_terms: list[torch.Tensor] = []
        for c in range(self.num_classes):
            target_c = ((targets == c) & valid).float()   # (N, H, W)
            pred_c = probs[:, c]                          # (N, H, W)

            p = pred_c[valid]
            t = target_c[valid]

            if t.numel() == 0 or t.sum() == 0:
                continue  # absent class — skip so macro average is fair

            intersection = (p * t).sum()
            dice = (2.0 * intersection + self.smooth) / (
                p.sum() + t.sum() + self.smooth
            )
            dice_terms.append(1.0 - dice)

        if not dice_terms:
            return logits.sum() * 0.0  # differentiable zero

        return torch.stack(dice_terms).mean()


class LCZUNetModule(nn.Module):
    """Wraps a segmentation model for multiclass LCZ segmentation.

    Loss: ``(1 − dice_weight) × CrossEntropy + dice_weight × MulticlassDice``
    Both losses use ``ignore_index=-1`` to skip unlabeled pixels.
    Monitored metric: val_miou (macro mIoU).

    Args:
        model: Any (B, C, H, W) → (B, num_classes, H, W) module.
        num_classes: Number of segmentation classes.
        lr: Adam learning rate.
        weight_decay: Adam L2 regularization.
        dice_weight: Weighting of Dice loss (0 = CE only, 1 = Dice only).
        max_epochs: Total training epochs (used for CosineAnnealingLR T_max).
    """

    monitor = "val_miou"

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dice_weight: float = 0.5,
        max_epochs: int = 50,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.dice_weight = dice_weight
        self.max_epochs = max_epochs

        self.dice_loss = MulticlassDiceLoss(num_classes, ignore_index=-1)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

        metric_kw = dict(task="multiclass", num_classes=num_classes, ignore_index=-1)
        self.val_miou = JaccardIndex(**metric_kw, average="macro")
        self.val_acc = Accuracy(**metric_kw)

        self._train_loss_sum = self._train_ce_sum = self._train_dice_sum = 0.0
        self._n_train = 0
        self._val_loss_sum = 0.0
        self._n_val = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.float())

    def _loss(
        self, logits: torch.Tensor, masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ce = self.ce_loss(logits, masks)
        dice = self.dice_loss(logits, masks)
        combined = (1.0 - self.dice_weight) * ce + self.dice_weight * dice
        return combined, ce, dice

    # ── Loop interface ────────────────────────────────────────────────────────

    def reset_train_metrics(self) -> None:
        self._train_loss_sum = self._train_ce_sum = self._train_dice_sum = 0.0
        self._n_train = 0

    def train_step(self, batch: dict, device: torch.device) -> torch.Tensor | None:
        images = batch["image"].to(device).float()
        masks = batch["mask"].to(device)
        # Skip all-nodata batches (CE returns NaN when every pixel is ignored)
        if (masks != -1).sum() == 0:
            return None
        logits = self.model(images)
        loss, ce, dice = self._loss(logits, masks)
        if torch.isnan(loss):
            return None
        self._train_loss_sum += loss.item()
        self._train_ce_sum += ce.item() if not torch.isnan(ce) else 0.0
        self._train_dice_sum += dice.item()
        self._n_train += 1
        return loss

    def compute_train_logs(self) -> dict[str, float]:
        n = max(1, self._n_train)
        return {
            "train_loss": self._train_loss_sum / n,
            "train_ce":   self._train_ce_sum / n,
            "train_dice": self._train_dice_sum / n,
        }

    def reset_val_metrics(self) -> None:
        self.val_miou.reset()
        self.val_acc.reset()
        self._val_loss_sum = 0.0
        self._n_val = 0

    def val_step(self, batch: dict, device: torch.device) -> None:
        images = batch["image"].to(device).float()
        masks = batch["mask"].to(device)
        logits = self.model(images)
        loss, _, _ = self._loss(logits, masks)
        preds = logits.argmax(dim=1)
        self.val_miou(preds, masks)
        self.val_acc(preds, masks)
        self._val_loss_sum += loss.item()
        self._n_val += 1

    def compute_val_logs(self) -> dict[str, float]:
        return {
            "val_loss": self._val_loss_sum / max(1, self._n_val),
            "val_miou": self.val_miou.compute().item(),
            "val_acc":  self.val_acc.compute().item(),
        }
