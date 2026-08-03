"""T7 — the A1..B3 experiment ladder: one command, one comparison table.

Each rung trains a light head on frozen embeddings and reports block-level
metrics (T5) so every rung is directly comparable. Per-class deltas vs the
previous rung are the paper's ablation; B3 skips gracefully (not an error)
when ``torch_geometric`` isn't installed.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from loguru import logger

from .config import TrainConfig
from .eval import dense_predict_full, evaluate_blocks, majority_vote_per_block
from .models import (
    A1Linear,
    A2MLP,
    A3DilatedConv,
    B1MeanPoolMLP,
    B2AttentionPoolMLP,
    B3GNN,
    MultiScaleWrapper,
    is_gnn_available,
)
from .train import block_forward_and_loss, dense_forward_and_loss, seed_all, train_loop

LADDER_ORDER = ["A1", "A1_MS", "A2_MS", "A3", "B1", "B2", "B3"]

EXPERIMENTS: dict[str, dict] = {
    "A1": {"family": "A", "multiscale": False},
    "A1_MS": {"family": "A", "multiscale": True},
    "A2_MS": {"family": "A", "multiscale": True, "hidden": True},
    "A3": {"family": "A", "multiscale": False, "dilated": True},
    "B1": {"family": "B", "pooling": "mean"},
    "B2": {"family": "B", "pooling": "attention"},
    "B3": {"family": "B", "pooling": "mean", "gnn": True},
}


def build_model(exp_id: str, in_channels: int, *, extra_features: int = 0) -> torch.nn.Module:
    """One model per ladder rung. Raises for an unknown ``exp_id``."""
    spec = EXPERIMENTS.get(exp_id)
    if spec is None:
        raise ValueError(f"unknown experiment {exp_id!r} — one of {LADDER_ORDER}")

    if spec["family"] == "A":
        if spec.get("dilated"):
            return A3DilatedConv(in_channels=in_channels)
        base_channels = in_channels * 3 if spec["multiscale"] else in_channels
        base = A2MLP(in_channels=base_channels) if spec.get("hidden") else A1Linear(in_channels=base_channels)
        return MultiScaleWrapper(base, in_channels=in_channels) if spec["multiscale"] else base

    # family == "B"
    if spec.get("gnn"):
        if not is_gnn_available():
            raise ImportError(
                f"{exp_id} requires torch_geometric (the 'gnn' extra) — not installed"
            )
        return B3GNN(in_features=in_channels + extra_features)
    if spec["pooling"] == "attention":
        return B2AttentionPoolMLP(channels=in_channels, extra_features=extra_features)
    return B1MeanPoolMLP(in_features=in_channels + extra_features)


def run_dense_experiment(
    exp_id: str,
    train_dataset,
    eval_mosaic: np.ndarray,
    eval_block_idx: np.ndarray,
    eval_gt: pd.DataFrame,
    cfg: TrainConfig,
    *,
    in_channels: int,
    device: torch.device | None = None,
) -> dict:
    """Train an A-family rung and score it at block level on held-out data."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(cfg.seed)
    model = build_model(exp_id, in_channels)
    n_params = sum(p.numel() for p in model.parameters())

    t0 = time.perf_counter()
    fit = train_loop(model, train_dataset, dense_forward_and_loss, cfg, device=device)
    train_time_s = time.perf_counter() - t0

    pred = dense_predict_full(model, eval_mosaic, device)  # 0-indexed classes
    n_blocks = int(eval_block_idx.max())
    pred_lcz = majority_vote_per_block(pred, eval_block_idx, n_blocks)  # -> 1-indexed LCZ codes
    metrics = evaluate_blocks(pred_lcz, eval_gt)

    return {"exp_id": exp_id, "family": "A", "n_params": n_params,
            "train_time_s": train_time_s, "final_loss": fit["final_loss"], "metrics": metrics}


def run_block_experiment(
    exp_id: str,
    train_dataset,
    eval_dataset,
    eval_gt: pd.DataFrame,
    cfg: TrainConfig,
    *,
    in_channels: int,
    extra_features: int = 0,
    device: torch.device | None = None,
) -> dict | None:
    """Train a B-family rung and score it; returns None if B3 is skipped."""
    try:
        model = build_model(exp_id, in_channels, extra_features=extra_features)
    except ImportError as e:
        logger.warning(f"[{exp_id}] skipped: {e}")
        return None

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(cfg.seed)
    n_params = sum(p.numel() for p in model.parameters())

    t0 = time.perf_counter()
    fit = train_loop(model, train_dataset, block_forward_and_loss, cfg, device=device)
    train_time_s = time.perf_counter() - t0

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(eval_dataset)):
            item = eval_dataset[i]
            x = item["embedding"]
            extra = item.get("extra")
            if extra is not None and extra.numel() > 0:
                x = torch.cat([x, extra], dim=-1)
            logits = model(x.unsqueeze(0).to(device))
            preds.append(int(logits.argmax(dim=-1).item()) + 1)   # LCZ code
    metrics = evaluate_blocks(np.array(preds), eval_gt)

    return {"exp_id": exp_id, "family": "B", "n_params": n_params,
            "train_time_s": train_time_s, "final_loss": fit["final_loss"], "metrics": metrics}


def _delta_per_class(prev: dict | None, cur: dict) -> dict:
    if prev is None or "per_class" not in prev.get("metrics", {}) or "per_class" not in cur["metrics"]:
        return {}
    out = {}
    for c, stats in cur["metrics"]["per_class"].items():
        prior = prev["metrics"]["per_class"].get(c)
        out[c] = stats["f1"] - prior["f1"] if prior else float("nan")
    return out


def render_ladder_table(results: list[dict | None]) -> str:
    """One markdown table: block metrics, params, train time, per-class deltas."""
    lines = ["# Experiment ladder", "",
             "| Rung | Family | OA | Macro-F1 | Coarse OA | Params | Train (s) |",
             "|---|---|---|---|---|---|---|"]
    prev = None
    for r in results:
        if r is None:
            lines.append("| _(skipped — torch_geometric unavailable)_ | | | | | | |")
            continue
        m = r["metrics"]
        oa = m.get("oa", float("nan"))
        f1 = m.get("macro_f1", float("nan"))
        coarse_oa = m.get("coarse_oa", float("nan"))
        lines.append(
            f"| {r['exp_id']} | {r['family']} | {oa:.3f} | {f1:.3f} | {coarse_oa:.3f} | "
            f"{r['n_params']:,} | {r['train_time_s']:.1f} |"
        )
        prev = r
    lines.append("")
    lines.append("## Per-class F1 delta vs previous rung")
    lines.append("")
    prev = None
    for r in results:
        if r is None:
            continue
        deltas = _delta_per_class(prev, r)
        if deltas:
            parts = ", ".join(f"LCZ{c}: {d:+.3f}" for c, d in sorted(deltas.items()))
            lines.append(f"- **{r['exp_id']}**: {parts}")
        prev = r
    return "\n".join(lines) + "\n"
