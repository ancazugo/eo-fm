"""Weights & Biases utilities for logging and sweeps."""

from typing import Any

import wandb
from lightning.pytorch.loggers import WandbLogger
from loguru import logger

from conf import SklearnConfig, WandbConfig


def get_wandb_logger(config: WandbConfig, **kwargs: Any) -> WandbLogger:
    """Create a WandB logger for Lightning training."""
    return WandbLogger(
        project=config.project,
        log_model=False,
        **kwargs,
    )


def log_sklearn_metrics(metrics: dict[str, float], config: dict | None = None) -> None:
    """Log sklearn metrics to the active WandB run."""
    if config:
        wandb.config.update(config)
    wandb.log(metrics)


def log_sklearn_cv_metrics(cv_results: dict[str, Any]) -> None:
    """Log cross-validation metrics to the active WandB run.

    Logs per-fold scores as a table and mean/std as summary metrics.
    """
    # Log mean/std as summary metrics
    summary = {k: v for k, v in cv_results.items() if not k.endswith("_per_fold")}
    wandb.log(summary)

    # Log per-fold scores as a wandb Table
    fold_keys = [k for k in cv_results if k.endswith("_per_fold")]
    if fold_keys:
        n_folds = len(cv_results[fold_keys[0]])
        metric_names = [k.replace("cv_test_", "").replace("_per_fold", "") for k in fold_keys]
        columns = ["fold"] + metric_names
        data = []
        for i in range(n_folds):
            row = [i + 1] + [cv_results[k][i] for k in fold_keys]
            data.append(row)
        table = wandb.Table(columns=columns, data=data)
        wandb.log({"cv_fold_metrics": table})


def run_sklearn_sweep(
    train_fn,
    wandb_config: WandbConfig,
    sweep_parameters: dict[str, Any],
) -> str:
    """Run a WandB Bayesian sweep for sklearn hyperparameter tuning.

    Args:
        train_fn: Callable that takes no args, reads from wandb.config, and logs metrics.
        wandb_config: WandB configuration.
        sweep_parameters: Dict of parameter specs for wandb.sweep.

    Returns:
        The sweep ID.
    """
    sweep_config = {
        "method": wandb_config.sweep_method,
        "metric": {"name": wandb_config.sweep_metric, "goal": wandb_config.sweep_goal},
        "parameters": sweep_parameters,
    }

    sweep_id = wandb.sweep(sweep=sweep_config, project=wandb_config.project)
    logger.info(f"Starting sweep {sweep_id} with {wandb_config.sweep_count} trials")
    wandb.agent(sweep_id, function=train_fn, count=wandb_config.sweep_count)
    return sweep_id
