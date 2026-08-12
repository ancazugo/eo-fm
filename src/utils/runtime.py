"""Shared runtime helpers for the training / inference entry scripts.

Centralises boilerplate used by patch_classification.py,
semantic_segmentation.py, infer_roi.py and the ensemble/SSL scripts:
device selection, checkpoint loading, dequantize-function selection,
input-channel detection, WandB run initialisation, and the per-city
full-ROI inference loop. (Shared argparse helpers live in utils.cli.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from loguru import logger

# Embeddings stored quantized on disk — dequantization is always required so
# it is applied automatically (--dequantize remains as an explicit force flag).
AUTO_DEQUANTIZE = {"alpha_earth_coop", "seamless"}


def resolve_device(accelerator: str = "auto") -> torch.device:
    """Map an --accelerator CLI value to a torch.device.

    Accepts "auto", "cpu", "cuda", "gpu" and "mps".
    """
    if accelerator == "cpu":
        return torch.device("cpu")
    if accelerator in ("cuda", "gpu"):
        return torch.device("cuda")
    if accelerator == "mps":
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint_weights(task, checkpoint: Path, device: torch.device) -> Path:
    """Load model weights from a training checkpoint into a task module.

    Accepts both the training-loop checkpoint format
    ``{"model_state_dict", "epoch", <monitor>}`` and a bare state dict.
    Returns the checkpoint path (mirrors run_training_loop's return).
    """
    logger.info(f"Loading checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location=device)
    task.model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    task.to(device)
    return checkpoint


def resolve_dequantize(
    embedding_name: str,
    force: bool = False,
) -> tuple[Callable | None, int | None]:
    """Select the dequantize function for an embedding type.

    Auto-applies for embeddings in AUTO_DEQUANTIZE (alpha_earth_coop, seamless);
    ``force=True`` (the --dequantize flag) enables it for any other embedding.

    Returns:
        (dequantize_fn | None, in_channels_override | None) — the override is 72
        for seamless, whose 13 raw ESD bands expand to 72 channels.
    """
    if not (force or embedding_name in AUTO_DEQUANTIZE):
        return None, None
    if force and embedding_name not in AUTO_DEQUANTIZE:
        logger.warning(
            f"--dequantize forced for '{embedding_name}', which is not a "
            "quantized source — already-float data (e.g. tesserav1.1) will be "
            "double-transformed. Drop --dequantize unless you know better."
        )
    if embedding_name == "seamless":
        from dequantize_embeddings import dequantize_esd
        logger.info("Seamless: dequantize_esd applied (13→72 channels)")
        return dequantize_esd, 72
    from dequantize_embeddings import dequantize_alphaearth_embeddings
    logger.info(f"{embedding_name}: dequantize_alphaearth_embeddings applied")
    return dequantize_alphaearth_embeddings, None


def detect_in_channels(first_npy: Path, override: int | None = None) -> int:
    """Read the channel count from the first .npy file (or use the override)."""
    in_channels = int(np.load(first_npy, mmap_mode="r").shape[0])
    if override is not None:
        in_channels = override
    logger.info(f"Detected in_channels = {in_channels}")
    return in_channels


def init_run(
    output_dir: Path,
    run_cfg: dict,
    run_name: str | None,
    default_name: str,
    wandb_project: str = "lcz-classification-dl",
    wandb_entity: str = "phd-thesis-team",
    no_wandb: bool = False,
) -> Path:
    """Initialise WandB (unless disabled) and create/return the run directory.

    With WandB the run directory is named after the generated run name;
    without it, ``run_name`` or ``default_name`` is used.
    """
    import wandb

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not no_wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            dir=str(output_dir),
            config=run_cfg,
            name=run_name,
        )
        run_dir = output_dir / wandb.run.name
    else:
        run_dir = output_dir / (run_name or default_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_city_inference(
    model: torch.nn.Module,
    model_type: str,
    task_label: str,
    preset: str,
    city_dirs: list[Path],
    run_dir: Path,
    device: torch.device,
    dequantize_fn: Callable | None,
    *,
    normalize: tuple | None = None,
    embedding_name: str,
    embedding_dir: Path,
    num_classes: int,
    patch_size: int,
    overlap: int,
    batch_size: int,
    year: str,
    margin_m: float,
) -> None:
    """Full-ROI inference for each city: writes one prediction GeoTIFF per city.

    The bbox is taken from {city}_grid.gpkg; the output is named
    ``{run_dir.name}_{model_type}-{preset}-{task_label}-prediction_{city}.tif``.

    ``normalize`` must be the same ``(mean, std)`` the model was trained with,
    or the maps are wrong; callers pass the arrays they fed to the datamodule.
    """
    import geopandas as gpd
    from infer_roi import infer_roi

    model.eval()
    for city_dir in city_dirs:
        city = city_dir.name
        grid_gpkg = city_dir / f"{city}_grid.gpkg"
        if not grid_gpkg.exists():
            logger.warning(f"  {city}: {grid_gpkg.name} not found — skipping inference")
            continue
        grid_gdf = gpd.read_file(grid_gpkg)
        west, south, east, north = grid_gdf.to_crs("EPSG:4326").total_bounds

        tif_path = run_dir / (
            f"{run_dir.name}_{model_type}-{preset}-{task_label}-prediction_{city}.tif"
        )
        infer_roi(
            model=model,
            model_type=model_type,
            embedding_name=embedding_name,
            embedding_dir=embedding_dir,
            bbox=(west, south, east, north),
            output_path=tif_path,
            num_classes=num_classes,
            patch_size=patch_size,
            overlap=overlap,
            batch_size=batch_size,
            device=device,
            dequantize_fn=dequantize_fn,
            normalize=normalize,
            year=year,
            city_name=city,
            title=f"LCZ — {city} — {embedding_name} — {model_type}/{preset}",
            margin_m=margin_m,
        )
        logger.info(f"  {city}: saved {tif_path.name}")
