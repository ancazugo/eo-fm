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


def load_checkpoint_weights(
    task,
    checkpoint: Path,
    device: torch.device,
    *,
    embedding_name=None,
    channel_mean=None,
    channel_std=None,
) -> Path:
    """Load model weights from a training checkpoint into a task module.

    Accepts both the training-loop checkpoint format
    ``{"model_state_dict", "epoch", <monitor>}`` and a bare state dict.
    Returns the checkpoint path (mirrors run_training_loop's return).

    ``embedding_name`` runs the same provenance guard ``infer_roi`` applies, so
    an evaluate-only run cannot score a checkpoint on a different product with
    the same channel count. ``channel_mean``/``channel_std`` are the statistics
    this run is about to feed the model; when the checkpoint recorded its own
    and they differ, the evaluation would be on mis-normalised inputs, so that
    is refused rather than reported as a number.
    """
    logger.info(f"Loading checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location=device)
    is_wrapped = isinstance(ckpt, dict) and "model_state_dict" in ckpt
    if embedding_name is not None and is_wrapped:
        from datasets.registry import check_checkpoint_provenance
        check_checkpoint_provenance(ckpt, embedding_name)
    if is_wrapped and channel_mean is not None and ckpt.get("channel_mean") is not None:
        for key, ours in (("channel_mean", channel_mean), ("channel_std", channel_std)):
            theirs = np.asarray(torch.as_tensor(ckpt[key]).cpu(), dtype=np.float32)
            if ours is None or theirs.shape != np.shape(ours) or not np.allclose(
                    theirs, np.asarray(ours, dtype=np.float32), rtol=1e-4, atol=1e-6):
                raise ValueError(
                    f"{key} recomputed for this run does not match the one stored in "
                    f"{checkpoint}. The model would be evaluated on inputs normalised "
                    "differently from training; re-run with the training run's "
                    "split/sample/seed flags (or --recompute-stats off)."
                )
    task.model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    task.to(device)
    return checkpoint


def eval_dataset_kwargs(ckpt, embedding_names, *, pipeline: str = "classification") -> dict:
    """``PatchDataset`` kwargs that reproduce a checkpoint's own input pipeline.

    Evaluators that rebuild a model from a checkpoint must feed it what it was
    trained on, or they score a model that never existed. Three things travel:

    * **provenance** -- refused if the checkpoint belongs to another product;
    * **normalisation** -- channel z-scoring with the checkpoint's own stats
      iff it recorded ``normalize == "channel"``; checkpoints from before
      2026-08-11 record nothing and were trained unnormalised;
    * **nodata handling** -- the recorded ``nodata_mode`` if present, else the
      default of the checkpoint's era: patch runs have masked by default since
      the same 2026-08-11 change (so a checkpoint that records ``normalize`` is
      post-change), segmentation runs only since 2026-10-06 (when the key
      started being written).

    ``embedding_names`` is one registry key or a list (fused sources).
    """
    from datasets.registry import check_checkpoint_provenance, get_nodata_predicate

    names = [embedding_names] if isinstance(embedding_names, str) else list(embedding_names)
    wrapped = isinstance(ckpt, dict) and "model_state_dict" in ckpt
    meta = ckpt if wrapped else {}
    if wrapped:
        check_checkpoint_provenance(ckpt, names[0] if len(names) == 1 else names)

    kw: dict = {}
    if meta.get("normalize") == "channel":
        mean, std = meta.get("channel_mean"), meta.get("channel_std")
        if mean is None or std is None:
            raise ValueError("checkpoint says normalize='channel' but carries no stats")
        kw.update(normalize="channel",
                  channel_mean=np.asarray(torch.as_tensor(mean).cpu(), dtype=np.float32),
                  channel_std=np.asarray(torch.as_tensor(std).cpu(), dtype=np.float32))
    mode = meta.get("nodata_mode")
    if mode is None:
        mode = ("mask" if pipeline == "classification" and "normalize" in meta
                else "zero")
    if mode == "mask":
        preds = [get_nodata_predicate(n) for n in names]
        kw.update(nodata_mode="mask",
                  nodata_predicate=preds[0] if len(preds) == 1 else preds)
    logger.info(f"Eval inputs from checkpoint: normalize={kw.get('normalize', 'none')}, "
                f"nodata_mode={mode}")
    return kw


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
    target_res_m: float | None = None,
    aggregate: str = "soft",
    write_native: bool = True,
) -> None:
    """Full-ROI inference for each city: writes one prediction GeoTIFF per city.

    The bbox is taken from {city}_grid.gpkg; the output is named
    ``{run_dir.name}_{model_type}-{preset}-{task_label}-prediction_{city}.tif``.

    ``normalize`` must be the same ``(mean, std)`` the model was trained with,
    or the maps are wrong; callers pass the arrays they fed to the datamodule.

    ``target_res_m``/``aggregate``/``write_native`` are forwarded to
    ``infer_roi``: segmentation runs therefore write a 100 m map plus the
    native 10 m one, rather than only the 10 m one.
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
            target_res_m=target_res_m,
            aggregate=aggregate,
            write_native=write_native,
        )
        logger.info(f"  {city}: saved {tif_path.name}")
