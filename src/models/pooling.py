"""Spatial pooling shared by the pooling-based classification families.

``LinearProbeModel`` and ``MLPModel`` both collapse ``(B, C, H, W)`` to
``(B, C)`` before their head. When the data layer supplies a validity mask
(``--nodata-mode mask``), the pooling has to skip invalid pixels: averaging
them in would fold the nodata fill value into every patch that touches a city
edge or a coastline.
"""

from __future__ import annotations

import torch


def pool_mean_std(
    x: torch.Tensor, valid: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Spatial mean and std of ``(B, C, H, W)``, optionally masked.

    Args:
        x: ``(B, C, H, W)`` features.
        valid: ``(B, 1, H, W)`` float mask, 1 = usable. ``None`` pools everything.

    Returns:
        ``(mean, std)``, each ``(B, C)``. std uses ddof=0, matching
        ``torch.std(unbiased=False)`` and numpy's default.

    Patches with no valid pixel at all fall back to unmasked pooling — there is
    no signal to prefer, and returning NaN would poison the batch.
    """
    if valid is None:
        return x.mean(dim=(-2, -1)), x.std(dim=(-2, -1), unbiased=False)

    m = valid.to(x.dtype)
    n = m.sum(dim=(-2, -1)).clamp_min(1e-6)              # (B, 1)
    mean = (x * m).sum(dim=(-2, -1)) / n                 # (B, C)
    var = (((x - mean[..., None, None]) ** 2) * m).sum(dim=(-2, -1)) / n
    std = var.clamp_min(0).sqrt()

    # All-invalid patches: fall back rather than propagate the clamp artefact.
    empty = valid.sum(dim=(-2, -1)) <= 0                 # (B, 1)
    if empty.any():
        mean = torch.where(empty, x.mean(dim=(-2, -1)), mean)
        std = torch.where(empty, x.std(dim=(-2, -1), unbiased=False), std)
    return mean, std
