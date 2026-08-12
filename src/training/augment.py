"""Training-time augmentation: exact pixel ops only (flips, 90° rotations).

No interpolation is used anywhere so integer masks (0-16 class indices,
-1 nodata) are preserved exactly.

``noise_sigma`` is in units of the **normalized** per-channel standard
deviation, which is what it becomes once inputs are z-scored per channel
(``--normalize channel``). Before that fix the same absolute sigma meant a
14x different perturbation depending on the embedding family — 4.4% of a
channel std for Tessera against 47.4% for AlphaEarth — so it silently
regularized some families far harder than others. Treat the 0.05 default as
untuned: it is the historical value, not a chosen one, and Task 2.3 sweeps it.
"""

from __future__ import annotations

import torch

DEFAULT_NOISE_SIGMA = 0.05
DEFAULT_NOISE_PROB = 0.5


def augment_images(
    images: torch.Tensor,
    valid: torch.Tensor | None = None,
    noise_sigma: float = DEFAULT_NOISE_SIGMA,
    noise_prob: float = DEFAULT_NOISE_PROB,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Random flips, exact 90° rotations and noise over a whole batch.

    Augmentations, drawn independently per sample:
    - Random horizontal flip (p=0.5)
    - Random vertical flip (p=0.5)
    - Random 90° rotation k ∈ {0,1,2,3} (uniform, exact pixel op)
    - Gaussian noise σ=``noise_sigma`` (p=``noise_prob``), image only

    Vectorized over the batch — flips and noise are single batch ops and the
    rotation is done in four buckets — so this belongs on the GPU inside
    ``train_step``, not in a DataLoader collate function. The per-sample Python
    loop it replaces ran at ~670 patches/s on CPU against ~17k/s on GPU.

    Args:
        images: (N, C, H, W) float tensor on any device. H must equal W.
        valid: Optional (N, 1, H, W) validity mask. It gets the SAME geometric
            transform as the image — flipping one without the other silently
            misaligns the mask from the data it describes — and never gets noise.
        noise_sigma: Noise scale in normalized per-channel std units (0 = off).
        noise_prob: Probability of adding noise to a given sample.

    Returns:
        Augmented ``images``, or ``(images, valid)`` when a mask is given.

    Note: the random draws differ from the pre-Phase-1 per-sample loop (one
    batched draw instead of N scalar draws), so a fixed seed does not reproduce
    the old augmentation sequence. The distribution is unchanged.
    """
    n = images.shape[0]
    dev = images.device

    def _geom(x: torch.Tensor, fh, fv, k) -> torch.Tensor:
        x = torch.where(fh.view(-1, 1, 1, 1), x.flip(-1), x)
        x = torch.where(fv.view(-1, 1, 1, 1), x.flip(-2), x)
        out = x.clone()
        for kk in (1, 2, 3):                      # four rotation buckets
            m = k == kk
            out[m] = torch.rot90(x[m], kk, dims=(-2, -1))
        return out

    fh = torch.rand(n, device=dev) < 0.5
    fv = torch.rand(n, device=dev) < 0.5
    k = torch.randint(0, 4, (n,), device=dev)

    out = _geom(images, fh, fv, k)
    if noise_sigma > 0:
        add = (torch.rand(n, device=dev) < noise_prob).view(-1, 1, 1, 1)
        out = out + torch.randn_like(out) * noise_sigma * add

    if valid is None:
        return out
    return out, _geom(valid, fh, fv, k)


def augment_batch(
    images: torch.Tensor,
    masks: torch.Tensor,
    noise_sigma: float = DEFAULT_NOISE_SIGMA,
    noise_prob: float = DEFAULT_NOISE_PROB,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply random flips and exact 90° rotations per sample (segmentation).

    Geometric transforms are applied identically to image and mask; Gaussian
    noise (σ=``noise_sigma``, p=``noise_prob``) is applied to the image only.

    Args:
        images: (N, C, H, W) float tensor on any device.
        masks:  (N, H, W) long tensor, values 0–16 or -1 (nodata).
        noise_sigma: Noise scale in normalized per-channel std units (0 = off).
        noise_prob: Probability of adding noise to a given sample.

    Returns:
        Augmented (images, masks) with the same shapes and dtypes.
    """
    aug_images, aug_masks = [], []
    for img, msk in zip(images, masks):
        if torch.rand(1) < 0.5:
            img = img.flip(-1)
            msk = msk.flip(-1)
        if torch.rand(1) < 0.5:
            img = img.flip(-2)
            msk = msk.flip(-2)
        k = torch.randint(0, 4, (1,)).item()
        if k:
            img = torch.rot90(img, k, dims=(-2, -1))
            msk = torch.rot90(msk, k, dims=(-2, -1))
        if noise_sigma > 0 and torch.rand(1) < noise_prob:
            img = img + torch.randn_like(img) * noise_sigma
        aug_images.append(img)
        aug_masks.append(msk)
    return torch.stack(aug_images), torch.stack(aug_masks)
