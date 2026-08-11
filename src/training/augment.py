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
    noise_sigma: float = DEFAULT_NOISE_SIGMA,
    noise_prob: float = DEFAULT_NOISE_PROB,
) -> torch.Tensor:
    """Apply random flips and exact 90° rotations to images only (classification).

    Augmentations applied per sample:
    - Random horizontal flip (p=0.5)
    - Random vertical flip (p=0.5)
    - Random 90° rotation k ∈ {0,1,2,3} (uniform, exact pixel op)
    - Gaussian noise σ=``noise_sigma`` (p=``noise_prob``)

    Args:
        images: (N, C, H, W) float tensor on any device.
        noise_sigma: Noise scale in normalized per-channel std units (0 = off).
        noise_prob: Probability of adding noise to a given sample.

    Returns:
        Augmented images with the same shape.
    """
    aug = []
    for img in images:
        if torch.rand(1) < 0.5:
            img = img.flip(-1)
        if torch.rand(1) < 0.5:
            img = img.flip(-2)
        k = torch.randint(0, 4, (1,)).item()
        if k:
            img = torch.rot90(img, k, dims=(-2, -1))
        if noise_sigma > 0 and torch.rand(1) < noise_prob:
            img = img + torch.randn_like(img) * noise_sigma
        aug.append(img)
    return torch.stack(aug)


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
