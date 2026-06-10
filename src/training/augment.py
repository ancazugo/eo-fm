"""Training-time augmentation: exact pixel ops only (flips, 90° rotations).

No interpolation is used anywhere so integer masks (0-16 class indices,
-1 nodata) are preserved exactly.
"""

from __future__ import annotations

import torch


def augment_images(images: torch.Tensor) -> torch.Tensor:
    """Apply random flips and exact 90° rotations to images only (classification).

    Augmentations applied per sample:
    - Random horizontal flip (p=0.5)
    - Random vertical flip (p=0.5)
    - Random 90° rotation k ∈ {0,1,2,3} (uniform, exact pixel op)
    - Gaussian noise σ=0.05 (p=0.5)

    Args:
        images: (N, C, H, W) float tensor on any device.

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
        if torch.rand(1) < 0.5:
            img = img + torch.randn_like(img) * 0.05
        aug.append(img)
    return torch.stack(aug)


def augment_batch(
    images: torch.Tensor, masks: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply random flips and exact 90° rotations per sample (segmentation).

    Geometric transforms are applied identically to image and mask; Gaussian
    noise (σ=0.05, p=0.5) is applied to the image only.

    Args:
        images: (N, C, H, W) float tensor on any device.
        masks:  (N, H, W) long tensor, values 0–16 or -1 (nodata).

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
        if torch.rand(1) < 0.5:
            img = img + torch.randn_like(img) * 0.05
        aug_images.append(img)
        aug_masks.append(msk)
    return torch.stack(aug_images), torch.stack(aug_masks)
