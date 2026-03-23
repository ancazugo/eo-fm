"""DataModule for combining embedding and label GeoDatasets."""

import pandas as pd
from shapely.geometry.base import BaseGeometry
from torch.utils.data import DataLoader
from torchgeo.datasets import IntersectionDataset
from torchgeo.datasets.geo import GeoDataset
from torchgeo.samplers import GridGeoSampler, RandomBatchGeoSampler, Units

from conf import SamplerConfig


class EmbeddingLabelDataModule:
    """DataModule that creates IntersectionDataset(embeddings, labels).

    Uses RandomBatchGeoSampler for training and GridGeoSampler for val/test.
    Spatial train/val/test splits via ROI polygons passed to samplers.
    """

    def __init__(
        self,
        embedding_ds: GeoDataset,
        label_ds: GeoDataset | None = None,
        train_label_ds: GeoDataset | None = None,
        val_label_ds: GeoDataset | None = None,
        test_label_ds: GeoDataset | None = None,
        sampler_config: SamplerConfig | None = None,
        task: str = "classification",
        train_roi: BaseGeometry | None = None,
        val_roi: BaseGeometry | None = None,
        test_roi: BaseGeometry | None = None,
        train_toi: pd.Interval | None = None,
        val_toi: pd.Interval | None = None,
        test_toi: pd.Interval | None = None,
        num_workers: int = 4,
        augment: bool = True,
        pred_stride: float | None = None,
    ) -> None:
        """Initialize the DataModule.

        Args:
            embedding_ds: GeoDataset providing embedding rasters (returns "image" key).
            label_ds: GeoDataset providing label rasters (returns "mask" key). Used for
                all splits when no per-split label datasets are provided. Optional;
                if None the dataset is embedding-only (no confusion matrix will be computed).
            train_label_ds: If set, used instead of label_ds for the training split.
            val_label_ds: If set, used instead of label_ds for the validation split.
            test_label_ds: If set, used instead of label_ds for the test split.
            sampler_config: Sampler hyperparameters (patch size, batch size, stride, length).
                Defaults to SamplerConfig() if not provided.
            task: Task type ("classification" or "segmentation"). Controls how
                the mask is converted: classification uses majority vote per patch,
                segmentation keeps the spatial mask.
            train_roi: Shapely geometry (Polygon or MultiPolygon) restricting training samples to this region.
            val_roi: Shapely geometry (Polygon or MultiPolygon) restricting validation samples to this region.
            test_roi: Shapely geometry (Polygon or MultiPolygon) restricting test samples to this region.
            train_toi: pd.Interval restricting training samples to this time range.
            val_toi: pd.Interval restricting validation samples to this time range.
            test_toi: pd.Interval restricting test samples to this time range.
            num_workers: Number of DataLoader worker processes.
            augment: Whether to apply random flips and rotation during training.
        """
        self.embedding_ds = embedding_ds
        self.label_ds = label_ds
        self.train_label_ds = train_label_ds
        self.val_label_ds = val_label_ds
        self.test_label_ds = test_label_ds
        self.cfg = sampler_config or SamplerConfig()
        self.task = task
        self.train_roi = train_roi
        self.val_roi = val_roi
        self.test_roi = test_roi
        self.train_toi = train_toi
        self.val_toi = val_toi
        self.test_toi = test_toi
        self.num_workers = num_workers
        self.augment = augment
        self._augment = augment
        self.pred_stride = pred_stride
        self.train_dataset: IntersectionDataset | GeoDataset | None = None
        self.val_dataset: IntersectionDataset | GeoDataset | None = None
        self.test_dataset: IntersectionDataset | GeoDataset | None = None

        # Augmentation uses exact pixel ops (flip, rot90) — no bilinear interpolation.
        # Kornia's RandomRotation(degrees=90) rotates by a *random* angle in [-90, 90]
        # using bilinear interpolation, which corrupts integer mask values and causes
        # stitching artifacts. We use augment_batch() from train_unet instead.

    def setup(self) -> None:
        """Create per-split datasets: intersection if labels are available, else embedding-only."""
        train_lbl = self.train_label_ds or self.label_ds
        val_lbl = self.val_label_ds or self.label_ds
        test_lbl = self.test_label_ds or self.label_ds

        self.train_dataset = self.embedding_ds & train_lbl if train_lbl else self.embedding_ds
        self.val_dataset = self.embedding_ds & val_lbl if val_lbl else self.embedding_ds
        self.test_dataset = self.embedding_ds & test_lbl if test_lbl else self.embedding_ds

    def _collate_fn(self, batch: list[dict]) -> dict:
        """Collate samples, adapting labels for the task type.

        For classification: reduces spatial mask to a single label per sample
        via majority vote (most frequent non-zero class).
        For segmentation: keeps the spatial mask as-is.
        """
        import torch
        from torchgeo.datasets.utils import stack_samples

        collated = stack_samples(batch)
        if "mask" in collated:
            mask = collated.pop("mask")
            if self.task == "classification":
                # mask shape: (N, 1, H, W) or (N, H, W) -> (N,) majority class
                mask = mask.view(mask.shape[0], -1)  # (N, H*W)
                labels = []
                for i in range(mask.shape[0]):
                    pixels = mask[i]
                    # Filter out nodata (0)
                    valid = pixels[pixels > 0]
                    if len(valid) > 0:
                        labels.append(valid.mode().values.item())
                    else:
                        labels.append(0)  # all-nodata patch → will shift to -1 below
                # Remap from 1-based class IDs to 0-based indices.
                # All-nodata patches: 0 → -1, which matches ignore_index=-1 in the task.
                collated["label"] = torch.tensor(labels, dtype=torch.long) - 1
            else:
                # Segmentation: remap 1-based to 0-based.
                # Nodata pixels (0) → -1, which matches ignore_index=-1 in the task.
                collated["mask"] = mask - 1
        return collated

    def _train_collate_fn(self, batch: list[dict]) -> dict:
        """Collate + apply training augmentations (exact flips and 90° rotations)."""
        collated = self._collate_fn(batch)
        if not self._augment:
            return collated

        import torch
        from train_unet import augment_batch

        image = collated["image"].float()

        if self.task == "segmentation" and "mask" in collated:
            image, mask = augment_batch(image, collated["mask"])
            collated["mask"] = mask
        else:
            # Classification: augment image only (label is a scalar).
            aug_images = []
            for img in image:
                if torch.rand(1) < 0.5:
                    img = img.flip(-1)
                if torch.rand(1) < 0.5:
                    img = img.flip(-2)
                k = torch.randint(0, 4, (1,)).item()
                if k:
                    img = torch.rot90(img, k, dims=(-2, -1))
                if torch.rand(1) < 0.5:
                    img = img + torch.randn_like(img) * 0.05
                aug_images.append(img)
            image = torch.stack(aug_images)

        collated["image"] = image
        return collated

    def train_dataloader(self) -> DataLoader:
        """Return a DataLoader with RandomBatchGeoSampler for training."""
        sampler = RandomBatchGeoSampler(
            self.train_dataset,
            size=self.cfg.patch_size,
            batch_size=self.cfg.batch_size,
            length=self.cfg.length,
            roi=self.train_roi,
            toi=self.train_toi,
            units=Units.PIXELS,
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self._train_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Return a DataLoader with GridGeoSampler for validation."""
        stride = self.cfg.stride or self.cfg.patch_size
        sampler = GridGeoSampler(
            self.val_dataset,
            size=self.cfg.patch_size,
            stride=stride,
            roi=self.val_roi,
            toi=self.val_toi,
            units=Units.PIXELS,
        )
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Return a DataLoader with GridGeoSampler for testing."""
        stride = self.cfg.stride or self.cfg.patch_size
        sampler = GridGeoSampler(
            self.test_dataset,
            size=self.cfg.patch_size,
            stride=stride,
            roi=self.test_roi,
            toi=self.test_toi,
            units=Units.PIXELS,
        )
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def predict_dataloader(self) -> DataLoader:
        """Return a DataLoader over the full embedding ROI for prediction.

        Uses embedding_ds directly (no label intersection) so the GridGeoSampler
        covers the entire bbox, not just where labeled test polygons exist.
        Uses pred_stride if set (enables overlapping patches for smoother predictions).
        """
        stride = self.pred_stride or self.cfg.stride or self.cfg.patch_size
        sampler = GridGeoSampler(
            self.embedding_ds,
            size=self.cfg.patch_size,
            stride=stride,
            roi=self.test_roi,
            toi=self.test_toi,
            units=Units.PIXELS,
        )
        return DataLoader(
            self.embedding_ds,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )
