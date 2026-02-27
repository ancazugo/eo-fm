"""LightningDataModule for combining embedding and label GeoDatasets."""

import lightning as L
import pandas as pd
from shapely.geometry.base import BaseGeometry
from torch.utils.data import DataLoader
from torchgeo.datasets import IntersectionDataset
from torchgeo.datasets.geo import GeoDataset
from torchgeo.samplers import GridGeoSampler, RandomBatchGeoSampler, Units

from conf import SamplerConfig


class EmbeddingLabelDataModule(L.LightningDataModule):
    """DataModule that creates IntersectionDataset(embeddings, labels).

    Uses RandomBatchGeoSampler for training and GridGeoSampler for val/test.
    Spatial train/val/test splits via ROI polygons passed to samplers.
    """

    def __init__(
        self,
        embedding_ds: GeoDataset,
        label_ds: GeoDataset | None = None,
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
    ) -> None:
        """Initialize the DataModule.

        Args:
            embedding_ds: GeoDataset providing embedding rasters (returns "image" key).
            label_ds: GeoDataset providing label rasters (returns "mask" key). Optional;
                if None the dataset is embedding-only (no confusion matrix will be computed).
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
        super().__init__()
        self.embedding_ds = embedding_ds
        self.label_ds = label_ds
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
        self.dataset: IntersectionDataset | None = None

        if augment:
            import kornia.augmentation as K
            aug_transforms = [
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
                K.RandomRotation(degrees=90.0, p=0.5),
            ]
            # Segmentation: transform image and mask with the same random params.
            # Classification: label is a scalar — transform image only.
            if task == "segmentation":
                self._aug = K.AugmentationSequential(
                    *aug_transforms, data_keys=["input", "mask"], same_on_batch=False
                )
            else:
                self._aug = K.AugmentationSequential(
                    *aug_transforms, data_keys=["input"], same_on_batch=False
                )

    def setup(self, stage: str | None = None) -> None:
        """Create the dataset: intersection if labels are available, else embedding-only."""
        if self.label_ds is not None:
            self.dataset = self.embedding_ds & self.label_ds
        else:
            self.dataset = self.embedding_ds

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
        """Collate + apply training augmentations (flips, rotation)."""
        collated = self._collate_fn(batch)
        if not self.augment:
            return collated

        import torch
        image = collated["image"].float()

        if self.task == "segmentation" and "mask" in collated:
            # Kornia expects mask as (N, 1, H, W) float; returns same shape.
            mask = collated["mask"].unsqueeze(1).float()
            image, mask = self._aug(image, mask)
            collated["mask"] = mask.squeeze(1).long()
        else:
            image = self._aug(image)

        collated["image"] = image
        return collated

    def train_dataloader(self) -> DataLoader:
        """Return a DataLoader with RandomBatchGeoSampler for training."""
        sampler = RandomBatchGeoSampler(
            self.dataset,
            size=self.cfg.patch_size,
            batch_size=self.cfg.batch_size,
            length=self.cfg.length,
            roi=self.train_roi,
            toi=self.train_toi,
            units=Units.PIXELS,
        )
        return DataLoader(
            self.dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self._train_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Return a DataLoader with GridGeoSampler for validation."""
        stride = self.cfg.stride or self.cfg.patch_size
        sampler = GridGeoSampler(
            self.dataset,
            size=self.cfg.patch_size,
            stride=stride,
            roi=self.val_roi,
            toi=self.val_toi,
            units=Units.PIXELS,
        )
        return DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Return a DataLoader with GridGeoSampler for testing."""
        stride = self.cfg.stride or self.cfg.patch_size
        sampler = GridGeoSampler(
            self.dataset,
            size=self.cfg.patch_size,
            stride=stride,
            roi=self.test_roi,
            toi=self.test_toi,
            units=Units.PIXELS,
        )
        return DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def predict_dataloader(self) -> DataLoader:
        """Return a DataLoader with GridGeoSampler for prediction (same as test)."""
        return self.test_dataloader()
