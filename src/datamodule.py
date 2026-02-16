"""LightningDataModule for combining embedding and label GeoDatasets."""

from typing import Any

import lightning as L
import pandas as pd
from shapely.geometry import Polygon
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
        label_ds: GeoDataset,
        sampler_config: SamplerConfig | None = None,
        train_roi: Polygon | None = None,
        val_roi: Polygon | None = None,
        test_roi: Polygon | None = None,
        train_toi: pd.Interval | None = None,
        val_toi: pd.Interval | None = None,
        test_toi: pd.Interval | None = None,
        num_workers: int = 4,
    ) -> None:
        """Initialize the DataModule.

        Args:
            embedding_ds: GeoDataset providing embedding rasters (returns "image" key).
            label_ds: GeoDataset providing label rasters (returns "mask" key).
            sampler_config: Sampler hyperparameters (patch size, batch size, stride, length).
                Defaults to SamplerConfig() if not provided.
            train_roi: Shapely Polygon restricting training samples to this region.
            val_roi: Shapely Polygon restricting validation samples to this region.
            test_roi: Shapely Polygon restricting test samples to this region.
            train_toi: pd.Interval restricting training samples to this time range.
            val_toi: pd.Interval restricting validation samples to this time range.
            test_toi: pd.Interval restricting test samples to this time range.
            num_workers: Number of DataLoader worker processes.
        """
        super().__init__()
        self.embedding_ds = embedding_ds
        self.label_ds = label_ds
        self.cfg = sampler_config or SamplerConfig()
        self.train_roi = train_roi
        self.val_roi = val_roi
        self.test_roi = test_roi
        self.train_toi = train_toi
        self.val_toi = val_toi
        self.test_toi = test_toi
        self.num_workers = num_workers
        self.dataset: IntersectionDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Create the IntersectionDataset from embeddings and labels."""
        self.dataset = self.embedding_ds & self.label_ds

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
            collate_fn=self.dataset.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Return a DataLoader with GridGeoSampler for validation."""
        sampler = GridGeoSampler(
            self.dataset,
            size=self.cfg.patch_size,
            stride=self.cfg.stride,
            roi=self.val_roi,
            toi=self.val_toi,
            units=Units.PIXELS,
        )
        return DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self.dataset.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Return a DataLoader with GridGeoSampler for testing."""
        sampler = GridGeoSampler(
            self.dataset,
            size=self.cfg.patch_size,
            stride=self.cfg.stride,
            roi=self.test_roi,
            toi=self.test_toi,
            units=Units.PIXELS,
        )
        return DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self.dataset.collate_fn,
        )
