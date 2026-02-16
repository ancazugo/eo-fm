"""Configuration dataclasses for the eo-fm pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EmbeddingConfig:
    """Configuration for embedding dataset."""

    name: str  # Registry key: "tessera", "google_satellite", "seamless"
    root: str  # Path to embedding data directory


@dataclass
class LabelConfig:
    """Configuration for label dataset."""

    name: str  # e.g. "demuzere_lcz"
    root: str  # Path to label data directory
    remap: Optional[dict[int, int]] = None  # Optional class remapping


@dataclass
class SamplerConfig:
    """Configuration for geo-samplers."""

    patch_size: float = 256  # Patch size in pixels
    length: int = 1000  # Number of patches per epoch (RandomBatchGeoSampler)
    stride: float = 256  # Stride for GridGeoSampler (val/test)
    batch_size: int = 32


@dataclass
class SklearnConfig:
    """Configuration for sklearn pixel classifier."""

    classifier: str = "mlp"  # "mlp" or "random_forest"
    n_samples_per_class: int = 2000
    test_size: float = 0.3
    random_state: int = 411
    # MLP hyperparameters
    hidden_layer_sizes: tuple[int, ...] = (100, 50)
    alpha: float = 0.0001
    learning_rate_init: float = 0.001
    max_iter: int = 300
    # RandomForest hyperparameters
    n_estimators: int = 100
    # Cross-validation
    cv_folds: int = 5


@dataclass
class LightningConfig:
    """Configuration for PyTorch Lightning training."""

    task: str = "classification"  # "classification" or "segmentation"
    model: str = "resnet18"  # Model name or backbone
    num_classes: int = 17
    lr: float = 1e-3
    max_epochs: int = 50
    accelerator: str = "auto"
    devices: int = 1
    output_dir: Optional[str] = None  # Checkpoint save directory


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases."""

    project: str = "eo-fm"
    enabled: bool = True
    sweep_count: int = 20
    sweep_method: str = "bayes"
    sweep_metric: str = "f1"
    sweep_goal: str = "maximize"


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    embedding: EmbeddingConfig
    label: LabelConfig
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    sklearn: SklearnConfig = field(default_factory=SklearnConfig)
    lightning: LightningConfig = field(default_factory=LightningConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
