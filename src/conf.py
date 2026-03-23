"""Configuration dataclasses for the eo-fm pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EmbeddingConfig:
    """Configuration for embedding dataset."""

    name: str  # Registry key: "tessera", "alpha_earth", "seamless"
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
    stride: float | None = None  # Stride for GridGeoSampler (val/test); defaults to patch_size
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
    # RandomForest / ExtraTrees hyperparameters
    n_estimators: int = 100
    # LightGBM hyperparameters
    num_leaves: int = 31
    lgbm_learning_rate: float = 0.1
    min_child_samples: int = 20
    # XGBoost hyperparameters
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    # Logistic Regression hyperparameters
    logreg_C: float = 1.0
    # Cross-validation
    cv_folds: int = 5


@dataclass
class TrainConfig:
    """Configuration for pure-PyTorch training."""

    task: str = "classification"  # "classification" or "segmentation"
    model: str = "resnet18"  # Classification: any timm name (resnet18/34/50/101/152, vit_*).
                              # Segmentation: SMP architecture (unet, deeplabv3+, segformer, upernet, dpt).
    backbone: Optional[str] = None  # Segmentation only: SMP encoder backbone.
                                     # resnet18/34/50/101/152, mit_b0-b5 (SegFormer),
                                     # timm-universal-vit_base_patch16_224, etc.
                                     # Defaults to "resnet50" if not set.
    weights: Optional[str] = None  # Pretrained weights: torchgeo weight name
                                    # (e.g. "ResNet50_Weights.LANDSAT_TM_TOA_MOCO"),
                                    # "imagenet" / "true" for ImageNet, or None for
                                    # random initialisation.
    num_classes: int = 17
    lr: float = 1e-3
    max_epochs: int = 50
    output_dir: Optional[str] = None  # Checkpoint save directory


# Backward-compat alias
LightningConfig = TrainConfig


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
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
