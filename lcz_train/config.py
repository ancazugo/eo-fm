"""T-config — training-harness configuration (pydantic, YAML-serialisable).

Mirrors ``lcz_labels.config``'s conventions (config hash stamped into every
checkpoint/report, YAML round-trip) so the two packages feel like one system.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class TrainConfig(BaseModel):
    """Single-experiment configuration consumed by :mod:`lcz_train.train`."""

    exp_id: str = "A1"
    embedding: str = "tesserav1.1_global"
    years: list[int] = Field(default_factory=lambda: [2025])
    pixel_res_m: float = 10.0

    # T2 dataset params
    min_conf: float = 0.5
    erosion_px: int = 1
    window_px: int = 128
    samples_per_epoch: int | None = None
    attention_k: int = 256
    use_ucp_features: bool = False

    # T3 loss params
    class_weights: list[float] | None = None
    smoothing_eps: float = 0.0
    conf_gamma: float = 0.0

    # T5 training loop
    steps: int = 1000
    batch_size: int = 32
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    warmup_steps: int = 0
    grad_clip_norm: float = 5.0
    seed: int = 42
    log_every: int = 50

    # paths
    labels_dir: Path = Field(default_factory=lambda: Path("."))
    mosaic_dir: Path = Field(default_factory=lambda: Path("."))
    splits_path: Path = Field(default_factory=lambda: Path("."))
    output_dir: Path = Field(default_factory=lambda: Path("."))

    wandb: bool = False

    model_config = {"extra": "forbid"}

    def _canonical(self) -> dict:
        return json.loads(self.model_dump_json())

    @property
    def config_hash(self) -> str:
        blob = json.dumps(self._canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def to_yaml(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            yaml.safe_dump(self._canonical(), fh, sort_keys=False)
        return path

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        with Path(path).open() as fh:
            data = yaml.safe_load(fh) or {}
        return cls.model_validate(data)
