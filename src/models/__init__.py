"""Model architectures, organised as registered families.

Importing this package populates MODEL_REGISTRY with every family module in
the package. Public API:

    from models import build_model, families_for, get_family, MODEL_REGISTRY
"""

from models.registry import (
    MODEL_REGISTRY,
    ModelFamily,
    build_model,
    families_for,
    get_family,
    register,
    resolve_arch,
)

# Importing the family modules triggers their register() calls.
from models import timm_families  # noqa: F401  (resnet, efficientnet, convnext, densenet, mobilenet, vit)
from models import mlp            # noqa: F401
from models import aspp           # noqa: F401
from models import linear_probe   # noqa: F401
from models import shallow_cnn    # noqa: F401
from models import unet           # noqa: F401
from models import resnet_unet    # noqa: F401
from models import fcn8           # noqa: F401

from models.mlp import MLPModel, build_mlp
from models.aspp import LightASPPHead, build_aspp
from models.linear_probe import LinearProbeModel, build_linear_probe, load_legacy_linear_probe
from models.shallow_cnn import ShallowCNN, build_shallow_cnn
from models.timm_families import build_timm
from models.unet import DoubleConv, UNet, build_unet
from models.resnet_unet import ResNetUNet, build_resnet_unet
from models.fcn8 import FCN8, build_fcn8

__all__ = [
    "MODEL_REGISTRY",
    "ModelFamily",
    "build_model",
    "families_for",
    "get_family",
    "register",
    "resolve_arch",
    "MLPModel",
    "build_mlp",
    "LightASPPHead",
    "build_aspp",
    "LinearProbeModel",
    "build_linear_probe",
    "load_legacy_linear_probe",
    "ShallowCNN",
    "build_shallow_cnn",
    "build_timm",
    "DoubleConv",
    "UNet",
    "build_unet",
    "ResNetUNet",
    "build_resnet_unet",
    "FCN8",
    "build_fcn8",
]
