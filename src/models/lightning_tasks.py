"""Lightning task builders for classification and segmentation."""

from torchgeo.trainers import ClassificationTask, SemanticSegmentationTask

from conf import LightningConfig
from datasets.registry import get_in_channels


def build_task(
    config: LightningConfig,
    embedding_name: str,
) -> ClassificationTask | SemanticSegmentationTask:
    """Build a torchgeo Lightning task from config.

    Args:
        config: Lightning training configuration.
        embedding_name: Name of the embedding (used to look up in_channels).

    Returns:
        A ClassificationTask or SemanticSegmentationTask instance.
    """
    in_channels = get_in_channels(embedding_name)

    if config.task == "classification":
        return ClassificationTask(
            model=config.model,
            in_channels=in_channels,
            num_classes=config.num_classes,
            lr=config.lr,
        )
    elif config.task == "segmentation":
        return SemanticSegmentationTask(
            model="unet",
            backbone=config.model,
            in_channels=in_channels,
            num_classes=config.num_classes,
            lr=config.lr,
        )
    else:
        raise ValueError(f"Unknown task: {config.task}. Choose 'classification' or 'segmentation'.")
