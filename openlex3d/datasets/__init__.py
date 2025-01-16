import logging
from omegaconf import DictConfig
from importlib import import_module

# Define constants used across the datasets
GT_VISIBLE_CLOUD_FILE = "gt_visible_cloud.pcd"
GT_CATEGORIES_FILE = "gt_categories.json"

# Data association threshold for BallTree data association
# Used to assign instance labels from the ground truth to
# the visible ground truth
GT_DATA_ASSOCIATION_THR = 0.05  # meters

logger = logging.getLogger(__name__)


def load_dataset(config: DictConfig):
    dataset_name = config.name
    logger.info(f"Loading dataset [{config.name}]," f" scene [{config.scene}]")

    assert config.name, "dataset.name not defined, check your config"
    assert config.scene, "dataset.scene not defined, check your config"
    assert config.path, "dataset.path not defined, check your config"
    assert (
        config.openlex3d_path
    ), "dataset.openlex3d_path not defined, check your config"

    dataset_module = import_module(f"openlex3d.datasets.{dataset_name}")
    gt_cloud, gt_instance_labels = dataset_module.load_dataset(
        name=config.name,
        scene=config.scene,
        base_path=config.path,
        openlex3d_path=config.openlex3d_path,
    )

    # TODO: check if we want to run extra postprocessing common to all datasets

    return gt_cloud, gt_instance_labels
