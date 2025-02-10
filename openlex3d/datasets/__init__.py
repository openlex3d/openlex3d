import logging
import open3d as o3d
import numpy as np

from omegaconf import DictConfig
from importlib import import_module
from sklearn.neighbors import BallTree
from pathlib import Path


# Define constants used across the datasets
GT_VISIBLE_CLOUD_FILE = "gt_visible_cloud.pcd"
GT_CATEGORIES_FILE = "gt_categories.json"

# Data association threshold for BallTree data association
# Used to assign instance labels from the ground truth to
# the visible ground truth
GT_DATA_ASSOCIATION_THR = 0.05  # meters

logger = logging.getLogger(__name__)


def load_dataset(config: DictConfig, load_openlex3d: bool = False):
    dataset_name = config.name
    logger.info(f"Loading dataset [{config.name}]," f" scene [{config.scene}]")

    assert config.name, "dataset.name not defined, check your config"
    assert config.scene, "dataset.scene not defined, check your config"
    assert config.path, "dataset.path not defined, check your config"

    dataset_module = import_module(f"openlex3d.datasets.{dataset_name}")
    gt_cloud, original_gt_labels = dataset_module.load_dataset(
        name=config.name,
        scene=config.scene,
        base_path=config.path,
    )

    assert isinstance(gt_cloud, o3d.t.geometry.PointCloud)
    assert isinstance(original_gt_labels, np.ndarray)

    openlex3d_gt_handler = None
    if load_openlex3d:
        # Load 'visible' point cloud (if exists)
        gt_cloud, original_gt_labels = load_openlex3d_visible_cloud()

        # Load openlex3d labels
        openlex3d_gt_handler = load_openlex3d_labels_handler(
            name=config.name, scene=config.scene, base_path=config.openlex3d_path
        )

    return gt_cloud, original_gt_labels, openlex3d_gt_handler


def load_openlex3d_visible_cloud(
    config: DictConfig, gt_cloud: o3d.t.geometry.PointCloud, gt_labels: np.ndarray
):
    assert (
        config.openlex3d_path
    ), "dataset.openlex3d_path not defined, check your config"

    try:
        # We try to check if a visible point cloud exists
        # Some datasets do not need it
        gt_visible_cloud = load_openlex3d_cloud(
            name=config.name, scene=config.scene, base_path=config.openlex3d_path
        )

        # Associate points from ground truth cloud to visible ground truth cloud
        gt_points = gt_cloud.point.positions.numpy()
        gt_visible_points = gt_visible_cloud.point.positions.numpy()

        # We use BallTree data association
        ball_tree = BallTree(gt_points)
        distances, indices = ball_tree.query(gt_visible_points, k=1)
        gt_visible_labels = np.full(gt_visible_points.shape[0], -1)

        mask_valid = distances.flatten() < GT_DATA_ASSOCIATION_THR
        gt_visible_labels[mask_valid] = gt_labels[indices.flatten()[mask_valid]]
        return gt_visible_cloud, gt_visible_labels

    except Exception:
        # If the visible point cloud does not exist, we return the original cloud and labels
        return gt_cloud, gt_labels


def load_openlex3d_cloud(name: str, scene: str, base_path: str):
    # Load openlex3d dataset data
    # Read visible cloud
    openlex3d_root = Path(base_path, name, scene)
    visible_cloud_path = openlex3d_root / GT_VISIBLE_CLOUD_FILE
    assert visible_cloud_path.exists()
    gt_visible_cloud = o3d.t.io.read_point_cloud(str(visible_cloud_path))
    assert gt_visible_cloud.point.positions.shape[0] > 0

    return gt_visible_cloud


def load_openlex3d_labels_handler(name: str, scene: str, base_path: str):
    # Load openlex3d dataset data
    openlex3d_root = Path(base_path, name, scene)
    gt_labels_path = openlex3d_root / GT_CATEGORIES_FILE
    assert gt_labels_path.exists()

    from openlex3d.core.categories import CategoriesHandler  # noqa

    openlex3d_gt_handler = CategoriesHandler(str(gt_labels_path))

    return openlex3d_gt_handler
