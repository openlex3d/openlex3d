import numpy as np
import open3d as o3d
import itertools
import yaml

from typing import List, Dict, Any
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from openlex3d.core.categories import get_color

PROMPT_LIST_FILE = "prompt_list.txt"

CLOUD_ALLOWED_FORMATS = ["*.pcd", "*.ply"]
FEATURE_FILE = "embeddings.npy"
INDEX_FILE = "index.npy"


def load_predicted_features(
    predictions_path: str, voxel_downsampling_size: float = 0.05
):
    # Prepare paths
    pred_root = Path(predictions_path)
    feat_path = pred_root / FEATURE_FILE
    assert feat_path.exists(), f"Features file {feat_path} does not exist"

    cloud_path = list(
        itertools.chain.from_iterable(
            pred_root.glob(pattern) for pattern in CLOUD_ALLOWED_FORMATS
        )
    )[0]
    assert cloud_path.exists(), f"Point cloud file {cloud_path} does not exist"

    index_file = pred_root / INDEX_FILE
    assert index_file.exists(), f"Index file {index_file} does not exist"

    # Load predicted cloud
    pred_cloud = o3d.t.io.read_point_cloud(cloud_path)
    points = pred_cloud.point.positions.numpy()
    N = len(points)

    # Post-processing of the predicted cloud
    # Downsampling the cloud and discarding corresponding features
    downsampled_pcd = pred_cloud.voxel_down_sample(voxel_size=voxel_downsampling_size)
    points = downsampled_pcd.point.positions.numpy()
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(
        pred_cloud.point.positions.numpy()
    )
    _, keep_indices = nbrs.kneighbors(points)
    keep_indices = keep_indices.flatten()
    pred_cloud = downsampled_pcd

    # Load features
    pred_feats_mask = np.load(feat_path)  # (n_objects, D)
    pcd_to_mask = np.load(index_file).astype(int)  # (n_points,)

    # Make sure pcd_to_mask indices has the same length as the number of points in original cloud
    assert (
        len(pcd_to_mask) == N
    ), f"Length of index.npy ({len(pcd_to_mask)}) does not match the number of points in the predicted point cloud ({N})"

    # Assign features to points
    pcd_to_mask = pcd_to_mask[keep_indices]
    pred_feats = pred_feats_mask[pcd_to_mask]

    return pred_cloud, pred_feats


def load_prompt_list(base_path: str):
    """
    Read semantic classes for replica dataset
    :param gt_labels_path: path to ground truth labels txt file
    :return: class id names
    """

    prompt_list_path = Path(base_path, PROMPT_LIST_FILE)
    assert prompt_list_path.exists()

    with open(str(prompt_list_path), "r") as f:
        prompt_list = []
        for line in f:
            line = line.strip()
            prompt_list.append(line)

    assert len(prompt_list), "Prompt list is empty!"
    return prompt_list


def save_results(
    output_path: str,
    dataset: str,
    scene: str,
    algorithm: str,
    point_labels: np.ndarray,
    point_categories: np.ndarray,
    reference_cloud: o3d.t.geometry.PointCloud,
    pred_categories=List[str],
    results=Dict[str, Any],
):
    output_path = Path(output_path, algorithm, dataset, scene)

    # Create output path
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare outputh path
    output_cloud = Path(output_path, "point_cloud.pcd")

    # Map categories to colors
    colors = np.array(
        [get_color(category) for category in pred_categories], dtype=float
    )

    # Reconstruct output cloud
    cloud = reference_cloud.clone()
    cloud.point.colors = o3d.core.Tensor(colors)

    assert cloud.point.positions.shape[0] > 0
    assert cloud.point.colors.shape[0] > 0

    # Save
    o3d.io.write_point_cloud(str(output_cloud), cloud.to_legacy())

    # o3d.visualization.draw_geometries([cloud.to_legacy()])

    # Prepare results yaml file
    output_results = Path(output_path, "results.yaml")

    with open(str(output_results), "w") as file:
        yaml.dump(results, file, default_flow_style=False)

    # Save predicted labels for each point
    output_labels = Path(output_path, "point_labels.npy")
    np.save(output_labels, point_labels)

    # Save predicted category for each label of each point
    output_categories = Path(output_path, "point_categories.npy")
    np.save(output_categories, point_categories)
