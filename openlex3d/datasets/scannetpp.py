import open3d as o3d
import torch
import numpy as np

from pathlib import Path


def load_dataset(name: str, scene: str, base_path: str):
    # Read original ground truth PLY
    # Prepare input paths
    dataset_root = Path(base_path, "prepared_semantics")

    gt_path = dataset_root / f"{scene}.pth"
    assert gt_path.exists()

    # Load cloud
    data = torch.load(str(gt_path), weights_only=False)

    # create ground truth pcd
    coords = data["sampled_coords"]
    colors = data["sampled_colors"]

    # TODO Note: We need to check if we need the visible cloud for this as well
    gt_cloud = o3d.t.geometry.PointCloud()
    gt_cloud.point.positions = o3d.core.Tensor(coords)
    gt_cloud.point.colors = o3d.core.Tensor(colors)

    # get ground truth labels
    gt_instance_labels = data["sampled_instance_anno_id"]

    return gt_cloud, gt_instance_labels


def load_dataset_with_obj_ids(name: str, scene: str, base_path: str):
    # Read original ground truth PLY
    # Prepare input paths
    dataset_root = Path(base_path)

    gt_path = dataset_root / "prepared_semantics" / f"{scene}.pth"
    assert gt_path.exists()

    # Load cloud
    data = torch.load(str(gt_path), weights_only=False)

    # create ground truth coords and obj_ids
    coords = data["sampled_coords"]
    colors = data["sampled_colors"]

    obj_ids = np.array(data["sampled_instance_anno_id"])

    # TODO Note: We need to check if we need the visible cloud for this as well
    gt_cloud = o3d.t.geometry.PointCloud()
    gt_cloud.point.positions = o3d.core.Tensor(coords)
    gt_cloud.point.colors = o3d.core.Tensor(colors)

    return gt_cloud, obj_ids
