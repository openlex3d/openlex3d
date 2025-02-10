import open3d as o3d
import torch
import json
import numpy as np

from pathlib import Path


def load_dataset(name: str, scene: str, base_path: str, openlex3d_path: str):
    # Read original ground truth PLY
    # Prepare input paths
    dataset_root = Path(base_path, name, scene)

    gt_path = dataset_root / f"{scene}.pth"
    assert gt_path.exists()

    # Load cloud
    data = torch.load(str(gt_path))

    # create ground truth pcd
    coords = data["sampled_coords"]
    colors = data["sampled_colors"]
    # obj_ids = data["sampled_instance_labels"]

    # TODO Note: We need to check if we need the visible cloud for this as well
    gt_cloud = o3d.t.geometry.PointCloud()
    gt_cloud.point.positions = o3d.utility.Vector3dVector(coords)
    gt_cloud.point.colors = o3d.utility.Vector3dVector(colors)

    # get ground truth labels
    gt_instance_labels = data["sampled_instance_anno_id"]

    return gt_cloud, gt_instance_labels


def load_mesh_vertices(mesh_file):
    try:
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        return np.asarray(mesh.vertices)
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return None


def load_seg_indices(seg_indices_file):
    """Load the segmentation indices (for mesh vertices) from JSON."""
    try:
        with open(seg_indices_file, "r") as f:
            data = json.load(f)
        return np.array(data["segIndices"])
    except Exception as e:
        print(f"Error loading seg indices: {e}")
        return None


def load_seg_annotations(seg_anno_file):
    """
    Loads the segmentation annotations.
    Returns three lists: obj_ids, labels, segments.
    Each element in segments is a list of segment indices for the object.
    """
    try:
        with open(seg_anno_file, "r") as f:
            seg_anno = json.load(f)["segGroups"]
        obj_ids = [item["id"] for item in seg_anno]
        labels = [item["label"] for item in seg_anno]
        segments = [item["segments"] for item in seg_anno]
        return (obj_ids, labels, segments)
    except Exception as e:
        print(f"Error loading seg annotations: {e}")
        return None, None, None


def load_dataset_gt_files(base_path: str, scene: str):
    mesh_file_path = (
        Path(base_path) / "data" / scene / "scans" / "mesh_aligned_0.05.ply"
    )
    seg_indices_path = Path(base_path) / "data" / scene / "scans" / "segments.json"
    seg_anno_path = Path(base_path) / "data" / scene / "scans" / "segments_anno.json"

    mesh_vertices = load_mesh_vertices(str(mesh_file_path))
    seg_indices = load_seg_indices(str(seg_indices_path))
    seg_anno = load_seg_annotations(str(seg_anno_path))

    return mesh_vertices, seg_indices, seg_anno
