import json
import os
import numpy as np
import open3d as o3d
import itertools
import yaml

from typing import List, Dict, Any
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from openlex3d.core.categories import get_color


SEGMENTS_ANNOTATION_FILE = "segments_anno.json"
PROMPT_LIST_FILE = "prompt_list.txt"

FEATURES_ALLOWED_FORMATS = ["*.npy", "*.npz"]
CLOUD_ALLOWED_FORMATS = ["*.pcd", "*.ply"]


def load_predicted_features(
    predictions_path: str, voxel_downsampling_size: float = 0.05
):
    # Prepare paths
    pred_root = Path(predictions_path)
    feat_path = list(
        itertools.chain.from_iterable(
            pred_root.glob(pattern) for pattern in FEATURES_ALLOWED_FORMATS
        )
    )[0]
    assert feat_path.exists()

    cloud_path = list(
        itertools.chain.from_iterable(
            pred_root.glob(pattern) for pattern in CLOUD_ALLOWED_FORMATS
        )
    )[0]
    assert cloud_path.exists()

    segment_annotation_path = pred_root / SEGMENTS_ANNOTATION_FILE
    assert segment_annotation_path.exists()

    # Load mask_feats
    pred_feats = np.load(feat_path)

    # TODO Note (matias): this is hacky, why is it needed?
    # It seems like a patch for a corner case
    file_ext = os.path.splitext(feat_path)[1]
    if file_ext == ".npz":
        pred_feats = pred_feats["arr_0"]

    # Get dimensions of predicted features array
    B, D = pred_feats.shape

    # Load predicted cloud
    pred_cloud = o3d.t.io.read_point_cloud(cloud_path)
    points = pred_cloud.point.positions.numpy()
    N = len(points)

    # Load segment annotations
    with open(segment_annotation_path, "r") as f:
        segment_annotations = json.load(f)

    # Note (matias): It would be nice to redesign the code below
    # The main problem I see is that it copies features into a list
    # so we have a dynamic array there. Perhaps there is some array
    # operation we can do instead
    mask_feats = np.full((N, D), -1, dtype=np.float32)

    for group in segment_annotations["segGroups"]:
        objectId = group["objectId"]
        feat = pred_feats[objectId]
        for segment in group["segments"]:
            if segment < len(mask_feats):
                mask_feats[segment] = feat

    filtered_mask_feats = []
    filtered_points = []
    for feat, point in zip(mask_feats, points):
        if not np.all(feat == -1):
            filtered_mask_feats.append(feat)
            filtered_points.append(point)

    filtered_mask_feats = np.array(filtered_mask_feats)
    filtered_points = np.array(filtered_points)

    # Apply mask
    pred_cloud.point.positions = np.array(filtered_points)
    pred_feats = np.array(filtered_mask_feats)

    # Post-processing of the predicted cloud
    # Downsampling the cloud and discarding corresponding features
    downsampled_pcd = pred_cloud.voxel_down_sample(voxel_size=voxel_downsampling_size)
    points = downsampled_pcd.point.positions.numpy()
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(
        pred_cloud.point.positions.numpy()
    )
    distances, indices = nbrs.kneighbors(points)
    pred_feats = pred_feats[indices[:, 0]]
    pred_cloud = downsampled_pcd

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
    reference_cloud: o3d.t.geometry.PointCloud,
    pred_categories=List[str],
    results=Dict[str, Any],
):
    # Prepare outputh path
    output_cloud = Path(output_path, f"{dataset}_{scene}_{algorithm}.pcd")

    # Map categories to colors
    colors = np.array(
        [get_color(category) for category in pred_categories], dtype=float
    )

    # Reconstruct output cloud
    cloud = reference_cloud.clone()
    cloud.point.colors = o3d.core.Tensor(colors / 255.0)

    assert cloud.point.positions.shape[0] > 0
    assert cloud.point.colors.shape[0] > 0

    # Save
    o3d.t.io.write_point_cloud(str(output_cloud), cloud)

    # Prepare results yaml file
    output_results = Path(output_path, f"{dataset}_{scene}_{algorithm}_result.yaml")  # noqa

    with open(str(output_results), "w") as file:
        yaml.dump(results, file, default_flow_style=False)


def load_query_json(query_json_file):
    """
    Loads the query JSON file.
    Expected format:
    {
        "level0": {"cushion": [65, 66], "pillow case": [66, 67]},
        "level1": {"gingham cushion": [65, 66], "gingham pillow case": [66, 67]}
    }
    We'll flatten this to a list of dicts.
    """
    with open(query_json_file, "r") as f:
        queries = json.load(f)
    query_list = []
    for level, subqueries in queries.items():
        for query_text, obj_ids in subqueries.items():
            query_list.append(
                {
                    "query_id": f"{level}_{query_text}",
                    "query_text": query_text,
                    "object_ids": obj_ids,
                }
            )
    return query_list


def load_raw_predictions(predictions_path, scene_name):
    pcd_path = Path(predictions_path) / scene_name / "point_cloud.pcd"
    masks_path = Path(predictions_path) / scene_name / "index.npy"
    features_path = Path(predictions_path) / scene_name / "embeddings.npy"

    pcd = load_pcd(str(pcd_path))
    masks = load_mask_indices(str(masks_path))
    features = load_features(str(features_path))

    return pcd, masks, features


def load_pcd(pcd_file):
    """Load prediction point cloud (n_points, 3)."""
    pcd = o3d.io.read_point_cloud(pcd_file)
    return np.asarray(pcd.points)


def load_mask_indices(mask_file):
    """
    Load the mask indices file.
    Assume it is a file containing n_points integers.
    """
    masks = np.load(mask_file)
    return masks.astype(int)


def load_features(features_file):
    """Load the predicted features (n_objects, n_dim)."""
    return np.load(features_file)
