import json
import os
import numpy as np
import open3d as o3d
import itertools

from pathlib import Path
from sklearn.neighbors import NearestNeighbors


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
