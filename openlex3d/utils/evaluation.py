import json
import os
import numpy as np
import open3d as o3d
import torch
import itertools

from pathlib import Path
from typing import List
from sklearn.neighbors import NearestNeighbors

from openlex3d.models.base import VisualLanguageEncoder


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


def compute_feature_to_prompt_similarity(
    model: torch.nn.Module,
    features: np.ndarray,
    prompt_list: List[str],
    batch_size: int = 500,
):
    """
    Compute similarity between text and mask_feats
    :param clip_model: CLIP model
    :param clip_feat_dim: CLIP feature dimension
    :param mask_feats: mask features
    :param text: text
    :param templates: whether to use templates
    :return: similarity
    """

    assert isinstance(model, VisualLanguageEncoder)

    text_feats = model.compute_text_features(prompt_list)
    text_features_tensor = torch.from_numpy(text_feats).unsqueeze(0)
    num_feats = features.shape[0]
    similarity = np.zeros((num_feats, text_feats.shape[0]))

    for i in range(0, num_feats, batch_size):
        batch = torch.from_numpy(features[i : i + batch_size]).unsqueeze(1)
        batch_similarity = torch.nn.functional.cosine_similarity(
            batch, text_features_tensor, dim=2
        )
        similarity[i : i + batch_size, :] = batch_similarity.cpu().numpy()
    return similarity


def get_label_from_logits(logits: np.ndarray, text_prompt: List[str], method="max"):
    """
    Convert similarity matrix to labels
    :param similarity: similarity matrix
    :param labels_id: labels id
    :return: labels
    """
    # find the label index with the highest similarity
    if method == "max":
        label_indices = logits.argmax(axis=1)
        # convert label indices to label names
        # labels = np.array([input_labels[i] for i in label_indices])
        labels = np.array(label_indices)
    else:
        raise NotImplementedError(f"method [{method}] not implemented")
    return labels.flatten()


##################################################################################
# def load_caption_map(pcd_path, anno_path):
#     pcd = o3d.io.read_point_cloud(pcd_path)
#     points = np.asarray(pcd.points)

#     with open(anno_path, "r") as f:
#         json_data = json.load(f)

#     labels = [-1] * len(points)

#     for group in json_data["segGroups"]:
#         label = group["objectId"]
#         for segment in group["segments"]:
#             if segment < len(labels):
#                 labels[segment] = label

#     unique_labels = list(set(labels))
#     colors = np.zeros((len(points), 3))
#     label_to_color = {
#         label: [random.random(), random.random(), random.random()]
#         for label in unique_labels
#     }

#     for i in range(len(points)):
#         colors[i] = label_to_color[labels[i]]

#     # Set the colors to the point cloud
#     pcd.colors = o3d.utility.Vector3dVector(colors)

#     return pcd, labels


# def read_gt_classes_replica(gt_labels_path):
#     """
#     Read semantic classes for replica dataset
#     :param gt_labels_path: path to ground truth labels txt file
#     :return: class id names
#     """

#     with open(gt_labels_path, "r") as f:
#         class_id_names = []
#         for line in f:
#             line = line.strip()
#             class_id_names.append(line)

#     return class_id_names


# def text_prompt(clip_model, clip_feat_dim, mask_feats, text):
#     """
#     Compute similarity between text and mask_feats
#     :param clip_model: CLIP model
#     :param clip_feat_dim: CLIP feature dimension
#     :param mask_feats: mask features
#     :param text: text
#     :param templates: whether to use templates
#     :return: similarity
#     """

#     text_feats = get_text_feats(text, clip_model, clip_feat_dim)
#     text_features_tensor = torch.from_numpy(text_feats).unsqueeze(0)
#     num_mask_feats = mask_feats.shape[0]
#     similarity = np.zeros((num_mask_feats, text_feats.shape[0]))
#     batch_size = 500
#     for i in range(0, num_mask_feats, batch_size):
#         mask_batch = torch.from_numpy(mask_feats[i : i + batch_size]).unsqueeze(1)
#         batch_similarity = torch.nn.functional.cosine_similarity(
#             mask_batch, text_features_tensor, dim=2
#         )
#         similarity[i : i + batch_size, :] = batch_similarity.cpu().numpy()
#     return similarity


# def sim_2_label(similarity, input_labels):
#     """
#     Convert similarity matrix to labels
#     :param similarity: similarity matrix
#     :param labels_id: labels id
#     :return: labels
#     """
#     # find the label index with the highest similarity
#     label_indices = similarity.argmax(axis=1)
#     # convert label indices to label names
#     # labels = np.array([input_labels[i] for i in label_indices])
#     labels = np.array(label_indices)
#     return labels
