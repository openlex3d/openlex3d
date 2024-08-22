import json
import os
import sys

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import open3d as o3d
import open_clip
import plyfile
from scipy.spatial import cKDTree, KDTree
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import BallTree
import torch
from utils.clip_utils import get_text_feats
import random


def load_caption_map(pcd_path, anno_path):

    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    with open(anno_path, "r") as f:
        json_data = json.load(f)

    labels = [-1] * len(points)

    for group in json_data['segGroups']:
        label = group['objectId']
        for segment in group['segments']:
            if segment < len(labels):
                labels[segment] = label
    
    unique_labels = list(set(labels))
    colors = np.zeros((len(points), 3))
    label_to_color = {label: [random.random(), random.random(), random.random()] for label in unique_labels}

    for i in range(len(points)):
        colors[i] = label_to_color[labels[i]]

    # Set the colors to the point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd, labels

def load_feature_map(pcd_path, feat_path, json_file_path, normalize=True):
    """
    Load features map from disk, mask_feats.pt and objects/pcd_i.ply
    :param path: path to feature map
    :param normalize: whether to normalize features
    :return: mask_pcds, mask_feats
    """
    if not os.path.exists(feat_path):
        raise FileNotFoundError("Feature map not found in {}".format(feat_path))
    
    # load mask_feats
    feats = np.load(feat_path)

    file_ext = os.path.splitext(feat_path)[1]
    
    if file_ext == '.npz':
        feats = feats['arr_0']
    
    print(np.shape(feats))

    # load segment info
    with open(json_file_path, "r") as f:
        json_data = json.load(f)

    # load pcd
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    
    mask_feats = np.full((len(points), 1024), -1, dtype=np.float32)

    for group in json_data['segGroups']:
        objectId = group['objectId']
        feat = feats[objectId]
        for segment in group['segments']:
            if segment < len(mask_feats):
                mask_feats[segment] = feat
   
    filtered_mask_feats = []
    filtered_points = []
    for feat, point in zip(mask_feats, points):
        if not np.all(feat==-1):
            filtered_mask_feats.append(feat)
            filtered_points.append(point)

    filtered_mask_feats = np.array(filtered_mask_feats)
    filtered_points = np.array(filtered_points)

    pcd.points = o3d.utility.Vector3dVector(filtered_points)

    return pcd, filtered_mask_feats
    
def read_gt_classes_replica(gt_labels_path):
    """
    Read semantic classes for replica dataset
    :param gt_labels_path: path to ground truth labels txt file
    :return: class id names
    """

    with open(gt_labels_path, "r") as f:
        class_id_names = []
        for line in f:
            line = line.strip()
            class_id_names.append(line)

    return class_id_names

def text_prompt(clip_model, clip_feat_dim, mask_feats, text):
    """
    Compute similarity between text and mask_feats
    :param clip_model: CLIP model
    :param clip_feat_dim: CLIP feature dimension
    :param mask_feats: mask features
    :param text: text
    :param templates: whether to use templates
    :return: similarity
    """
   
    text_feats = get_text_feats(text, clip_model, clip_feat_dim)
    text_features_tensor = torch.from_numpy(text_feats).unsqueeze(0)
    num_mask_feats = mask_feats.shape[0]
    similarity = np.zeros((num_mask_feats, text_feats.shape[0]))
    batch_size = 500
    for i in range(0, num_mask_feats, batch_size):
        mask_batch = torch.from_numpy(mask_feats[i:i+batch_size]).unsqueeze(1)
        batch_similarity = torch.nn.functional.cosine_similarity(
            mask_batch, text_features_tensor, dim=2
        )
        similarity[i:i+batch_size, :] = batch_similarity.cpu().numpy()
    return similarity

def sim_2_label(similarity, input_labels):
    """
    Convert similarity matrix to labels
    :param similarity: similarity matrix
    :param labels_id: labels id
    :return: labels
    """
    # find the label index with the highest similarity
    label_indices = similarity.argmax(axis=1)
    # convert label indices to label names
    # labels = np.array([input_labels[i] for i in label_indices])
    labels = np.array(label_indices)
    return labels


# Function to read PLY file and assign colors based on object_id for replica dataset
def read_ply_and_assign_colors_replica(file_path, semantic_info_path):
    """
    Read PLY file and assign colors based on object_id for replica dataset
    :param file_path: path to PLY file
    :param semantic_info_path: path to semantic info JSON file
    :return: point cloud, class ids, point cloud instance, object ids
    """
    # Read PLY file
    plydata = plyfile.PlyData.read(file_path)
    # Read semantic info
    with open(semantic_info_path) as f:
        semantic_info = json.load(f)


    object_class_mapping = {
        obj["id"]: {
            "class_id": obj["class_id"],
            "synonyms": obj["synonyms"],
            "vis_sim": obj["vis_sim"],
            "related": obj["related"]
        }
        for obj in semantic_info["objects"]
    }

    unique_class_ids = {obj["class_id"] for obj in semantic_info["objects"]}
    unique_class_ids = np.array(list(unique_class_ids))

    # Extract vertex data
    vertices = np.vstack([plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]]).T
    # Extract object_id and normalize it to use as color
    face_vertices = plydata["face"]["vertex_indices"]
    object_ids = plydata["face"]["object_id"]
    vertices1 = []
    object_ids1 = []
    for i, face in enumerate(face_vertices):
        vertices1.append(vertices[face])
        object_ids1.append(np.repeat(object_ids[i], len(face)))
    vertices1 = np.vstack(vertices1)
    object_ids1 = np.hstack(object_ids1)

    # set random color for every unique object_id/instance id
    unique_object_ids = np.unique(object_ids)
    instance_colors = np.zeros((len(object_ids1), 3))
    unique_colors = np.random.rand(len(unique_object_ids), 3)
    for i, object_id in enumerate(unique_object_ids):
        instance_colors[object_ids1 == object_id] = unique_colors[i]

    # semantic colors
    class_ids = []
    for object_id in object_ids1:
        if object_id in object_class_mapping.keys():
            # class_ids.append(object_class_mapping[object_id])
            class_ids.append(object_id)
        else:
            class_ids.append(0)
    class_ids = np.array(class_ids)
    print("class_ids: ", class_ids.shape)
    class_colors = np.zeros((len(object_ids1), 3))
    unique_class_colors = np.random.rand(len(unique_class_ids), 3)
    for i, class_id in enumerate(unique_class_ids):
        class_colors[class_ids == class_id] = unique_class_colors[i]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices1)
    pcd.colors = o3d.utility.Vector3dVector(class_colors)
    pcd_instance = o3d.geometry.PointCloud()
    pcd_instance.points = o3d.utility.Vector3dVector(vertices1)
    pcd_instance.colors = o3d.utility.Vector3dVector(instance_colors)
    return pcd, class_ids, pcd_instance, object_ids1

