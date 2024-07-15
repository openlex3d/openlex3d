import json
import os
import sys

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import open3d as o3d
import open_clip
import plyfile
from scipy.spatial import cKDTree
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import BallTree
import torch
from utils.clip_utils import get_text_feats

def load_feature_map(path, normalize=True):
    """
    Load features map from disk, mask_feats.pt and objects/pcd_i.ply
    :param path: path to feature map
    :param normalize: whether to normalize features
    :return: mask_pcds, mask_feats
    """
    if not os.path.exists(path):
        raise FileNotFoundError("Feature map not found in {}".format(path))
    # load mask_feats
    mask_feats = torch.load(os.path.join(path, "mask_feats.pt")).float()
    if normalize:
        mask_feats = torch.nn.functional.normalize(mask_feats, p=2, dim=-1).cpu().numpy()
    else:
        mask_feats = mask_feats.cpu().numpy()
    print("full pcd feats loaded from disk with shape {}".format(mask_feats.shape))
    # load masked pcds
    if os.path.exists(os.path.join(path, "objects")):
        mask_pcds = []
        number_of_pcds = len(os.listdir(os.path.join(path, "objects")))
        not_found = []
        for i in range(number_of_pcds):
            if os.path.exists(os.path.join(path, "objects", "pcd_{}.ply".format(i))):
                mask_pcds.append(
                    o3d.io.read_point_cloud(os.path.join(path, "objects", "pcd_{}.ply".format(i)))
                )
            else:
                print("masked pcd {} not found in {}".format(i, path))
                not_found.append(i)
        print("number of masked pcds loaded from disk {}".format(len(mask_pcds)))
        # remove masks_feats that are not found
        mask_feats = np.delete(mask_feats, not_found, axis=0)
        print("number of mask_feats loaded from disk {}".format(len(mask_feats)))
        return mask_pcds, mask_feats
    else:
        raise FileNotFoundError("objects directory not found in {}".format(path))
    
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

def create_color_map(class_id_names):
    """
    Create color map for ground truth classes
    :param semantic_info_path: path to semantic info JSON file
    :return color map
    """



    # with open(semantic_info_path) as f:
    #     semantic_info = json.load(f)
    
    # class_id_names = {obj["id"]: obj["name"] for obj in semantic_info["classes"]}
    # unique_class_ids = np.unique(list(class_id_names.keys()))
    unique_colors = np.random.rand(len(class_id_names), 3)
    class_id_colors = {i: unique_colors[i] for i, class_id in enumerate(class_id_names)}
    # convert to string
    class_id_colors = {int(k): v.tolist() for k, v in class_id_colors.items()}
    with open("class_id_colors.json", "w") as f:
            json.dump(class_id_colors, f)
    return class_id_colors


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
    similarity = torch.nn.functional.cosine_similarity(
        torch.from_numpy(mask_feats).unsqueeze(1), torch.from_numpy(text_feats).unsqueeze(0), dim=2
    )
    similarity = similarity.cpu().numpy()
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
    print("label_indices: ", label_indices)
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

def knn_interpolation(cumulated_pc: np.ndarray, full_sized_data: np.ndarray, k):
    """
    Using k-nn interpolation to find labels of points of the full sized pointcloud
    :param cumulated_pc: cumulated pointcloud results after running the network
    :param full_sized_data: full sized point cloud
    :param k: k for k nearest neighbor interpolation
    :return: pointcloud with predicted labels in last column and ground truth labels in last but one column
    """

    labeled = cumulated_pc[cumulated_pc[:, -1] != -1]
    to_be_predicted = full_sized_data.copy()

    ball_tree = BallTree(labeled[:, :3], metric="minkowski")

    knn_classes = labeled[ball_tree.query(to_be_predicted[:, :3], k=k)[1]][:, :, -1].astype(int)
    print("knn_classes: ", knn_classes.shape)

    interpolated = np.zeros(knn_classes.shape[0])

    for i in range(knn_classes.shape[0]):
        interpolated[i] = np.bincount(knn_classes[i]).argmax()

    output = np.zeros((to_be_predicted.shape[0], to_be_predicted.shape[1] + 1))
    output[:, :-1] = to_be_predicted

    output[:, -1] = interpolated

    assert output.shape[0] == full_sized_data.shape[0]

    return output