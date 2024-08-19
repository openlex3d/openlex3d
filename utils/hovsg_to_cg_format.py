import torch
import open3d as o3d
import numpy as np
import os
import json
import glob
import scipy

import argparse
from pathlib import Path
import open_clip

parser = argparse.ArgumentParser()
parser.add_argument("--path")
parser.add_argument("--scene_name")
args = parser.parse_args()

exp_dir = Path(args.path).resolve()
scene_name = args.scene_name

def read_semantic_classes_replica(semantic_info_path, crete_color_map=False):
    with open(semantic_info_path) as f:
        semantic_info = json.load(f)
    class_id_names = {obj["id"]: obj["name"] for obj in semantic_info["classes"]}
    if crete_color_map:
        unique_class_ids = np.unique(list(class_id_names.keys()))
        unique_colors = np.random.rand(len(unique_class_ids), 3)
        class_id_colors = {class_id: unique_colors[i] for i, class_id in enumerate(unique_class_ids)}
        # convert to string
        class_id_colors = {str(k): v.tolist() for k, v in class_id_colors.items()}
        # save class_id_colors to json file to use later
        with open("class_id_colors.json", "w") as f:
            json.dump(class_id_colors, f)
    return class_id_names


def get_text_feats(in_text, clip_model, clip_feat_dim, batch_size=64):
    """
    Get the text features from the CLIP model
    :param in_text (list): the text to get the features from
    :param clip_model (CLIP): the CLIP model
    :param clip_feat_dim (int): the dimension of the CLIP features
    :param batch_size (int): the batch size for the inference
    :return: the text features
    """
    # in_text = ["a {} in the scene.".format(in_text)]
    text_tokens = open_clip.tokenize(in_text).cuda()
    text_id = 0
    text_feats = np.zeros((len(in_text), clip_feat_dim), dtype=np.float32)
    while text_id < len(text_tokens):  # Batched inference.
        batch_size = min(len(in_text) - text_id, batch_size)
        text_batch = text_tokens[text_id : text_id + batch_size]
        with torch.no_grad():
            batch_feats = clip_model.encode_text(text_batch).float()
        batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
        batch_feats = np.float32(batch_feats.cpu())
        text_feats[text_id : text_id + batch_size, :] = batch_feats
        text_id += batch_size
    return text_feats


def get_text_feats_multiple_templates(in_text, clip_model, clip_feat_dim, batch_size=64):
    """
    Get the text features from the CLIP model with text templates
    :param in_text (list): the text to get the features from
    :param clip_model (CLIP): the CLIP model
    :param clip_feat_dim (int): the dimension of the CLIP features
    :param batch_size (int): the batch size for the inference
    :return: the text features
    """
    multiple_templates = [
        "{}",
        "There is the {} in the scene.",
    ]
    mul_tmp = multiple_templates.copy()
    multi_temp_landmarks_other = [x.format(lm) for lm in in_text for x in mul_tmp]
    # format the text with multiple templates except for "background"
    text_feats = get_text_feats(multi_temp_landmarks_other, clip_model, clip_feat_dim)
    # average the features
    text_feats = text_feats.reshape((-1, len(mul_tmp), text_feats.shape[-1]))
    text_feats = np.mean(text_feats, axis=1)
    return text_feats

if __name__ == '__main__':

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14",
        pretrained="/home/buechner/ovsg-dataset/checkpoints/laion2b_s32b_b79k.bin",
        device="cuda",
    )
    clip_feat_dim = 1024
    clip_model.eval()

    # Load the mask_feats.pt
    mask_feats_unnormalized = torch.load(os.path.join(exp_dir, "mask_feats.pt")).float().cuda()
    mask_feats = torch.nn.functional.normalize(mask_feats_unnormalized, p=2, dim=-1).cpu().numpy()

    # save as npz
    np.savez(os.path.join(exp_dir, "clip_features.npz"), mask_feats)

    # construct segments_anno.json
    segments_anno = {"sceneId": exp_dir.stem, "segGroups": []}
    # Load the point cloud
    masked_pcd = o3d.io.read_point_cloud(os.path.join(exp_dir, "predicted_pcd.ply"))
    points = np.asarray(masked_pcd.points)
    masked_pcd.colors = o3d.utility.Vector3dVector(np.asarray(masked_pcd.colors) * 0.1)

    masked_tree = scipy.spatial.cKDTree(np.asarray(masked_pcd.points))
    
    with open("HOVSG_maps/class_id_colors.json", "r") as f:
        colors_map = json.load(f)
        colors_map = {int(k): v for k, v in colors_map.items()}
        color2id = {tuple(v): k for k, v in colors_map.items()}    

    semantic_info_path = os.path.join("/data/replica_v1/", scene_name, "habitat/info_semantic.json")
    class_id_name = read_semantic_classes_replica(semantic_info_path, crete_color_map=True)
    # add background class with id len(class_id_name)+1
    class_id_name[0] = "background"
    labels = list(class_id_name.values())
    labels_id = list(class_id_name.keys())

    text_feats = get_text_feats_multiple_templates(labels, clip_model, clip_feat_dim)
    similarity = torch.nn.functional.cosine_similarity(
        torch.from_numpy(mask_feats).unsqueeze(1), torch.from_numpy(text_feats).unsqueeze(0), dim=2
    )
    sim = similarity.cpu().numpy()

    # find the label index with the highest similarity
    label_indices = similarity.argmax(axis=1)
    print("label_indices: ", label_indices)
    text_labels = np.array([labels[i] for i in label_indices])

    object_pcd_paths = glob.glob(os.path.join(exp_dir, "objects/*.ply"))
    object_pcd_idcs = sorted(enumerate([int(path.split("_")[-1].split(".")[0]) for path in object_pcd_paths]), key=lambda i: i[1])
    object_pcd_paths_sorted = [object_pcd_paths[tuple[0]] for tuple in object_pcd_idcs]
    for obj_idx, object_pcd_path in enumerate(object_pcd_paths_sorted):
        object_pcd = o3d.io.read_point_cloud(object_pcd_path)
        closest_points = masked_tree.query(np.asarray(np.asarray(object_pcd.points)), k=1, distance_upper_bound=0.01, p=2)
        segments_anno["segGroups"].append({"id": obj_idx, "objectId": obj_idx, "label": text_labels[obj_idx], "segments": closest_points[1].tolist()})

    # write out the segments_anno.json
    with open(os.path.join(exp_dir, "segments_anno.json"), "w") as f:
        json.dump(segments_anno, f)

     

