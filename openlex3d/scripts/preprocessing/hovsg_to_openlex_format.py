import glob
import itertools
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import open3d as o3d
import torch
import yaml
from omegaconf import DictConfig
from sklearn.neighbors import NearestNeighbors

from openlex3d.core.categories import get_color

CLOUD_ALLOWED_FORMATS = ["*.pcd", "*.ply"]
FEATURE_FILE = "embeddings.npy"
INDEX_FILE = "index.npy"


if __name__ == "__main__":
    scene_path = sys.argv[1]

    full_feats_file = glob.glob(scene_path + "/full_feats.pt")[0]
    full_cloud_file = glob.glob(scene_path + "/full_pcd.ply")[0]
    masked_cloud_file = glob.glob(scene_path + "/masked_pcd.ply")[0]
    mask_feats_file = glob.glob(scene_path + "/mask_feats.pt")[0]

    object_files = glob.glob(scene_path + "/objects/pcd_*.ply")
    obj_idcs = [int(obj_file.split("/")[-1].split("_")[-1].split(".")[0]) for obj_file in object_files]

    mask_feats = torch.load(mask_feats_file, weights_only=False).float()
    assert len(obj_idcs) == mask_feats.shape[0]

    objects = dict()
    coords = list()
    colors = list()
    mask_colors = list()
    feats =list()
    mask_idcs = list()
    for obj_idx in range(len(obj_idcs)):
        obj_cloud = o3d.io.read_point_cloud(scene_path + f"/objects/pcd_{obj_idx}.ply")

        objects[obj_idx] = obj_cloud
        obj_points = np.asarray(obj_cloud.points)
        obj_feat = mask_feats[obj_idx].numpy()
        mask_color = np.repeat(np.random.rand(1,3), len(obj_points), axis=0)
        
        assert obj_points.shape[0] == mask_color.shape[0]
        feats.append(obj_feat)
        coords.append(obj_points)
        colors.append(np.asarray(obj_cloud.colors))
        mask_colors.append(mask_color)
        mask_idcs.append(np.ones((obj_points.shape[0], 1)) * obj_idx)

    pred_feats_mask = np.stack(feats)
    coords = np.concatenate(coords, axis=0)
    colors = np.concatenate(colors, axis=0)
    mask_colors = np.concatenate(mask_colors, axis=0)
    mask_idcs = np.concatenate(mask_idcs, axis=0).squeeze(-1)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(coords)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    mask_cloud = o3d.geometry.PointCloud()
    mask_cloud.points = o3d.utility.Vector3dVector(coords)
    mask_cloud.colors = o3d.utility.Vector3dVector(mask_colors)

    print(pred_feats_mask.shape, np.unique(mask_idcs).shape[0], coords.shape, colors.shape, mask_colors.shape, mask_idcs.shape)

    np.save(scene_path + "/embeddings.npy", pred_feats_mask)
    np.save(scene_path + "/index.npy", mask_idcs)
    o3d.io.write_point_cloud(scene_path + "/mask.ply", mask_cloud)
    o3d.io.write_point_cloud(scene_path + "/input.ply", cloud)
    