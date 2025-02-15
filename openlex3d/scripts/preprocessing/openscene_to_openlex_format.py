import glob
import itertools
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import open3d as o3d
import yaml
from omegaconf import DictConfig
from sklearn.neighbors import NearestNeighbors

from openlex3d.core.categories import get_color

PROMPT_LIST_FILE = "prompt_list.txt"

CLOUD_ALLOWED_FORMATS = ["*.pcd", "*.ply"]
FEATURE_FILE = "embeddings.npy"
INDEX_FILE = "index.npy"


if __name__ == "__main__":
    scene_path = sys.argv[1]

    coords_file = glob.glob(scene_path + "/*coords.npy")[0]
    feats_file = glob.glob(scene_path + "/*feat_3d.npy")[0]
    fusion_cloud_file = glob.glob(scene_path + "/*fusion.ply")[0]
    input_cloud_file = glob.glob(scene_path + "/*input.ply")[0]
    
    scale = 0.05

    coords = np.load(coords_file)
    feats = np.load(feats_file)
    fusion_cloud = o3d.t.io.read_point_cloud(fusion_cloud_file)
    input_cloud = o3d.t.io.read_point_cloud(input_cloud_file)

    metric_points = fusion_cloud.point.positions.numpy() * 0.05
    fusion_cloud.point.positions = o3d.core.Tensor(metric_points)
    input_cloud.point.positions = o3d.core.Tensor(metric_points)
    
    mask_feats = defaultdict(list)
    for i, feat in enumerate(feats):
        mask_id = coords[i, 3]
        mask_feats[mask_id].append(feat)
    
    averaged_mask_feats = defaultdict(list)
    for mask_id, feats in mask_feats.items():
        averaged_mask_feats[mask_id] = np.mean(np.stack(mask_feats[mask_id]), axis=0)
    pred_feats_mask = np.stack(list(averaged_mask_feats.values()))

    np.save(scene_path + "/embeddings.npy", pred_feats_mask)
    np.save(scene_path + "/index.npy", coords[:, 3])
    o3d.t.io.write_point_cloud(scene_path + "/fusion.ply", fusion_cloud)
    o3d.t.io.write_point_cloud(scene_path + "/input.ply", input_cloud)
    