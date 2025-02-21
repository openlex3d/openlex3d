import glob
import itertools
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import open3d as o3d
import torch
import tqdm
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
    print(scene_path)

    coords_file = glob.glob(scene_path + "/*coords.npy")[0]
    feats_file = glob.glob(scene_path + "/*feat_3d.npy")[0]
    fusion_cloud_file = glob.glob(scene_path + "/*fusion.ply")[0]
    input_cloud_file = glob.glob(scene_path + "/*input.ply")[0]
    
    scale = 0.05

    coords = np.load(coords_file)
    feats = np.load(feats_file)
    fusion_cloud = o3d.io.read_point_cloud(fusion_cloud_file)
    input_cloud = o3d.io.read_point_cloud(input_cloud_file)

    metric_points = np.asarray(fusion_cloud.points) * scale
    fusion_cloud.points = o3d.utility.Vector3dVector(metric_points)
    input_cloud.points = o3d.utility.Vector3dVector(metric_points)

    point2mask = np.linspace(0, len(coords), len(coords), dtype=np.int32)

    np.save(scene_path + "/embeddings.npy", feats)
    np.save(scene_path + "/index.npy", point2mask)

    o3d.io.write_point_cloud(scene_path + "/fusion.ply", fusion_cloud)
    o3d.io.write_point_cloud(scene_path + "/input.ply", input_cloud)
    