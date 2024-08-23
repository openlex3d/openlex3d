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
from sklearn.neighbors import BallTree, NearestNeighbors
import torch
import torchmetrics as tm
import plotly.graph_objs as go
import random
import matplotlib.pyplot as plt
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from utils.eval_utils import read_ply_and_assign_colors_replica

def is_point_in_fov(points, transformation_matrix, intrinsic_matrix, image_width, image_height, depth_image):

    points_3d = points.T
    points_3d_homogeneous = np.vstack((points_3d, np.ones(points_3d.shape[1])))
    points_3d_camera_homogeneous = np.linalg.inv(transformation_matrix) @ points_3d_homogeneous
    points_3d_camera = points_3d_camera_homogeneous[:3, :]

    in_front_of_camera = points_3d_camera[2, :] > 0
    
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    
    u = fx * (points_3d_camera[0, :] / points_3d_camera[2, :]) + cx
    v = fy * (points_3d_camera[1, :] / points_3d_camera[2, :]) + cy
    
    in_fov = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)

    valid_points = in_fov & in_front_of_camera
    
    u_valid = u[valid_points].astype(int)
    v_valid = v[valid_points].astype(int)
    depths_valid = points_3d_camera[2, valid_points]
    
    depth_values = depth_image[v_valid, u_valid]
    valid_depths = np.abs(depths_valid - depth_values) < 0.1
    
    in_fov_result = np.zeros(points_3d.shape[1], dtype=bool)
    
    in_fov_result[valid_points] = valid_depths

    return valid_points, (u[in_fov_result], v[in_fov_result])

def calculate_grid_dimensions(image_size, grid_size, whitespace):
    rows, cols = grid_size
    img_width, img_height = image_size

    total_width = cols * img_width + (cols - 1) * whitespace
    total_height = rows * img_height + (rows - 1) * whitespace

    return total_width, total_height

def place_images_on_canvas(canvas, images, grid_size, image_size, whitespace):
    rows, cols = grid_size
    img_width, img_height = image_size

    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            if index >= len(images):
                break

            img = images[index]
            x_offset = j * (img_width + whitespace)
            y_offset = i * (img_height + whitespace)
            canvas[y_offset:y_offset + img_height, x_offset:x_offset + img_width] = img


@hydra.main(version_base=None, config_path="config", config_name="crop_config")
def main(params: DictConfig):

    print(f"Loading Ground Truth PCD: {params.main.dataset} {params.main.scene_name}")
    scene_name = params.main.scene_name 

    semantic_info_path = os.path.join(
    params.main.replica_dataset_gt_path, scene_name, "habitat",
        "info_semantic_extended.json"
    )

    ply_path = os.path.join(params.main.replica_dataset_gt_path, scene_name, "habitat", "mesh_semantic.ply")
    gt_pcd, gt_labels, _, object_ids = read_ply_and_assign_colors_replica(
        ply_path, semantic_info_path
    )

    # Split gt pcd into individual segments
    with open(semantic_info_path) as f:
        semantic_info = json.load(f)

    points = np.asarray(gt_pcd.points)
    unique_object_ids = np.unique(object_ids)
    segments = {}
    for obj_id in unique_object_ids:
        if obj_id == 0:
            continue

        for obj in semantic_info["objects"]:
            if obj["id"] == obj_id and obj["class_name"] not in ["wall", "floor", "ceiling", "rug", "undefined", "window", "pillar", "wall-plug"]:

                mask = (object_ids == obj_id)
                segment_points = points[mask]

                segment_pcd = o3d.geometry.PointCloud()
                segment_pcd.points = o3d.utility.Vector3dVector(segment_points)
                
                segments[obj_id] = segment_pcd
    

 
    image_path = os.path.join(params.main.replica_dataset_traj_path, scene_name, "results", "frames")
    image_file_list = sorted(os.listdir(image_path))
    depth_path = os.path.join(params.main.replica_dataset_traj_path, scene_name, "results", "depth")
    depth_file_list = sorted(os.listdir(depth_path))
    
    pose_path = os.path.join(params.main.replica_dataset_traj_path, scene_name, "traj.txt")
    f = open(pose_path, "r")
    gt_transforms = []
    for line in f:
        line = line.split()
        transform = np.float64(np.array(line)).reshape(4,4)
        gt_transforms.append(transform)
    f.close()

    # Intrinsics 
    fx = 600
    fy = 600
    cx = 599.5
    cy = 339.5
    scaling_factor = 6553.5

    intrinsic_matrix = np.array([[fx, 0, cx ],
                                [0, fy, cy],
                                [0, 0, 1]])
    
    width = 1200
    height = 680
    scale = 1.5

    output_path = "outputs/" + params.main.scene_name
    os.makedirs(output_path, exist_ok=True)

    # Generate Crops
    grid_size = (2, 2)
    resize_width = 300
    resize_height = 300
    whitespace = 5
    for id, segment in segments.items():
        points = np.array(segment.points)
        visible_frames = []
        for depth_idx, depth_image in enumerate(depth_file_list):
            depth_cv2_image = cv2.imread(os.path.join(depth_path, depth_image), cv2.IMREAD_UNCHANGED)
            depth_cv2_image = depth_cv2_image / scaling_factor

            in_fov, valid_points_2d =  is_point_in_fov(points, gt_transforms[depth_idx], intrinsic_matrix, width, height, depth_cv2_image)

            num_points_in_fov = np.sum(in_fov)
            num_points = len(points)
            percent_in_view = (num_points_in_fov / num_points) * 100
            if percent_in_view > 80:
                visible_frames.append((depth_idx, percent_in_view, valid_points_2d))
        
        visible_frames.sort(key=lambda x: x[1], reverse=True)

        top_4_frames = visible_frames[:4]
        cropped_images = []
        for frame in top_4_frames:
            u, v = frame[2]

            if len(u) == 0 or len(v) == 0:
                print(f"Skipping frame due to empty valid_points_2d: {frame[0]}")
                continue  

            x_min, x_max = np.clip([int(np.min(u)), int(np.max(u))], 0, width)
            y_min, y_max = np.clip([int(np.min(v)), int(np.max(v))], 0, height)

            frame_name = image_file_list[frame[0]]
            img = cv2.imread(os.path.join(image_path, frame_name))
            cropped_image = img[y_min:y_max, x_min:x_max]
            resized_image = cv2.resize(cropped_image, (resize_width, resize_height))
            cropped_images.append(resized_image)
        
    
        total_width, total_height = calculate_grid_dimensions((resize_width, resize_height), grid_size, whitespace)
        canvas = np.full((total_height, total_width, 3), (255, 255, 255), dtype=np.uint8)
        place_images_on_canvas(canvas, cropped_images, grid_size, (resize_width, resize_height), whitespace)
        cv2.imwrite(os.path.join(output_path, "object_" + str(id) + ".jpg"), canvas)


if __name__ == "__main__":
    main()

