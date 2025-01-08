import os
import sys
sys.path.append("/home/christina/git/segment-anything-2")
import hydra
import json
import numpy as np
from omegaconf import DictConfig, OmegaConf
import open3d as o3d
import open_clip
import plyfile
from scipy.spatial import cKDTree
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import BallTree, NearestNeighbors
import torch
import torchmetrics as tm
import plotly.graph_objs as go
import random
import matplotlib.pyplot as plt
import cv2


from utils.eval_utils import read_ply_and_assign_colors_replica

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

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
    valid_depths = np.abs(depths_valid - depth_values) < 0.01
    
    in_fov_result = np.zeros(points_3d.shape[1], dtype=bool)
    
    in_fov_result[valid_points] = valid_depths

    return valid_points, (u[in_fov_result], v[in_fov_result])

def calculate_grid_dimensions(images, grid_size, whitespace, extra_padding):
    rows, cols = grid_size
    total_width = 0
    total_height = 0

    for i in range(rows):
        row_height = 0
        row_width = 0
        for j in range(cols):
            index = i * cols + j
            if index >= len(images):
                break
            img_height, img_width = images[index].shape[:2]
            row_height = max(row_height, img_height)
            row_width += img_width + whitespace

        total_width = max(total_width, row_width - whitespace)
        total_height += row_height + whitespace

    total_height -= whitespace  # Remove extra whitespace after the last row

    # Add extra padding to the total dimensions
    total_width += 2 * extra_padding
    total_height += 2 * extra_padding

    return total_width, total_height

def place_images_on_canvas(canvas, images, grid_size, whitespace, extra_padding):
    rows, cols = grid_size
    y_offset = extra_padding  

    for i in range(rows):
        row_height = 0
        x_offset = extra_padding  
        for j in range(cols):
            index = i * cols + j
            if index >= len(images):
                break

            img = images[index]
            img_height, img_width = img.shape[:2]
            canvas[y_offset:y_offset + img_height, x_offset:x_offset + img_width] = img
            x_offset += img_width + whitespace
            row_height = max(row_height, img_height)

        y_offset += row_height + whitespace


def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True, save_path=None):
    highest_score_index = np.argmax(scores)
    mask = masks[highest_score_index]
    score = scores[highest_score_index]

    white_background_image = np.ones((image.shape[0], image.shape[1], 3), dtype=image.dtype) * 255  # White background
    for i in range(3):
        white_background_image[..., i] = image[..., i]

    # Create a mask for the RGBA image
    masked_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=image.dtype) 
    for i in range(3): 
        masked_image[..., i] = np.where(mask, image[..., i], 255)
    
    return masked_image

 

@hydra.main(version_base=None, config_path="config", config_name="crop_config")
def main(params: DictConfig):

    ###################################################################################################################################################

    print("Loading SAM")
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    sam2_checkpoint = "/home/christina/git/segment-anything-2/checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

    predictor = SAM2ImagePredictor(sam2_model)

    ###################################################################################################################################################

    print(f"Loading Ground Truth PCD: {params.main.dataset} {params.main.scene_name}")
    scene_name = params.main.scene_name 

    semantic_info_path = os.path.join(
    params.main.replica_dataset_gt_path, scene_name, "habitat",
        "info_semantic.json"
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
    output_dir = "segments_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    segments = {}
    for obj_id in unique_object_ids:
        if obj_id == 0:
            continue

        for obj in semantic_info["objects"]:
            # if obj["id"] == obj_id and obj["class_name"] not in ["wall", "floor", "ceiling"]:

            mask = (object_ids == obj_id)
            segment_points = points[mask]

            segment_pcd = o3d.geometry.PointCloud()
            segment_pcd.points = o3d.utility.Vector3dVector(segment_points)
            
            segments[obj_id] = segment_pcd

            file_path = os.path.join(output_dir, f"segment_{obj_id}.ply")
            o3d.io.write_point_cloud(file_path, segment_pcd)
            print(f"Saved segment {obj_id} to {file_path}")
    #####################################################################################################################################################

 
    # image_path = os.path.join(params.main.replica_dataset_traj_path, scene_name, "results", "frames")
    # image_file_list = sorted(os.listdir(image_path))
    # depth_path = os.path.join(params.main.replica_dataset_traj_path, scene_name, "results", "depth")
    # depth_file_list = sorted(os.listdir(depth_path))

    # print(len(image_file_list))
    # print(len(depth_file_list))
    
    # pose_path = os.path.join(params.main.replica_dataset_traj_path, scene_name, "traj.txt")
    # f = open(pose_path, "r")
    # gt_transforms = []
    # for line in f:
    #     line = line.split()
    #     transform = np.float64(np.array(line)).reshape(4,4)
    #     gt_transforms.append(transform)
    # f.close()

    # print(len(gt_transforms))

    # # Intrinsics 
    # fx = 600
    # fy = 600
    # cx = 599.5
    # cy = 339.5
    # # cy = 479.5
    # scaling_factor = 6553.5

    # intrinsic_matrix = np.array([[fx, 0, cx ],
    #                             [0, fy, cy],
    #                             [0, 0, 1]])
    
    # width = 1200
    # height = 680
    # # height = 960
    # scale = 1.5

    # output_path = "outputs/" + params.main.scene_name
    # os.makedirs(output_path, exist_ok=True)

    # # Generate Crops
    # grid_size = (2, 2)
    # resize_width = 300
    # resize_height = 300
    # whitespace = 10
    # extra_padding = 20  # Adjust this value to increase or decrease the padding
    # for id, segment in segments.items():
    #     points = np.array(segment.points)
    #     visible_frames = []
    #     for depth_idx, depth_image in enumerate(depth_file_list):
    #         depth_cv2_image = cv2.imread(os.path.join(depth_path, depth_image), cv2.IMREAD_UNCHANGED)
    #         depth_cv2_image = depth_cv2_image / scaling_factor

    #         in_fov, valid_points_2d =  is_point_in_fov(points, gt_transforms[depth_idx], intrinsic_matrix, width, height, depth_cv2_image)

    #         num_points_in_fov = np.sum(in_fov)
    #         num_points = len(points)
    #         percent_in_view = (num_points_in_fov / num_points) * 100
    #         if percent_in_view > 80:
    #             visible_frames.append((depth_idx, percent_in_view, valid_points_2d))
        
    #     visible_frames.sort(key=lambda x: x[1], reverse=True)

    #     top_4_frames = visible_frames[:4]
    #     cropped_images = []
    #     for index, frame in enumerate(top_4_frames):
    #         u, v = frame[2]

    #         if len(u) == 0 or len(v) == 0:
    #             print(f"Skipping frame due to empty valid_points_2d: {frame[0]}")
    #             continue  

    #         x_min, x_max = np.clip([int(np.min(u)), int(np.max(u))], 0, width)
    #         y_min, y_max = np.clip([int(np.min(v)), int(np.max(v))], 0, height)

    #         x_min_scaled = int((x_min - (x_max - x_min) * (scale - 1) / 2))
    #         x_max_scaled = int((x_max + (x_max - x_min) * (scale - 1) / 2))
    #         y_min_scaled = int((y_min - (y_max - y_min) * (scale - 1) / 2))
    #         y_max_scaled = int((y_max + (y_max - y_min) * (scale - 1) / 2))

    #         x_min_scaled = np.clip(x_min_scaled, 0, width)
    #         x_max_scaled = np.clip(x_max_scaled, 0, width)
    #         y_min_scaled = np.clip(y_min_scaled, 0, height)
    #         y_max_scaled = np.clip(y_max_scaled, 0, height)

    #         frame_name = image_file_list[frame[0]]
    #         img = cv2.imread(os.path.join(image_path, frame_name))
    #         cropped_image = img[y_min_scaled:y_max_scaled, x_min_scaled:x_max_scaled]

    #         if index == 0:
    #             if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
    #                 print(f"Skipped image in segment: {id} ")
    #                 continue
    #             else:
    #                 predictor.set_image(cropped_image)
    #                 adjusted_u = u - x_min_scaled
    #                 adjusted_v = v - y_min_scaled
    #                 input_points = [[round(x), round(y)] for x, y in zip(adjusted_u, adjusted_v)]
    #                 if len(input_points) > 20:
    #                     input_points = random.sample(input_points, 20)
    #                 else:
    #                     input_points = input_points

    #                 input_labels = np.ones(len(input_points), dtype=int) 
    #                 masks, scores, logits = predictor.predict(
    #                     point_coords=input_points,
    #                     point_labels=input_labels,
    #                     multimask_output=True,
    #                 )
    #                 masked_image = show_masks(cropped_image, masks, scores, input_labels=input_labels, borders=False)
                
    #                 cropped_images.append(masked_image)
    #         else:
    #             cropped_images.append(cropped_image)
                
    #     # Calculate canvas size based on actual image dimensions and add extra padding
    #     total_width, total_height = calculate_grid_dimensions(cropped_images, grid_size, whitespace, extra_padding)
    #     canvas = np.full((total_height, total_width, 3), (255, 255, 255), dtype=np.uint8)
    #     place_images_on_canvas(canvas, cropped_images, grid_size, whitespace, extra_padding)

    #     cv2.imwrite(os.path.join(output_path, "object_" + str(id) + ".jpg"), canvas)



if __name__ == "__main__":
    main()

