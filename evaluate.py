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
import torchmetrics as tm
import plotly.graph_objs as go

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from utils.eval_utils import (
    load_feature_map,
    read_gt_classes_replica,
    create_color_map,
    text_prompt,
    sim_2_label,
    read_ply_and_assign_colors_replica,
    knn_interpolation,
)
from utils.metric import (
    # frequency_weighted_IU,
    mean_IOU,
    # mean_accuracy,
    # pixel_accuracy,
    # per_class_IU,
)

@hydra.main(version_base=None, config_path="config", config_name="eval_config")
def main(params: DictConfig):
    # Load CLIP model
    if params.models.clip.type == "ViT-H-14":
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14",
            pretrained=str(params.models.clip.checkpoint),
            device=params.main.device,
        )
        clip_feat_dim = 1024
    clip_model.eval()

    # Load features
    masked_pcd, mask_feats = load_feature_map(params.main.feature_path)
    
    # Get semantic classes
    scene_name = params.main.scene_name

    if params.main.dataset == "replica":
        gt_labels_path = os.path.join(
            params.main.replica_dataset_gt_path, scene_name, "habitat", "unique_labels.txt"
        )
        class_id_names = read_gt_classes_replica(gt_labels_path)
        labels = list(class_id_names)

        semantic_info_path = os.path.join(
            params.main.replica_dataset_gt_path, scene_name, "habitat",
            "info_semantic_extended.json"
        )
        color_map = create_color_map(labels)

    # Get predicted labels
    sim = text_prompt(clip_model, clip_feat_dim, mask_feats, labels)
    predicted_labels = sim_2_label(sim, labels)
    predicted_labels = np.array(predicted_labels)

    # Create new pcd 
    pcd = o3d.geometry.PointCloud()

    if params.main.dataset == "replica":
        color_map = {int(k): v for k, v in color_map.items()}
        # create colors for labels based on colors_map
        colors = np.zeros((len(predicted_labels), 3))
        for i, label in enumerate(predicted_labels):
            colors[i] = color_map[label]
    
    ## FOR MASK BASED SEGMENTATION ##
    for i in range(len(masked_pcd)):
        pcd += masked_pcd[i].paint_uniform_color(colors[i])
    
    # o3d.io.write_point_cloud("pred_pcd.ply", pcd)

    # Get ground truth pcd
    if params.main.dataset == "replica":
        ply_path = os.path.join(params.main.replica_dataset_gt_path, scene_name, "habitat", "mesh_semantic.ply")
        gt_pcd, gt_labels, _, _ = read_ply_and_assign_colors_replica(
            ply_path, semantic_info_path
        )

    
    if params.main.dataset == "replica":
        pred_labels = []
        for i in range(len(masked_pcd)):
            pred_labels.append(np.repeat(predicted_labels[i], len(masked_pcd[i].points)))
        pred_labels = np.hstack(pred_labels)

        pred_labels = pred_labels.reshape(-1, 1)
        gt_labels = gt_labels.reshape(-1, 1)

        # concat coords and labels for predicted pcd
        coords_labels = np.zeros((len(pcd.points), 4))
        coords_labels[:, :3] = np.asarray(pcd.points)
        coords_labels[:, -1] = pred_labels[:, 0]

        # downsampled_indices = np.random.choice(len(coords_labels), size=100000, replace=False) 
        # downsampled_coords_labels = coords_labels[downsampled_indices]

        # fig = go.Figure()

        # # Add 3D scatter plot
        # fig.add_trace(go.Scatter3d(
        #     x=downsampled_coords_labels[:, 0],
        #     y=downsampled_coords_labels[:, 1],
        #     z=downsampled_coords_labels[:, 2],
        #     mode='markers',
        #     marker=dict(
        #         size=5,
        #         color=downsampled_coords_labels[:, -1],  # Use predicted labels as colors
        #         colorscale='Viridis',                   # Choose a colorscale
        #         opacity=0.8
        #     ),
        #     text=downsampled_coords_labels[:, -1],       # Hover text will show predicted labels
        #     hoverinfo='text'
        # ))

        # # Customize layout
        # fig.update_layout(
        #     scene=dict(
        #         xaxis=dict(title='X Axis'),
        #         yaxis=dict(title='Y Axis'),
        #         zaxis=dict(title='Z Axis'),
        #     ),
        #     margin=dict(l=0, r=0, b=0, t=0),
        # )

        # # Show plot
        # fig.show()
            

        # concat coords and labels for gt pcd
        coords_gt = np.zeros((len(gt_pcd.points), 4))
        coords_gt[:, :3] = np.asarray(gt_pcd.points)
        coords_gt[:, -1] = gt_labels[:, 0]

        # knn interpolation
        match_pc = knn_interpolation(coords_labels, coords_gt, k=5)
        pred_labels = match_pc[:, -1].reshape(-1, 1)

        ## MATCHING ##
        labels_gt = gt_labels
        label_pred = pred_labels
        assert len(labels_gt) == len(pred_labels)

        # print("Number of unique labels in the GT pcd: ", len(np.unique(labels_gt)))
        # print("Number of unique labels in the pred pcd ", len(np.unique(label_pred)))

    
    ignore = None
    print("################ {} ################".format(scene_name))
    ious, mapping_labels = mean_IOU(label_pred, labels_gt, labels, semantic_info_path, ignore=ignore)
    print(ious)

    mapping_labels_list = mapping_labels.tolist() if isinstance(mapping_labels, np.ndarray) else mapping_labels

    label_colors = {
        'none': [0, 0, 0],       # black
        'synonyms': [0, 255, 0], # green
        'vis_sim': [255, 255, 0],# yellow
        'related': [200, 100, 0],# orange
        'incorrect': [255, 0, 0] # red
    }
    colors = np.array([label_colors[label] for label in mapping_labels_list], dtype=np.uint8)
    print(len(coords_gt))
    # Convert coordinates to Open3D format
    points = np.column_stack((coords_gt[:, 0], coords_gt[:, 1], coords_gt[:, 2]))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    assert pcd.has_points(), "The point cloud has no points."
    assert pcd.has_colors(), "The point cloud has no colors."

    # Save as PCD file
    pcd_filename = 'output_point_cloud_with_labels.pcd'
    o3d.io.write_point_cloud(pcd_filename, pcd)

    print(f'PCD file saved: {pcd_filename}')


    
    # fmiou = frequency_weighted_IU(label_pred, labels_gt, ignore=ignore)
    # print("fmiou: ", fmiou)
    # macc = mean_accuracy(label_pred, labels_gt, ignore=ignore)
    # print("macc: ", macc)
    # pacc = pixel_accuracy(label_pred, labels_gt, ignore=ignore)
    # print("pacc: ", pacc)
    # print("#######################################")

if __name__ == "__main__":
    main()
