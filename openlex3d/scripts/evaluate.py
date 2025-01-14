#!/usr/bin/env python
# -*- coding: UTF8 -*-
# PYTHON_ARGCOMPLETE_OK

import os

import hydra
import numpy as np
from omegaconf import DictConfig
import open3d as o3d
import open_clip

# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from sklearn.neighbors import NearestNeighbors

from openlex3d import get_path
from openlex3d.utils.eval_utils import (
    load_feature_map,
    read_gt_classes_replica,
    text_prompt,
    sim_2_label,
)
from openlex3d.utils.metric import (
    # frequency_weighted_IU,
    IOU,
    # mean_accuracy,
    # pixel_accuracy,
    # per_class_IU,
)


import logging

from openlex3d.datasets import load_dataset
from openlex3d.models import load_model
from openlex3d.utils.evaluation import load_predicted_features

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path=f"{get_path()}/config", config_name="eval_config"
)
def main(config: DictConfig):
    # Load dataset
    gt_visible_cloud, gt_instance_labels = load_dataset(config.dataset)

    # Run evaluation
    if config.evaluation.type == "features":
        # Load language model
        model, feature_dim = load_model(config.model)

        # Load predicted features
        pred_cloud, pred_feats = load_predicted_features(
            config.evaluation.predictions_path
        )

    elif config.evaluation.type == "caption":
        pass
    else:
        raise NotImplementedError(f"{config.evaluation.type} not supported")

    if config.main.caption_eval:
        print("Running Caption Evaluation")

        # # Load predictions
        # pcd_path = os.path.join(config.main.pred_path, "point_cloud.pcd")
        # json_file_path = os.path.join(config.main.pred_path, "segments_anno.json")
        # pred_pcd, pred_labels = load_caption_map(pcd_path, json_file_path)
        # pred_labels = np.array(pred_labels).reshape(-1, 1)

        # # Voxel downsampling
        # voxel_size = 0.05
        # downsampled_pcd = pred_pcd.voxel_down_sample(voxel_size=voxel_size)
        # points = np.asarray(downsampled_pcd.points)
        # nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(
        #     np.asarray(pred_pcd.points)
        # )
        # distances, indices = nbrs.kneighbors(points)
        # downsampled_labels = pred_labels[indices]
        # downsampled_labels = np.array(downsampled_labels).reshape(-1, 1)

        # # o3d.visualization.draw_geometries([pred_pcd])
        # # points = np.asarray(downsampled_pcd.points)

        # # scatter = go.Scatter3d(
        # # x=points[:, 0],
        # # y=points[:, 1],
        # # z=points[:, 2],
        # # mode='markers',
        # # marker=dict(
        # #         size=2,
        # #         color=downsampled_labels.flatten(),  # Flatten to ensure the correct shape
        # #         colorscale='Viridis',  # Choose a colorscale
        # #         opacity=0.8
        # #     ),
        # #     text=downsampled_labels.flatten(),  # Use labels for hover text
        # #     hoverinfo='text'
        # # )

        # # # Create the layout
        # # layout = go.Layout(
        # #     title='3D Point Cloud with Hover Labels',
        # #     scene=dict(
        # #         xaxis_title='X Axis',
        # #         yaxis_title='Y Axis',
        # #         zaxis_title='Z Axis'
        # #     )
        # # )

        # # # Create the figure
        # # fig = go.Figure(data=[scatter], layout=layout)

        # # # Show the plot
        # # fig.show()

        # # concat coords and labels for predicted pcd
        # coords_pred = np.zeros((len(downsampled_pcd.points), 4))
        # coords_pred[:, :3] = np.asarray(downsampled_pcd.points)
        # coords_pred[:, -1] = downsampled_labels[:, 0]

        # # concat coords and labels for gt pcd
        # gt_labels = gt_labels.reshape(-1, 1)
        # coords_gt = np.zeros((len(gt_pcd.points), 4))
        # coords_gt[:, :3] = np.asarray(gt_pcd.points)
        # coords_gt[:, -1] = gt_labels[:, 0]

        # typ = "caption"
        # print("################ {} ################".format(scene_name))
        # ious, accs, mapping_labels = IOU(
        #     coords_pred, coords_gt, json_file_path, semantic_info_path, typ=typ
        # )
        # print(ious)
        # print(accs)

        # mapping_labels_list = (
        #     mapping_labels.tolist()
        #     if isinstance(mapping_labels, np.ndarray)
        #     else mapping_labels
        # )

        # label_colors = {
        #     "none": [220, 220, 220],  # grey
        #     "synonyms": [34, 139, 34],  # green
        #     "vis_sim": [255, 255, 0],  # yellow
        #     "related": [255, 165, 0],  # orange
        #     "incorrect": [255, 0, 0],  # red
        #     "missing": [0, 0, 0],  # black
        # }
        # colors = np.array(
        #     [label_colors[label] for label in mapping_labels_list], dtype=np.uint8
        # )
        # points = np.column_stack((coords_gt[:, 0], coords_gt[:, 1], coords_gt[:, 2]))
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

        # assert pcd.has_points(), "The point cloud has no points."
        # assert pcd.has_colors(), "The point cloud has no colors."

        # # Save as PCD file
        # pcd_filename = "output_point_cloud_with_labels_cg_1.pcd"
        # o3d.io.write_point_cloud(pcd_filename, pcd)

        # print(f"PCD file saved: {pcd_filename}")

    else:
        print("Running Feature Evaluation")

        # Load CLIP model
        if config.models.clip.type == "ViT-H-14":
            clip_model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-H-14",
                pretrained=str(config.models.clip.checkpoint),
                device=config.main.device,
            )
            clip_feat_dim = 1024
        clip_model.eval()

        pred_path = config.main.pred_path
        for file_name in os.listdir(pred_path):
            if file_name.endswith((".npy", ".npz")):
                feat_path = os.path.join(pred_path, file_name)
            elif file_name.endswith((".pcd", "ply")):
                pcd_path = os.path.join(pred_path, file_name)
        json_file_path = os.path.join(config.main.pred_path, "segments_anno.json")

        print("Loading Feature Map")
        pred_pcd, pred_feats = load_feature_map(pcd_path, feat_path, json_file_path)

        # Downsample point cloud
        # We need to mesh the predicted cloud and the sample only the points that already exist in the ground truth cloud (so we have a 1-to-1 correspondence)
        voxel_size = 0.05
        downsampled_pcd = pred_pcd.voxel_down_sample(voxel_size=voxel_size)
        points = np.asarray(downsampled_pcd.points)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(
            np.asarray(pred_pcd.points)
        )
        distances, indices = nbrs.kneighbors(points)
        downsampled_feats = pred_feats[indices[:, 0]]

        # Get predicted labels
        if config.main.dataset == "replica":
            gt_labels_path = os.path.join(
                config.main.dataset_gt_path,
                config.dataset.scene_name,
                "habitat",
                "unique_labels.txt",  # this is the prompt list for CLIP
            )
            class_id_names = read_gt_classes_replica(gt_labels_path)
            labels = list(class_id_names)
        elif config.main.dataset == "scannetpp":
            unique_labels.discard("dd")  # noqa
            unique_labels.discard("df")  # noqa
            unique_labels.discard("")  # noqa

            labels = list(sorted(unique_labels))  # noqa

            with open("labels.txt", "w") as file:  # This is the prompt list
                for label in labels:
                    file.write(f"{label}\n")

        sim = text_prompt(clip_model, clip_feat_dim, downsampled_feats, labels)
        predicted_labels = sim_2_label(sim, labels)
        predicted_labels = np.array(predicted_labels).reshape(-1, 1)

        points = np.asarray(downsampled_pcd.points)

        # scatter = go.Scatter3d(
        # x=points[:, 0],
        # y=points[:, 1],
        # z=points[:, 2],
        # mode='markers',
        # marker=dict(
        #         size=2,
        #         color=predicted_labels.flatten(),  # Flatten to ensure the correct shape
        #         colorscale='Viridis',  # Choose a colorscale
        #         opacity=0.8
        #     ),
        #     text=predicted_labels.flatten(),  # Use labels for hover text
        #     hoverinfo='text'
        # )

        # # Create the layout
        # layout = go.Layout(
        #     title='3D Point Cloud with Hover Labels',
        #     scene=dict(
        #         xaxis_title='X Axis',
        #         yaxis_title='Y Axis',
        #         zaxis_title='Z Axis'
        #     )
        # )

        # # Create the figure
        # fig = go.Figure(data=[scatter], layout=layout)

        # # Show the plot
        # fig.show()

        # concat coords and labels for predicted pcd
        coords_pred = np.zeros((len(downsampled_pcd.points), 4))
        coords_pred[:, :3] = np.asarray(downsampled_pcd.points)
        coords_pred[:, -1] = predicted_labels[:, 0]

        # concat coords and labels for gt pcd
        gt_labels = gt_labels.reshape(-1, 1)  # noqa
        coords_gt = np.zeros((len(gt_pcd.points), 4))  # noqa
        coords_gt[:, :3] = np.asarray(gt_pcd.points)  # noqa
        coords_gt[:, -1] = gt_labels[:, 0]

        typ = "feature"
        print("################ {} ################".format(scene_name))  # noqa

        if config.main.dataset == "replica":
            ious, accs, mapping_labels = IOU(
                coords_pred,
                coords_gt,
                labels,
                semantic_info_path,  # noqa
                dataset=config.main.dataset,
                typ=typ,
            )
        elif config.main.dataset == "scannetpp":
            ious, accs, mapping_labels = IOU(
                coords_pred,
                coords_gt,
                labels,
                object_class_mapping,  # noqa
                dataset=config.main.dataset,
                typ=typ,
            )

        print(ious)
        print(accs)

        mapping_labels_list = (
            mapping_labels.tolist()
            if isinstance(mapping_labels, np.ndarray)
            else mapping_labels
        )
        label_colors = {
            "none": [220, 220, 220],  # grey
            "synonyms": [34, 139, 34],  # green
            "vis_sim": [255, 255, 0],  # yellow
            "related": [255, 165, 0],  # orange
            "incorrect": [255, 0, 0],  # red
            "missing": [0, 0, 0],  # black
        }
        colors = np.array(
            [label_colors[label] for label in mapping_labels_list], dtype=np.uint8
        )
        points = np.column_stack((coords_gt[:, 0], coords_gt[:, 1], coords_gt[:, 2]))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

        assert pcd.has_points(), "The point cloud has no points."
        assert pcd.has_colors(), "The point cloud has no colors."

        # Save as PCD file
        pcd_filename = "output_point_cloud_cg_scannetpp.pcd"
        o3d.io.write_point_cloud(pcd_filename, pcd)

        print(f"PCD file saved: {pcd_filename}")


if __name__ == "__main__":
    main()
