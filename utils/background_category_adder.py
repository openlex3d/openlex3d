import json
import os
import sys

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import open3d as o3d
import random
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from utils.eval_utils import (
    read_ply_and_assign_colors_replica,
)

from utils import iou
from utils import box

def calculate_bounding_boxes(pcd, labels):

    pcd = np.asarray(pcd.points)
    labels = np.asarray(labels)

    bounding_boxes = {}

    unique_labels = np.unique(labels)

    for label in unique_labels:
        cluster_points = pcd[labels == label]

        points = o3d.utility.Vector3dVector(cluster_points)
        # o3d_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(points)
        o3d_bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(points)

        center = o3d_bounding_box.get_center()
        box_points = np.asarray(o3d_bounding_box.get_box_points())

        merged_array = np.vstack([center, box_points])

        rearranged_array = np.array([
            merged_array[0],
            merged_array[6],
            merged_array[3],
            merged_array[4],
            merged_array[1],
            merged_array[5],
            merged_array[8],
            merged_array[7],
            merged_array[2]
        ])

        bounding_box = box.Box(vertices=rearranged_array)
        
        bounding_boxes[label] = bounding_box

    return bounding_boxes

def calculate_bounding_box_ious(bounding_boxes, json_data):

    new_json = json_data.copy()

    for id1, bbox1 in bounding_boxes.items():
        matching_id = []
        for id2, bbox2 in bounding_boxes.items():
            if id2 != id1:
                loss = iou.IoU(bbox1, bbox2)

                calculated_iou = loss.iou()

                if calculated_iou > 0:
                    matching_id.append(str(id2))

        for sample in new_json['dataset']['samples']:
            if sample['object_id'] == id1:
                sample['background_ids'] = matching_id

    return new_json
    


@hydra.main(version_base=None, config_path="config", config_name="background_config")
def main(params: DictConfig):

    # Get ground truth pcd
    print(f"Loading Ground Truth PCD: {params.main.dataset} {params.main.scene_name}")
    scene_name = params.main.scene_name 

    json_file_path = params.main.json_path
    with open(json_file_path) as f:
        json_data = json.load(f)

    if params.main.dataset == "replica":
        ply_path = os.path.join(params.main.dataset_gt_path, scene_name, "habitat", "mesh_semantic.ply")
        semantic_info_path = os.path.join(
        params.main.dataset_gt_path, scene_name, "habitat",
            "info_semantic.json"
        )
        ply_path = os.path.join(params.main.dataset_gt_path, scene_name, "habitat", "mesh_semantic.ply")
        gt_pcd, gt_labels, _, _ = read_ply_and_assign_colors_replica(
            ply_path, semantic_info_path
        )

        print("Calculating bounding boxes")
        bounding_boxes = calculate_bounding_boxes(gt_pcd, gt_labels)

        print("Calculating IoUs")
        new_json = calculate_bounding_box_ious(bounding_boxes, json_data)

        with open(params.main.updated_json_path, "w") as f:
            json.dump(new_json, f, ensure_ascii=False, indent=4)

        print("Finished saving to json")

    elif params.main.dataset == "scannetpp":
        print(f"Loading Ground Truth PCD: {params.main.dataset} {params.main.scene_name}")
        # TO DO
    elif params.main.dataset == "hm3d":
        print(f"Loading Ground Truth PCD: {params.main.dataset} {params.main.scene_name}")
        # TO DO


if __name__ == "__main__":
    main()


