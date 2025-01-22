import json

import hydra
import numpy as np
from omegaconf import DictConfig
import open3d as o3d

from openlex3d import get_path
from openlex3d.datasets import load_dataset

from openlex3d.dataset_generation import iou
from openlex3d.dataset_generation import box


def calculate_bounding_boxes(pcd, labels):
    pcd_points = pcd.point.positions.numpy()
    # pcd = pcd.to_legacy()
    # pcd = np.asarray(pcd.points)
    labels = np.asarray(labels)

    bounding_boxes = {}

    unique_labels = np.unique(labels)

    for label in unique_labels:
        cluster_points = pcd_points[labels == label]

        points = o3d.utility.Vector3dVector(cluster_points)
        o3d_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(points)
        # o3d_bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        #     points
        # )

        center = o3d_bounding_box.get_center()
        box_points = np.asarray(o3d_bounding_box.get_box_points())

        merged_array = np.vstack([center, box_points])

        rearranged_array = np.array(
            [
                merged_array[0],
                merged_array[6],
                merged_array[3],
                merged_array[4],
                merged_array[1],
                merged_array[5],
                merged_array[8],
                merged_array[7],
                merged_array[2],
            ]
        )

        bounding_box = box.Box(vertices=rearranged_array)

        bounding_boxes[label] = bounding_box

    return bounding_boxes


def calculate_bounding_box_ious(bounding_boxes, json_data):
    new_json = json_data.copy()

    for sample in new_json["dataset"]["samples"]:
        sample["clutter"] = []

    for id1, bbox1 in bounding_boxes.items():
        matching_id = []
        for id2, bbox2 in bounding_boxes.items():
            if id2 != id1:
                loss = iou.IoU(bbox1, bbox2)

                calculated_iou = loss.iou()

                if calculated_iou > 0:
                    if any(
                        sample["object_id"] == id2
                        for sample in new_json["dataset"]["samples"]
                    ):
                        matching_id.append(str(id2))

        if len(matching_id) > 0:
            for sample in new_json["dataset"]["samples"]:
                if sample["object_id"] == id1:
                    sample["clutter"] = matching_id

    return new_json


@hydra.main(
    version_base=None, config_path=f"{get_path()}/config", config_name="clutter_config"
)
def main(config: DictConfig):
    json_file_path = config.jsons.json_path
    with open(json_file_path) as f:
        json_data = json.load(f)

    gt_pcd, gt_labels, _ = load_dataset(config.dataset, load_openlex3d=False)

    print("Calculating bounding boxes")
    bounding_boxes = calculate_bounding_boxes(gt_pcd, gt_labels)

    print("Calculating IoUs")
    new_json = calculate_bounding_box_ious(bounding_boxes, json_data)

    with open(config.jsons.updated_json_path, "w") as f:
        json.dump(new_json, f, ensure_ascii=False, indent=4)

    print("Finished saving to json")


if __name__ == "__main__":
    main()
