import numpy as np
import json
from sklearn.neighbors import BallTree


def intersection_over_union(prediction_points, gt_points):
    pass


def map_pred_to_gt(eval_segm, gt_segm, labels, object_class_mapping, mappings, typ=[]):
    if typ == "caption":
        with open(labels, "r") as f:
            json_data = json.load(f)

    intersection = np.zeros(len(mappings))
    mapping_labels = []

    ball_tree = BallTree(eval_segm[:, :3], metric="minkowski")
    distances, indices = ball_tree.query(gt_segm[:, :3], k=1, return_distance=True)

    for gt_point, (distance, index) in zip(gt_segm, zip(distances, indices)):
        eval_point = eval_segm[index[0]]
        if typ == "feature":
            eval_label = labels[int(eval_point[3])]
        else:
            for seg_group in json_data["segGroups"]:
                if seg_group["objectId"] == int(eval_point[3]):
                    eval_label = seg_group["label"].strip()

        gt_id = int(gt_point[3])

        # print(gt_id)
        if gt_id == 0 or gt_id == -100:  # don't count any unlabelled objects
            mapping_labels.append("none")
            continue

        gt_class_info = object_class_mapping.get(gt_id)
        if gt_id not in object_class_mapping or len(gt_class_info["synonyms"]) == 0:
            mapping_labels.append("none")
            continue

        # cl = gt_class_info["class_name"]
        # if cl in ["wall", "floor", "ceiling", "window", "door", "rug", "undefined", "switch", "pillar", "wall-plug"]:
        #     mapping_labels.append("none")
        #     continue

        if distance[0] > 0.05:
            mapping_labels.append("missing")
            intersection[4] += 1
            continue

        level = "incorrect"
        for i, mapping in enumerate(mappings):
            mapped_labels = gt_class_info.get(mapping, [])
            # union[i] += 1
            if eval_label in mapped_labels:
                intersection[i] += 1
                level = mapping
                break

        if level == "incorrect":
            intersection[3] += 1

        mapping_labels.append(level)

    union = sum(1 for label in mapping_labels if label != "none")

    ious = [
        intersection[i]
        / (
            union * 2
            - intersection[0]
            - intersection[1]
            - intersection[2]
            - intersection[3]
            - intersection[4]
        )
        for i in range(len(mappings))
    ]

    accs = [intersection[i] / union for i in range(len(mappings))]

    return ious, accs, mapping_labels


def IOU(
    eval_segm,
    gt_segm,
    labels,
    semantic_info,
    dataset,
    mappings=["synonyms", "vis_sim", "related", "incorrect", "missing"],
    typ=[],
):
    """
    Calculate mean Intersection over Union (IoU) for each mapping type.

    :param eval_segm: 3D point cloud and labels, predicted segmentation
    :param gt_segm: 3D point cloud and labels, ground truth segmentation
    :param labels: list or numpy array, label information, for converting object IDs to text labels
    :param semantic_info_path: str, path to semantic information JSON file, ground truth semantics
    :param mappings: list of str, mappings to calculate IoU for (default: ['synonyms', 'vis_sim', 'related'])
    :param ignore: list of classes to ignore
    :return: list of IoU values for each mapping
    """

    if dataset == "replica":
        # Read ground truth semantic info
        with open(semantic_info) as f:
            semantic_info = json.load(f)

        object_class_mapping = {
            obj["id"]: {
                "class_id": obj["class_id"],
                "class_name": obj["class_name"],
                "synonyms": obj["synonyms"],
                "vis_sim": obj["vis_sim"],
                "related": obj["related"],
            }
            for obj in semantic_info.get("objects", [])
        }
        ious, accs, mapping_labels = map_pred_to_gt(
            eval_segm, gt_segm, labels, object_class_mapping, mappings, typ
        )

    elif dataset == "scannetpp":
        object_class_mapping = semantic_info
        ious, accs, mapping_labels = map_pred_to_gt(
            eval_segm, gt_segm, labels, object_class_mapping, mappings, typ
        )

    return ious, accs, mapping_labels
