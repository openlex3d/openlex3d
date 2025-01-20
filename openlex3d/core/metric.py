import numpy as np
import open3d as o3d

from typing import List
from sklearn.neighbors import BallTree
from openlex3d.core.categories import get_categories, CategoriesHandler

# Data association threshold for BallTree data association
# Used to assign instance labels from the ground truth to
# the visible ground truth
GT_DATA_ASSOCIATION_THR = 0.05  # meters


def intersection_over_union(
    pred_cloud: o3d.t.geometry.PointCloud,
    pred_labels: np.ndarray,
    gt_cloud: o3d.t.geometry.PointCloud,
    gt_ids: np.ndarray,
    gt_labels_handler: CategoriesHandler,
    prompt_list: List[str],
    excluded_labels: List[str] = [],
):
    # Find closest match between prediction and ground truth
    ball_tree = BallTree(pred_cloud.point.positions.numpy(), metric="minkowski")
    distances, indices = ball_tree.query(
        gt_cloud.point.positions.numpy(), k=1, return_distance=True
    )

    # Next step aims to find which category the predicted label falls into
    hits_per_category = {category: 0.0 for category in get_categories()}
    pred_categories = []

    for gt_id, distance, index in zip(gt_ids, distances, indices):
        # Get predicted label
        pred_id = pred_labels[index]
        pred_label = prompt_list[pred_id]

        # Case 1: We check if the object ID (stored as gt_label_id) exists in openlex3d_labels
        # If it doesn't, set the predicted category to 'none'
        if not gt_labels_handler.has_object(gt_id):
            pred_categories.append("none")

        # Case 2: We check if the ground truth class is a label that is not considered for evaluation  (wall, floor, ceiling)
        # If it does, predicted category is "none"
        matches = gt_labels_handler.batch_match(
            id=gt_id, query=excluded_labels, category="synonyms"
        )
        if np.sum(matches) > 0:  # this implies a match was found
            pred_categories.append("none")

        # Case 3: Check if there is a valid ground truth
        if distance[0] > GT_DATA_ASSOCIATION_THR:
            hits_per_category["missing"] += 1
            pred_categories.append("missing")
            continue

        # Case 4: Look for other matches (including clutter category)
        matches = gt_labels_handler.match(id=gt_id, query=pred_label, category="all")

        for i, cat in get_categories():
            match = matches[i]
            if match > 0.0:
                hits_per_category[cat] += 1
                pred_categories.append(cat)
                break

    # Compute IoU
    N = len(pred_categories)
    assert N == pred_cloud.point.positions.shape[0]

    return ious, pred_categories  # noqa
