from collections import defaultdict
from typing import List

import numpy as np
import open3d as o3d
import torch
import tqdm
from sklearn.neighbors import BallTree

from openlex3d.core.categories import CategoriesHandler, get_categories

# Data association threshold for BallTree data association
# Used to assign instance labels from the ground truth to
# the visible ground truth
GT_DATA_ASSOCIATION_THR = 0.05  # meters
EPS = 1e-8


def intersection_over_union(
    pred_cloud: o3d.t.geometry.PointCloud,
    pred_labels: np.ndarray,
    gt_cloud: o3d.t.geometry.PointCloud,
    gt_ids: np.ndarray,
    gt_labels_handler: CategoriesHandler,
    excluded_labels: List[str] = [],
):
    # Find closest match between prediction and ground truth
    ball_tree = BallTree(pred_cloud.point.positions.numpy(), metric="minkowski")
    distances, indices = ball_tree.query(
        gt_cloud.point.positions.numpy(), k=1, return_distance=True
    )
    indices = indices.flatten()  # balltree at k=1 will output a single index

    # Next step aims to find which category the predicted label falls into
    pred_categories = []

    for gt_id, distance, index in zip(gt_ids, distances, indices):
        # Get predicted label
        pred_label = pred_labels[index]
        gt_id = gt_id.item()

        # Case 1: We check if the object ID (stored as gt_label_id) exists in openlex3d_labels
        # If it doesn't, set the predicted category to 'none'
        if not gt_labels_handler.has_object(gt_id):
            pred_categories.append("none")
            continue

        # Case 2: We check if the ground truth class is a label that is not considered for evaluation  (wall, floor, ceiling)
        # If it does, predicted category is "none"
        matches = gt_labels_handler.batch_category_match(
            id=gt_id, query=excluded_labels, category="synonyms"
        )
        if sum(matches) > 0:  # this implies a match was found
            pred_categories.append("none")
            continue

        # Case 3: Check if there is a valid ground truth
        if distance[0] > GT_DATA_ASSOCIATION_THR:
            pred_categories.append("missing")
            continue

        # Case 4: Look any match (including clutter category)
        matching_category = gt_labels_handler.match(id=gt_id, query=pred_label)
        pred_categories.append(matching_category)

    # Compute IoU
    N = len(pred_categories)
    assert N == gt_cloud.point.positions.shape[0]

    ious = {category: 0.0 for category in get_categories()}
    total_hits = N - pred_categories.count("none")
    for cat, hits in ious.items():
        ious[cat] = pred_categories.count(cat) / total_hits

    return ious, pred_categories  # noqa


def intersection_over_union_normalized(
    pred_cloud: o3d.t.geometry.PointCloud,
    pred_labels: np.ndarray,
    gt_cloud: o3d.t.geometry.PointCloud,
    gt_ids: np.ndarray,
    gt_labels_handler: CategoriesHandler,
    excluded_labels: List[str] = [],
):
    # Find closest match between prediction and ground truth
    ball_tree = BallTree(pred_cloud.point.positions.numpy(), metric="minkowski")
    distances, indices = ball_tree.query(
        gt_cloud.point.positions.numpy(), k=1, return_distance=True
    )
    indices = indices.flatten()  # balltree at k=1 will output a single index

    # Next step aims to find which category the predicted label falls into
    pred_categories = []

    ious = {category: 0 for category in get_categories()}

    unique_ids = set()

    for gt_id, distance, index in zip(gt_ids, distances, indices):
        # Get predicted label
        pred_label = pred_labels[index]
        gt_id = gt_id.item()

        # Get number of points in segment
        count = np.sum(gt_ids == gt_id)

        # Case 1: We check if the object ID (stored as gt_label_id) exists in openlex3d_labels
        # If it doesn't, set the predicted category to 'none'
        if not gt_labels_handler.has_object(gt_id):
            pred_categories.append("none")
            continue

        # Case 2: We check if the ground truth class is a label that is not considered for evaluation  (wall, floor, ceiling)
        # If it does, predicted category is "none"
        matches = gt_labels_handler.batch_category_match(
            id=gt_id, query=excluded_labels, category="synonyms"
        )
        if sum(matches) > 0:  # this implies a match was found
            pred_categories.append("none")
            continue

        # Case 3: Check if there is a valid ground truth
        if distance[0] > GT_DATA_ASSOCIATION_THR:
            pred_categories.append("missing")
            ious["missing"] += 1 / count
            unique_ids.add(gt_id)
            continue

        # Case 4: Look any match (including clutter category)
        matching_category = gt_labels_handler.match(id=gt_id, query=pred_label)
        pred_categories.append(matching_category)

        ious[matching_category] += 1 / count
        unique_ids.add(gt_id)

    # Compute IoU
    num_objects = len(unique_ids)

    for cat, hits in ious.items():
        ious[cat] = float(ious[cat] / num_objects)

    return ious, pred_categories  # noqa


def intersection_over_union_topn(
    pred_cloud: o3d.t.geometry.PointCloud,
    pred_labels: np.ndarray,
    gt_cloud: o3d.t.geometry.PointCloud,
    gt_ids: np.ndarray,
    gt_labels_handler: CategoriesHandler,
    excluded_labels: List[str] = [],
):
    # Find closest match between prediction and ground truth
    ball_tree = BallTree(pred_cloud.point.positions.numpy(), metric="minkowski")
    distances, indices = ball_tree.query(
        gt_cloud.point.positions.numpy(), k=1, return_distance=True
    )
    indices = indices.flatten()  # balltree at k=1 will output a single index

    # Next step aims to find which category the predicted label falls into
    pred_categories = []
    point_labels = []  # Top n labels
    point_label_categories = []  # Top n label categories

    ious = {category: 0 for category in get_categories()}

    unique_ids = set()

    for gt_id, distance, index in zip(gt_ids, distances, indices):
        # Get predicted label
        pred_label = pred_labels[index]
        point_labels.append(pred_label)
        gt_id = gt_id.item()

        # Get number of points in segment
        count = np.sum(gt_ids == gt_id)

        # Case 1: We check if the object ID (stored as gt_label_id) exists in openlex3d_labels
        # If it doesn't, set the predicted category to 'none'
        if gt_id == -100:
            # This is a ScanNet-only case
            pred_categories.append("none")
            point_label_categories.append(["none"] * len(pred_label))
            continue

        if not gt_labels_handler.has_object(gt_id):
            pred_categories.append("none")
            point_label_categories.append(["none"] * len(pred_label))
            continue

        # Case 2: We check if the ground truth class is a label that is not considered for evaluation  (wall, floor, ceiling)
        # If it does, predicted category is "none"
        matches = gt_labels_handler.batch_category_match(
            id=gt_id, query=excluded_labels, category="synonyms"
        )
        if sum(matches) > 0:  # this implies a match was found
            pred_categories.append("none")
            point_label_categories.append(["bg"] * len(pred_label))
            continue

        # Case 3: Check if there is a valid ground truth
        if distance[0] > GT_DATA_ASSOCIATION_THR:
            pred_categories.append("missing")
            point_label_categories.append(["missing"] * len(pred_label))
            ious["missing"] += 1 / count
            unique_ids.add(gt_id)
            continue

        # Case 4: Find the category for the top n labels
        matching_categories = []
        for label in pred_label:
            matching_category = gt_labels_handler.match(id=gt_id, query=label)
            matching_categories.append(matching_category)
        point_label_categories.append(matching_categories)

        # mode_category = mode(matching_categories)
        for category in get_categories():
            if category in matching_categories:
                pred_categories.append(category)
                ious[category] += 1 / count
                break

        # print(category, pred_label, matching_categories, gt_id)

        unique_ids.add(gt_id)

    # Compute IoU
    num_objects = len(unique_ids)
    # print(num_objects)

    for cat, hits in ious.items():
        ious[cat] = float(ious[cat] / num_objects)

    return (
        ious,
        pred_categories,
        np.array(point_labels),
        np.array(point_label_categories),
    )  # noqa


def compute_set_ranking_score(ranks: List[int], set_rank_l=3, set_rank_r=5, max_label_idx=11):
    scores = []
    left_scores, right_scores = [], []
    for rank in ranks:
        left_box_constr = 1 + min((0, (rank - set_rank_l)/(set_rank_l + EPS)))
        right_box_constr = 1-max((0, (rank - set_rank_r)/(max_label_idx - set_rank_r)))
        scores.append(min(left_box_constr, right_box_constr))
        left_scores.append(left_box_constr)
        right_scores.append(right_box_constr)
    return scores, left_scores, right_scores


def set_based_ranking(pred_cloud: o3d.t.geometry.PointCloud,
                        gt_cloud: o3d.t.geometry.PointCloud,
                        gt_ids: np.ndarray,
                        gt_labels_handler: CategoriesHandler,
                        excluded_labels: List[str] = [],
                        logits: np.ndarray = None,
                        prompt_list: List[str] = None):
    
    assert logits is not None, "Logits must be provided for set-based ranking" 
    assert prompt_list is not None and len(prompt_list) > 0, "Prompt list must be provided for set-based ranking"
    ranked_labels = torch.argsort(torch.from_numpy(logits), dim=-1, descending=True)

    # Find closest match between prediction and ground truth
    ball_tree = BallTree(pred_cloud.point.positions.numpy(), metric="minkowski")
    distances, indices = ball_tree.query(
        gt_cloud.point.positions.numpy(), k=1, return_distance=True
    )
    indices = indices.flatten()  # balltree at k=1 will output a single index

    # Next step aims to find which category the predicted label falls into
    overall_scores, synonym_scores, sec_scores = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    
    syn_undershooting_scores, sec_overshooting_scores, sec_undershooting_scores = {}, {}, {}
    syn_inlier_rate, sec_inlier_rate = {}, {}

                
            

    stripped_prompt_list = [label.replace(" ", "") for label in prompt_list]

    for gt_id, distance, index in tqdm.tqdm(
        zip(gt_ids, distances, indices),
        total=len(gt_ids),
        desc="Computing set ranking metrics",
    ):
        gt_id = gt_id.item()
        pred_label_ranking = ranked_labels[index]

        # obtain synonym ranks
        synonyms = gt_labels_handler._get_labels_from_category(gt_id, "synonyms")
        synonym_idcs = [stripped_prompt_list.index(synonym) for synonym in synonyms if synonym in stripped_prompt_list]

        consider_synonyms, consider_secondary = False, False

        rolling_rank = 0
        syn_ranks = []
        if len(synonym_idcs) > 0:
            consider_synonyms = True
            syn_rank_l, syn_rank_r = rolling_rank, len(synonym_idcs) - 1
            rolling_rank += len(synonym_idcs)
            # get the ranks of all synonym_idcs in pred_label_ranking
            for syn_idx in synonym_idcs:
                syn_ranks.append(torch.where(pred_label_ranking == syn_idx)[0].item())
        else:
            syn_rank_l, syn_rank_r = None, None
            consider_synonyms = False

        # obtain depiction-ranks
        depiction = gt_labels_handler._get_labels_from_category(gt_id, "depictions")
        depiction_idcs = [
            stripped_prompt_list.index(depiction_label) for depiction_label in depiction if depiction_label in stripped_prompt_list
        ]

        depiction_ranks = []
        if len(depiction_idcs) > 0:
            consider_depictions = True
            depiction_rank_l, depiction_rank_r = (
                rolling_rank,
                rolling_rank + len(depiction_idcs) - 1,
            )
            rolling_rank += len(depiction_idcs)
            for depiction_idx in depiction_idcs:
                depiction_ranks.append(
                    torch.where(pred_label_ranking == depiction_idx)[0].item()
                )
        else:
            consider_depictions = False
            depiction_rank_l, depiction_rank_r = None, None

        # obtain vis-sim ranks
        vis_sim = gt_labels_handler._get_labels_from_category(gt_id, "vis_sim")
        vis_sim_idcs = [
            stripped_prompt_list.index(vis_sim_label) for vis_sim_label in vis_sim if vis_sim_label in stripped_prompt_list
        ]
        vis_sim_ranks = []
        if len(vis_sim_idcs) > 0:
            consider_vis_sim = True
            vis_sim_rank_l, vis_sim_rank_r = (
                rolling_rank,
                rolling_rank + len(vis_sim_idcs) - 1,
            )
            rolling_rank += len(vis_sim_idcs)
            for vis_sim_idx in vis_sim_idcs:
                vis_sim_ranks.append(
                    torch.where(pred_label_ranking == vis_sim_idx)[0].item()
                )
        else:
            vis_sim_rank_l, vis_sim_rank_r = None, None
            consider_vis_sim = False

        # definition of second-tier category
        if consider_depictions and consider_vis_sim:
            secondary_rank_l = depiction_rank_l
            secondary_rank_r = vis_sim_rank_r
            consider_secondary = True
        elif consider_depictions and not consider_vis_sim:
            secondary_rank_l = depiction_rank_l
            secondary_rank_r = depiction_rank_r
            consider_secondary = True
        elif not consider_depictions and consider_vis_sim:
            secondary_rank_l = vis_sim_rank_l
            secondary_rank_r = vis_sim_rank_r
            consider_secondary = True
        else:
            secondary_rank_l, secondary_rank_r = None, None
            consider_secondary = False            

        secondary_ranks = depiction_ranks + vis_sim_ranks
    
        L = len(stripped_prompt_list) - 1
        if consider_synonyms:
            indiv_syn_scores, indiv_syn_scores_left, indiv_syn_scores_right = compute_set_ranking_score(syn_ranks, syn_rank_l, syn_rank_r, L)
            syn_inlier_rate[gt_id] = sum([1 for score in indiv_syn_scores if score == 1]) / len(indiv_syn_scores)
            syn_undershooting_scores[gt_id] = [score for score in indiv_syn_scores_right if score < 1]
            synonym_scores[gt_id].extend(indiv_syn_scores)
            overall_scores[gt_id].extend(indiv_syn_scores)
        if consider_secondary:
            indiv_sec_scores, indiv_sec_scores_left, indiv_sec_scores_right = compute_set_ranking_score(secondary_ranks, secondary_rank_l, secondary_rank_r, L)
            sec_inlier_rate[gt_id] = sum([1 for score in indiv_sec_scores if score == 1]) / len(indiv_sec_scores)
            # evaluate left box constraint errors
            sec_overshooting_scores[gt_id] = [score for score in indiv_sec_scores_left if score < 1]
            sec_undershooting_scores[gt_id] = [score for score in indiv_sec_scores_right if score < 1]
            sec_scores[gt_id].extend(indiv_sec_scores)
            overall_scores[gt_id].extend(indiv_sec_scores)

    for gt_id in overall_scores.keys():
        if gt_id in list(overall_scores.keys()):
            overall_scores[gt_id] = float(np.nanmean(overall_scores[gt_id]))
        if gt_id in list(synonym_scores.keys()):
            synonym_scores[gt_id] = float(np.nanmean(synonym_scores[gt_id]))
            syn_undershooting_scores[gt_id] = float(np.nanmean(syn_undershooting_scores[gt_id]))
        if gt_id in list(sec_scores.keys()):
            sec_scores[gt_id] = float(np.nanmean(sec_scores[gt_id]))
            sec_overshooting_scores[gt_id] = float(np.nanmean(sec_overshooting_scores[gt_id]))
            sec_undershooting_scores[gt_id] = float(np.nanmean(sec_undershooting_scores[gt_id]))

    results = {
        "overall": float(np.nanmean(list(overall_scores.values()))),
        "synonyms": float(np.nanmean(list(synonym_scores.values()))),
        "secondary": float(np.nanmean(list(sec_scores.values()))),
        "synonym_undershooting": float(np.nanmean(list(syn_undershooting_scores.values()))),
        "secondary_overshooting": float(np.nanmean(list(sec_overshooting_scores.values()))),
        "secondary_undershooting": float(np.nanmean(list(sec_undershooting_scores.values()))),
        "synonym_inlier_rate": float(np.nanmean(list(syn_inlier_rate.values()))),
        "secondary_inlier_rate": float(np.nanmean(list(sec_inlier_rate.values()))),
    }

    return results
