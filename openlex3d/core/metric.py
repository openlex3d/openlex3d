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

# -----------------------------------
# Global evaluation parameters for AP
# -----------------------------------
MIN_REGION_SIZE = 1
opt = {}
opt["overlap_thresholds"] = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
opt["min_region_sizes"] = np.array([MIN_REGION_SIZE])
opt["distance_threshes"] = np.array([float("inf")])
opt["distance_confs"] = np.array([-float("inf")])

# For our case we only have one semantic class:
CLASS_LABELS = ["object"]
VALID_CLASS_IDS = np.array([1])
ID_TO_LABEL = {1: "object"}
LABEL_TO_ID = {"object": 1}


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


def compute_ap(matches):
    overlaps = opt["overlap_thresholds"]
    min_region_sizes = [opt["min_region_sizes"][0]]
    dist_threshes = [opt["distance_threshes"][0]]
    dist_confs = [opt["distance_confs"][0]]

    # results: class x overlap
    ap = np.zeros((len(dist_threshes), len(CLASS_LABELS), len(overlaps)), float)
    metric_dict = {}
    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
        zip(min_region_sizes, dist_threshes, dist_confs)
    ):
        for oi, overlap_th in enumerate(overlaps):
            pred_visited = {}
            for m in matches:
                for p in matches[m]["pred"]:
                    for label_name in CLASS_LABELS:
                        for p in matches[m]["pred"][label_name]:
                            if "uuid" in p:
                                pred_visited[p["uuid"]] = False
            for li, label_name in enumerate(CLASS_LABELS):
                y_true = np.empty(0)
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False
                for m in matches:
                    pred_instances = matches[m]["pred"][label_name]
                    gt_instances = matches[m]["gt"][label_name]
                    # filter groups in ground truth
                    gt_instances = [
                        gt
                        for gt in gt_instances
                        if gt["vert_count"] >= min_region_size
                        and gt["med_dist"] <= distance_thresh
                        and gt["dist_conf"] >= distance_conf
                    ]
                    if gt_instances:
                        has_gt = True
                    if pred_instances:
                        has_pred = True

                    cur_true = np.ones(len(gt_instances))
                    cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                    cur_match = np.zeros(len(gt_instances), dtype=bool)
                    # collect matches
                    for gti, gt in enumerate(gt_instances):
                        found_match = False
                        # num_pred = len(gt["matched_pred"])
                        for pred in gt["matched_pred"]:
                            # get pred with pred_id
                            # pred = next(
                            #     (
                            #         pred
                            #         for pred in pred_instances
                            #         if pred["uuid"] == pred_uuid
                            #     ),
                            #     None,
                            # )
                            # greedy assignments
                            if pred_visited[pred["uuid"]]:
                                continue
                            overlap = float(pred["intersection"]) / (
                                gt["vert_count"]
                                + pred["vert_count"]
                                - pred["intersection"]
                            )
                            if overlap > overlap_th:
                                confidence = pred["confidence"]
                                # if already have a prediction for this gt,
                                # the prediction with the lower score is automatically a false positive
                                if cur_match[gti]:
                                    max_score = max(cur_score[gti], confidence)
                                    min_score = min(cur_score[gti], confidence)
                                    cur_score[gti] = max_score
                                    # append false positive
                                    cur_true = np.append(cur_true, 0)
                                    cur_score = np.append(cur_score, min_score)
                                    cur_match = np.append(cur_match, True)
                                # otherwise set score
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred["uuid"]] = True
                        if not found_match:
                            hard_false_negatives += 1
                    # remove non-matched ground truth instances
                    cur_true = cur_true[cur_match]
                    cur_score = cur_score[cur_match]

                    # collect non-matched predictions as false positive
                    for pred in pred_instances:
                        found_gt = False
                        for gt in pred["matched_gt"]:
                            # gt = next(
                            #     (gt for gt in gt_instances if gt["uuid"] == gt_uuid),
                            #     None,
                            # )
                            overlap = float(gt["intersection"]) / (
                                gt["vert_count"]
                                + pred["vert_count"]
                                - gt["intersection"]
                            )
                            if overlap > overlap_th:
                                found_gt = True
                                break
                        if not found_gt:
                            cur_true = np.append(cur_true, 0)
                            confidence = pred["confidence"]
                            cur_score = np.append(cur_score, confidence)

                    # append to overall results
                    y_true = np.append(y_true, cur_true)
                    y_score = np.append(y_score, cur_score)

                # compute average precision
                if has_gt and has_pred:
                    # compute precision recall curve first

                    # sorting and cumsum
                    score_arg_sort = np.argsort(y_score)
                    y_score_sorted = y_score[score_arg_sort]
                    y_true_sorted = y_true[score_arg_sort]
                    y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                    # unique thresholds
                    (thresholds, unique_indices) = np.unique(
                        y_score_sorted, return_index=True
                    )
                    num_prec_recall = len(unique_indices) + 1

                    # prepare precision recall
                    num_examples = len(y_score_sorted)
                    # https://github.com/ScanNet/ScanNet/pull/26
                    # all predictions are non-matched but also all of them are ignored and not counted as FP
                    # y_true_sorted_cumsum is empty
                    # num_true_examples = y_true_sorted_cumsum[-1]
                    num_true_examples = (
                        y_true_sorted_cumsum[-1] if len(y_true_sorted_cumsum) > 0 else 0
                    )
                    precision = np.zeros(num_prec_recall)
                    recall = np.zeros(num_prec_recall)

                    # deal with the first point
                    y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                    # deal with remaining
                    for idx_res, idx_scores in enumerate(unique_indices):
                        cumsum = y_true_sorted_cumsum[idx_scores - 1]
                        tp = num_true_examples - cumsum
                        fp = num_examples - idx_scores - tp
                        fn = cumsum + hard_false_negatives
                        p = float(tp) / (tp + fp)
                        r = float(tp) / (tp + fn)
                        precision[idx_res] = p
                        recall[idx_res] = r

                    # first point in curve is artificial
                    precision[-1] = 1.0
                    recall[-1] = 0.0

                    # compute average of precision-recall curve
                    recall_for_conv = np.copy(recall)
                    recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                    recall_for_conv = np.append(recall_for_conv, 0.0)

                    stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], "valid")
                    # integrate is now simply a dot product
                    ap_current = np.dot(precision, stepWidths)

                    # Save the precision and recall values for the current class, if overlap threshold is 0.25 or 0.5
                    if label_name not in metric_dict:
                        metric_dict[label_name] = {}
                    metric_dict[label_name][overlap_th] = {
                        "precision": precision,
                        "recall": recall,
                        "ap": ap_current,
                    }

                elif has_gt:
                    ap_current = 0.0
                else:
                    ap_current = float("nan")
                ap[di, li, oi] = ap_current
    return ap, metric_dict


def compute_ap_averages(aps, factor=100):  # factor is 1 or 100 (for percent)
    d_inf = 0
    o50 = np.where(np.isclose(opt["overlap_thresholds"], 0.5))
    o25 = np.where(np.isclose(opt["overlap_thresholds"], 0.25))
    oAllBut25 = np.where(np.logical_not(np.isclose(opt["overlap_thresholds"], 0.25)))
    avg_dict = {}
    avg_dict["all_ap"] = np.nanmean(aps[d_inf, :, oAllBut25]) * factor
    avg_dict["all_ap_50%"] = np.nanmean(aps[d_inf, :, o50]) * factor
    avg_dict["all_ap_25%"] = np.nanmean(aps[d_inf, :, o25]) * factor
    return avg_dict

def compute_query_inverse_rank(matches, iou_threshold):
    """
    Evaluate the rank metric using predictions.

    Args:
        matches (dict): Dictionary with structure:
            {scene_id: {"gt": {"object": [gt_instances]},
                        "pred": {"object": [pred_instances]}}}
        iou_threshold (float): IoU threshold for considering a prediction as a valid match.

    Returns:
        avg_inverse_rank (float): The average inverse rank.
        scene_query_ranks (dict): Dictionary with per-scene, per-query adjusted ranks.
            Format: {scene_id: {query_id: [list_of_adjusted_ranks]}}
    """
    scene_query_ranks = {}

    # Process each scene individually.
    for scene_id, match in matches.items():
        pred_instances = match["pred"]["object"]
        gt_instances = match["gt"]["object"]

        # Create a dictionary to track which predictions have already been used.
        pred_visited = {pred["uuid"]: False for pred in pred_instances}

        # For each ground-truth instance, find the best matching prediction (if any)
        query_ranks = {}  # Stores, per query, a list of raw ranks (one per GT instance)
        for gt in gt_instances:
            qid = gt["query_id"]
            query_ranks.setdefault(qid, [])

            best_rank = None
            best_iou = 0.0
            best_pred_uuid = None

            # Iterate over all predictions that were matched to this GT instance.
            for pred in gt.get("matched_pred", []):
                uuid = pred["uuid"]

                # Skip predictions already used for another GT.
                if pred_visited.get(uuid, False):
                    continue

                # Compute the IoU between the GT and prediction.
                intersection = pred.get("intersection", 0)
                union = gt["vert_count"] + pred["vert_count"] - intersection
                iou = intersection / union if union > 0 else 0.0

                # If the IoU is above the threshold and is the best seen so far, select this prediction.
                if iou > iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_rank = pred["rank"]
                    best_pred_uuid = uuid

            # Record the rank if a valid prediction was found; otherwise, record -1.
            if best_rank is None:
                query_ranks[qid].append(-1)
            else:
                query_ranks[qid].append(best_rank)
                pred_visited[best_pred_uuid] = True  # Mark the prediction as used.

        # Adjust the rank list for each query:
        # For each query, sort the valid ranks (ignoring -1), then subtract the positional index
        # from each valid rank. (Leave any -1 entries unchanged.)
        adjusted_query_ranks = {}
        for qid, ranks in query_ranks.items():
            valid_ranks = sorted([r for r in ranks if r != -1])
            invalid_ranks = [r for r in ranks if r == -1]
            adjusted_valid_ranks = []
            for i, r in enumerate(valid_ranks):
                adjusted_valid_ranks.append(r - i)  # subtract the positional index

            # Combine and sort: the -1 values remain unchanged.
            adjusted_ranks = sorted(
                adjusted_valid_ranks + invalid_ranks, key=lambda x: (x == -1, x)
            )
            adjusted_query_ranks[qid] = adjusted_ranks

        # Save the per-query adjusted ranks for the current scene.
        scene_query_ranks[scene_id] = adjusted_query_ranks

    # Concatenate the adjusted ranks from all queries and scenes.
    final_rank_list = []
    for scene_data in scene_query_ranks.values():
        for rank_list in scene_data.values():
            final_rank_list.extend(rank_list)
    # Sort the final list (ensuring that -1 values remain unchanged).
    final_rank_list = sorted(final_rank_list, key=lambda x: (x == -1, x))

    # Compute the average inverse rank:
    # For valid ranks (r > 0), use 1/r; for -1 or r <= 0, use 0.
    inverse_ranks = [0.0 if r == -1 or r <= 0 else 1.0 / r for r in final_rank_list]
    avg_inverse_rank = sum(inverse_ranks) / len(inverse_ranks) if inverse_ranks else 0.0

    return avg_inverse_rank, scene_query_ranks