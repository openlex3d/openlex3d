import numpy as np

# ----------------------------
# Global evaluation parameters
# ----------------------------
opt = {}
# We now use 'overlap_thresholds' in place of 'overlaps' in the evaluation code.
opt["overlap_thresholds"] = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
opt["min_region_sizes"] = 1  # adjust as needed
opt["distance_threshes"] = [float("inf")]
opt["distance_confs"] = [-float("inf")]

# For our case we only have one semantic class:
CLASS_LABELS = ["object"]
VALID_CLASS_IDS = np.array([1])
ID_TO_LABEL = {1: "object"}
LABEL_TO_ID = {"object": 1}


def evaluate_matches(matches):
    overlaps = opt["overlap_thresholds"]
    # For simplicity, use one minimum region size and one distance threshold/confidence:
    min_region_sizes = [opt["min_region_sizes"]]
    dist_threshes = [opt["distance_threshes"][0]]
    dist_confs = [opt["distance_confs"][0]]

    # Initialize AP array: (num_dist_thresh, num_classes, num_overlap_thresholds)
    ap = np.zeros((len(dist_threshes), len(CLASS_LABELS), len(overlaps)), float)
    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
        zip(min_region_sizes, dist_threshes, dist_confs)
    ):
        for oi, overlap_th in enumerate(overlaps):
            pred_visited = {}
            for m in matches:
                for p in matches[m]["pred"]["object"]:
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
                    # Filter ground truth instances
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
                    for gti, gt in enumerate(gt_instances):
                        found_match = False
                        for pred in gt["matched_pred"]:
                            if pred_visited[pred["uuid"]]:
                                continue
                            union = (
                                gt["vert_count"]
                                + pred["vert_count"]
                                - pred["intersection"]
                            )
                            overlap = (
                                float(pred["intersection"]) / union
                                if union > 0
                                else 0.0
                            )
                            if overlap > overlap_th:
                                confidence = pred["confidence"]
                                if cur_match[gti]:
                                    max_score = max(cur_score[gti], confidence)
                                    min_score = min(cur_score[gti], confidence)
                                    cur_score[gti] = max_score
                                    cur_true = np.append(cur_true, 0)
                                    cur_score = np.append(cur_score, min_score)
                                    cur_match = np.append(cur_match, True)
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred["uuid"]] = True
                        if not found_match:
                            hard_false_negatives += 1
                    cur_true = cur_true[cur_match]
                    cur_score = cur_score[cur_match]
                    for pred in pred_instances:
                        found_gt = False
                        for gt in pred["matched_gt"]:
                            union = (
                                gt["vert_count"]
                                + pred["vert_count"]
                                - gt.get("intersection", 0)
                            )
                            ovlp = (
                                float(gt.get("intersection", 0)) / union
                                if union > 0
                                else 0.0
                            )
                            if ovlp > overlap_th:
                                found_gt = True
                                break
                        if not found_gt:
                            cur_true = np.append(cur_true, 0)
                            cur_score = np.append(cur_score, pred["confidence"])
                    y_true = np.append(y_true, cur_true)
                    y_score = np.append(y_score, cur_score)
                if has_gt and has_pred and len(y_score) > 0:
                    score_arg_sort = np.argsort(y_score)
                    y_score_sorted = y_score[score_arg_sort]
                    y_true_sorted = y_true[score_arg_sort]
                    y_true_sorted_cumsum = np.cumsum(y_true_sorted)
                    thresholds, unique_indices = np.unique(
                        y_score_sorted, return_index=True
                    )
                    num_prec_recall = len(unique_indices) + 1
                    num_examples = len(y_score_sorted)
                    num_true_examples = (
                        y_true_sorted_cumsum[-1] if len(y_true_sorted_cumsum) > 0 else 0
                    )
                    precision = np.zeros(num_prec_recall)
                    recall = np.zeros(num_prec_recall)
                    y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                    for idx_res, idx_scores in enumerate(unique_indices):
                        cumsum = (
                            y_true_sorted_cumsum[idx_scores - 1]
                            if idx_scores > 0
                            else 0
                        )
                        tp = num_true_examples - cumsum
                        fp = num_examples - idx_scores - tp
                        fn = cumsum + hard_false_negatives
                        p = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
                        r = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
                        precision[idx_res] = p
                        recall[idx_res] = r
                    precision[-1] = 1.0
                    recall[-1] = 0.0
                    recall_for_conv = np.copy(recall)
                    recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                    recall_for_conv = np.append(recall_for_conv, 0.0)
                    stepWidths = np.convolve(
                        recall_for_conv, [-0.5, 0, 0.5], mode="valid"
                    )
                    ap_current = np.dot(precision, stepWidths)
                elif has_gt:
                    ap_current = 0.0
                else:
                    ap_current = float("nan")
                ap[di, li, oi] = ap_current
    return ap


def compute_averages(aps, factor=100):  # factor is 1 or 100 (for percent)
    d_inf = 0
    o50 = np.where(np.isclose(opt["overlap_thresholds"], 0.5))
    o25 = np.where(np.isclose(opt["overlap_thresholds"], 0.25))
    oAllBut25 = np.where(np.logical_not(np.isclose(opt["overlap_thresholds"], 0.25)))
    avg_dict = {}
    avg_dict["all_ap"] = np.nanmean(aps[d_inf, :, oAllBut25]) * factor
    avg_dict["all_ap_50%"] = np.nanmean(aps[d_inf, :, o50]) * factor
    avg_dict["all_ap_25%"] = np.nanmean(aps[d_inf, :, o25]) * factor
    return avg_dict
