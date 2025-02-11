import numpy as np
from uuid import uuid4
from copy import deepcopy
import hydra
from omegaconf import DictConfig
import logging
from collections import defaultdict

from openlex3d import get_path
from openlex3d.core.io import load_all_predictions, load_query_json
from openlex3d.datasets import load_dataset_with_obj_ids
from openlex3d.core.average_precision import evaluate_matches, compute_averages
from openlex3d.core.rank_metric import evaluate_rank
from openlex3d.core.align_masks import (
    get_pred_mask_indices_gt_aligned_global,
    get_pred_mask_indices_gt_aligned_per_mask,
)
from openlex3d.core.cosine_similarity import compute_normalized_cosine_similarities

logger = logging.getLogger(__name__)


# ----------------------------
# Global evaluation parameters
# ----------------------------
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


# ----------------------------
# Get All GT Mask Indices
# ----------------------------
def get_all_gt_mask_indices(obj_ids_pcd):
    gt_dict = {}
    obj_ids = np.unique(obj_ids_pcd)
    if obj_ids is None:
        raise ValueError("Segmentation annotations could not be loaded.")
    for obj_req in obj_ids:
        gt_mask = obj_ids_pcd == obj_req
        mask_indices = np.nonzero(gt_mask)[0]
        gt_dict[int(obj_req)] = mask_indices
    return gt_dict


# ----------------------------
# Create GT Instances
# ----------------------------
def create_gt_instances(all_gt_mask_indices, query_list):
    """
    For each query and for each object id specified in the query, lookup the precomputed
    gt_mask_indices from all_gt_mask_indices and create a GT instance.
    """
    gt_instances = []
    for query in query_list:
        qid = query["query_id"]
        for obj_req in query["object_ids"]:
            mask_indices = all_gt_mask_indices.get(int(obj_req), None)
            if mask_indices is None or len(mask_indices) < opt["min_region_sizes"]:
                continue
            gt_inst = {
                "instance_id": int(obj_req),
                "label_id": 1,
                "vert_count": len(mask_indices),
                "med_dist": -1,
                "dist_conf": 0.0,
                "query_id": qid,
                "mask_indices": mask_indices,  # storing indices only
                "matched_pred": [],
            }
            gt_instances.append(gt_inst)
    return gt_instances


# ----------------------------
# Create Predicted Instances (using aligned pointcloud indices)
# ----------------------------
def create_pred_instances(
    pred_features, aligned_pred_mask_indices, query_list, model_cfg, threshold=0.6
):
    """
    For each predicted instance and for each query, compute the normalized cosine similarity
    between the instance feature and the query text feature.
    If above the CLIP sim threshold, create a predicted instance using the aligned mesh indices.
    """
    query_texts = [q["query_text"] for q in query_list]
    query_ids = [q["query_id"] for q in query_list]

    norm_sim = compute_normalized_cosine_similarities(
        model_cfg, pred_features, query_texts
    )  # (n_instances, n_queries)

    pred_instances = []
    n_instances = pred_features.shape[0]
    indices_skipped = set()

    for i in range(n_instances):
        mask_indices = aligned_pred_mask_indices.get(i, None)
        if mask_indices is None or len(mask_indices) < opt["min_region_sizes"]:
            indices_skipped.add(i)
            continue

        for j, qid in enumerate(query_ids):
            conf = norm_sim[i, j]
            rank = np.argsort(-norm_sim[:, j]).tolist().index(i) + 1
            if conf > threshold:
                pred_inst = {
                    "uuid": str(uuid4()),
                    "pred_id": i,
                    "label_id": 1,
                    "vert_count": len(mask_indices),
                    "confidence": conf,
                    "rank": rank,
                    "query_id": qid,
                    "mask_indices": mask_indices,  # already aligned indices on GT mesh
                    "matched_gt": [],
                }
                pred_instances.append(pred_inst)

    print(f"Skipped {len(indices_skipped)} instances.")
    return pred_instances


def create_pred_instances_top_k(
    pred_features, aligned_pred_mask_indices, query_list, model_cfg, k=5
):
    """
    For each predicted instance and for each query, compute the normalized cosine similarity
    between the instance feature and the query text feature.
    Select the top-k predictions per query instead of using a fixed threshold.
    """
    query_texts = [q["query_text"] for q in query_list]
    query_ids = [q["query_id"] for q in query_list]

    print(f"Number of queries: {len(query_ids)}")
    print(f"Number of pred instances: {pred_features.shape[0]}")

    norm_sim = compute_normalized_cosine_similarities(
        model_cfg, pred_features, query_texts
    )  # (n_instances, n_queries)
    print(f"Computed normalized cosine similarities: {norm_sim.shape}")

    pred_instances = []
    # n_instances = pred_features.shape[0]

    # Get the top-k predictions per query
    top_k_indices = np.argsort(-norm_sim, axis=0)[:k, :]
    print(f"Top-k indices shape: {top_k_indices.shape}")

    indices_skipped = set()

    for j, qid in enumerate(query_ids):
        for i in top_k_indices[:, j]:  # Get top-k instances for query j
            conf = norm_sim[i, j]
            rank = np.argsort(-norm_sim[:, j]).tolist().index(i) + 1

            mask_indices = aligned_pred_mask_indices.get(i, None)
            if mask_indices is None or len(mask_indices) < opt["min_region_sizes"]:
                # print(f"Skipping instance {i}.")
                indices_skipped.add(i)
                continue

            pred_inst = {
                "uuid": str(uuid4()),
                "pred_id": i,
                "label_id": 1,
                "vert_count": len(mask_indices),
                "confidence": conf,
                "rank": rank,
                "query_id": qid,
                "mask_indices": mask_indices,  # already aligned indices on GT mesh
                "matched_gt": [],
            }
            pred_instances.append(pred_inst)

    print(f"Skipped {len(indices_skipped)} instances.")
    return pred_instances


# ----------------------------
# Matching function: assign predictions to GT only if query_id matches
# ----------------------------
def assign_instances_for_scene(scene_id, gt_instances, pred_instances):
    """
    For each predicted instance and each GT instance (of the same query),
    compute the intersection (using the stored mask indices) and store the matching information.
    Additionally, save statistics about the number of pred_instances matched to gt_instances
    before and after checking for intersection, as well as the total number of GT instances per query_id.
    """
    match_stats = {
        "before_intersection": {},  # Stores number of pred matches per gt before intersection check
        "after_intersection": {},  # Stores number of pred matches per gt after intersection check
        "total_gt_instances": {},  # Stores total number of GT instances per query_id
    }

    # Count total GT instances per query_id
    for gt in gt_instances:
        gt["matched_pred"] = []

        query_id = gt["query_id"]
        match_stats["total_gt_instances"].setdefault(query_id, 0)
        match_stats["total_gt_instances"][query_id] += 1

    for pred in pred_instances:
        pred["matched_gt"] = []

        query_id = pred["query_id"]
        match_stats["before_intersection"].setdefault(query_id, 0)
        match_stats["before_intersection"][query_id] += 1

    for pred in pred_instances:
        for gt in gt_instances:
            if pred["query_id"] != gt["query_id"]:
                continue
            if pred["label_id"] != gt["label_id"]:
                continue
            # Compute intersection as the number of common indices.
            intersection = len(np.intersect1d(gt["mask_indices"], pred["mask_indices"]))
            if intersection > 0:
                gt_copy = deepcopy(gt)
                gt_copy["intersection"] = intersection
                pred_copy = deepcopy(pred)
                pred_copy["intersection"] = intersection
                gt["matched_pred"].append(pred_copy)
                pred["matched_gt"].append(gt_copy)

    # Track matches after intersection check
    for gt in gt_instances:
        query_id = gt["query_id"]
        match_stats["after_intersection"].setdefault(query_id, len(gt["matched_pred"]))

    matches = {
        scene_id: {"gt": {"object": gt_instances}, "pred": {"object": pred_instances}}
    }
    return matches, match_stats


# ----------------------------
# Helper function: Print matched pred_ids for a specific query_id
# ----------------------------
def print_matched_pred_ids_for_query(matches, query_id):
    for m in matches:
        gt_instances = matches[m]["gt"]["object"]
        for gt_instance in gt_instances:
            if gt_instance["query_id"] == query_id:
                print("GT instance:", gt_instance["instance_id"])
                for pred in gt_instance["matched_pred"]:
                    print("  Matched pred:", pred["pred_id"])


# ----------------------------
# Helper function for viz: Convert matches to per-query mask indices
# ----------------------------
def matches_to_per_query_mask_indices(matches):
    # Aggregate mask indices per query for both ground truth and predictions
    mask_indices = defaultdict(lambda: {"gt": set(), "pred": set(), "inter": set()})

    scene = list(matches.keys())[0]

    for gt in matches[scene]["gt"]["object"]:
        query = gt["query_id"].split("_")[1]
        mask_indices[query]["gt"] = mask_indices[query]["gt"].union(gt["mask_indices"])

    for pred in matches[scene]["pred"]["object"]:
        query = pred["query_id"].split("_")[1]
        mask_indices[query]["pred"] = mask_indices[query]["pred"].union(
            pred["mask_indices"]
        )

    # Compute intersections
    for query in mask_indices:
        mask_indices[query]["inter"] = mask_indices[query]["gt"].intersection(
            mask_indices[query]["pred"]
        )
        mask_indices[query]["gt"] = (
            mask_indices[query]["gt"] - mask_indices[query]["inter"]
        )
        mask_indices[query]["pred"] = (
            mask_indices[query]["pred"] - mask_indices[query]["inter"]
        )

        # Convert to list
        for key in ["gt", "pred", "inter"]:
            mask_indices[query][key] = [int(i) for i in mask_indices[query][key]]

    return mask_indices


# ----------------------------
# Main evaluation function for a single scene
# ----------------------------
def get_matches_for_scene(cfg, scene_id):
    print("Evaluating scene:", scene_id)

    # Load prediction files
    pred_pcd, pred_mask_indices, pred_features = load_all_predictions(
        cfg.pred.path, scene_id
    )
    print("Loaded prediction files.")

    # Load ground truth: mesh and segindices
    gt_pcd, obj_ids = load_dataset_with_obj_ids(cfg.dataset, scene_id)
    gt_pcd_points = gt_pcd.point.positions.numpy()
    print("Loaded ground truth files.")

    # Load queries
    query_list = load_query_json(cfg.query.path, cfg.dataset.name, scene_id)
    print("Loaded query file.")

    # Get all GT mask indices and create GT instances
    all_gt_mask_indices = get_all_gt_mask_indices(obj_ids)
    gt_instances = create_gt_instances(all_gt_mask_indices, query_list)
    print("Created GT instances.")

    # Align predicted masks to the GT mesh using the chosen mode:
    if cfg.masks.alignment_mode == "global":
        aligned_pred_mask_indices = get_pred_mask_indices_gt_aligned_global(
            pred_pcd, pred_mask_indices, gt_pcd_points, cfg.masks.alignment_threshold
        )
    elif cfg.masks.alignment_mode == "per_mask":
        aligned_pred_mask_indices = get_pred_mask_indices_gt_aligned_per_mask(
            pred_pcd, pred_mask_indices, gt_pcd_points, cfg.masks.alignment_threshold
        )
    else:
        raise ValueError("Invalid alignment_mode: choose 'global' or 'per_mask'")
    print("Aligned predicted masks to GT mesh.")

    if cfg.eval.metric == "rank":
        top_k = cfg.eval.top_k  # NOTE: Can be changed later
    elif cfg.eval.metric == "ap":
        top_k = cfg.eval.top_k
    else:
        raise ValueError("Invalid evaluation metric: choose 'rank' or 'ap'")

    # Create predicted instances using the aligned mask indices
    if cfg.eval.criteria == "clip_threshold":
        pred_instances = create_pred_instances(
            pred_features,
            aligned_pred_mask_indices,
            query_list,
            cfg.model,
            threshold=cfg.eval.clip_threshold,
        )
    elif cfg.eval.criteria == "top_k":
        pred_instances = create_pred_instances_top_k(
            pred_features,
            aligned_pred_mask_indices,
            query_list,
            cfg.model,
            k=top_k,
        )
    else:
        raise ValueError(
            "Invalid evaluation criteria: choose 'clip_threshold' or 'top_k'"
        )
    print("Created predicted instances.")

    # Build the matches dictionary for a single scene
    matches, match_stats = assign_instances_for_scene(
        scene_id, gt_instances, pred_instances
    )
    print("Assigned instances.")

    # # Get mask indices per query for both ground truth and predictions
    # query_match_indices = matches_to_per_query_mask_indices(matches)
    # print("Converted matches to per-query mask indices.")

    # # For visualization
    # viz_path = (
    #     Path(cfg.output_path)
    #     / "viz"
    #     / cfg.dataset.name
    #     / scene_id
    #     / f"{cfg.pred.method}_{cfg.masks.alignment_mode}_{cfg.masks.alignment_threshold}_{cfg.eval.criteria}_{cfg.eval.clip_threshold}_{cfg.eval.top_k}"
    # )
    # viz_path.mkdir(parents=True, exist_ok=True)
    # with open(
    #     str(viz_path / "query_mask_indices.json"),
    #     "w",
    # ) as f:
    #     json.dump(query_match_indices, f)
    # o3d.t.io.write_point_cloud(str(viz_path / "point_cloud.pcd"), gt_pcd)

    # # Print matched pred_ids for a specific query_id
    # print_matched_pred_ids_for_query(matches, "level0_weight")

    return matches, match_stats


# ----------------------------
# Main evaluation pipeline
# ----------------------------
@hydra.main(
    version_base=None,
    config_path=f"{get_path()}/config",
    config_name="eval_query_config",
)
def main(cfg: DictConfig):
    scenes = cfg.dataset.scenes
    all_matches = {}

    for scene_id in scenes:
        matches, _ = get_matches_for_scene(cfg, scene_id)
        all_matches.update(matches)

    # Eval metric
    if cfg.eval.metric == "rank":
        avg_inverse_rank, scene_query_ranks = evaluate_rank(
            all_matches, cfg.eval.iou_threshold
        )
        print("Average Inverse Rank:", avg_inverse_rank)
    elif cfg.eval.metric == "ap":
        ap_score, metric_dict = evaluate_matches(all_matches)
        avg_results = compute_averages(ap_score)
        print("Average Precision results:", avg_results)


if __name__ == "__main__":
    main()
