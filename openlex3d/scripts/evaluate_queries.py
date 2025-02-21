from pathlib import Path
import numpy as np
from uuid import uuid4
from copy import deepcopy
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
import logging
import pickle
import json
import open3d as o3d

from openlex3d import get_path
from openlex3d.core.io import load_all_predictions, load_query_json
from openlex3d.datasets import load_dataset_with_obj_ids
from openlex3d.core.average_precision import evaluate_matches, compute_averages
from openlex3d.core.rank_metric import evaluate_rank
from openlex3d.core.align_masks import get_pred_mask_indices_gt_aligned
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
# Save Query Results in JSON
# ----------------------------
def save_query_results_json(cfg, per_scene_results, overall_results, json_output_dir):
    # Convert the entire cfg to a Python dictionary
    cfg_as_dict = OmegaConf.to_container(cfg, resolve=True)

    results_dict = {
        "cfg": cfg_as_dict,  # store everything if you want
        "results": {"per_scene": per_scene_results, "overall": overall_results},
    }

    json_output_path = Path(json_output_dir) / "results.json"
    with open(json_output_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    logger.info(f"Saved JSON results to: {json_output_path}")


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
    cfg, pred_features, aligned_pred_mask_indices, query_list, model_cfg
):
    # # NOTE: Can be changed later
    # if cfg.eval.metric == "rank":
    #     cfg.eval.top_k = pred_features.shape[0]

    if cfg.criteria == "clip_threshold":
        pred_instances = create_pred_instances_clip_thresh(
            pred_features,
            aligned_pred_mask_indices,
            query_list,
            model_cfg,
            threshold=cfg.clip_threshold,
        )
    elif cfg.criteria == "top_k":
        pred_instances = create_pred_instances_top_k(
            pred_features, aligned_pred_mask_indices, query_list, model_cfg, k=cfg.top_k
        )
    else:
        raise ValueError(
            "Invalid evaluation criteria: choose 'clip_threshold' or 'top_k'"
        )

    return pred_instances


def create_pred_instances_clip_thresh(
    pred_features, aligned_pred_mask_indices, query_list, model_cfg, threshold=0.6
):
    """
    For each predicted instance and for each query, compute the normalized cosine similarity
    between the instance feature and the query text feature.
    If above the CLIP sim threshold, create a predicted instance using the aligned mesh indices.
    """
    query_texts = [q["query_text"] for q in query_list]
    query_ids = [q["query_id"] for q in query_list]

    logger.info(f"Number of queries: {len(query_ids)}")
    logger.info(f"Number of pred instances: {pred_features.shape[0]}")

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

    logger.info(f"Skipped {len(indices_skipped)} instances.")
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

    logger.info(f"Number of queries: {len(query_ids)}")
    logger.info(f"Number of pred instances: {pred_features.shape[0]}")

    norm_sim = compute_normalized_cosine_similarities(
        model_cfg, pred_features, query_texts
    )  # (n_instances, n_queries)

    pred_instances = []
    # n_instances = pred_features.shape[0]

    # Get the top-k predictions per query
    top_k_indices = np.argsort(-norm_sim, axis=0)[:k, :]

    indices_skipped = set()

    for j, qid in enumerate(query_ids):
        for i in top_k_indices[:, j]:  # Get top-k instances for query j
            conf = norm_sim[i, j]
            rank = np.argsort(-norm_sim[:, j]).tolist().index(i) + 1

            mask_indices = aligned_pred_mask_indices.get(i, None)
            if mask_indices is None or len(mask_indices) < opt["min_region_sizes"]:
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

    logger.info(f"Skipped {len(indices_skipped)} instances.")
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
# Main evaluation function for a single scene
# ----------------------------
def get_matches_for_scene(cfg, scene_id):
    # Load prediction files
    logger.info(
        f"Loading predictions for method {cfg.pred.method} on dataset {cfg.dataset.name}, scene {scene_id}"
    )
    pred_pcd, pred_mask_indices, pred_features = load_all_predictions(
        cfg.pred.path, scene_id
    )

    # Load ground truth: mesh and segindices
    gt_pcd, obj_ids = load_dataset_with_obj_ids(cfg.dataset, scene_id)
    gt_pcd_points = gt_pcd.point.positions.numpy()

    # Load queries
    logger.info("Loading queries")
    query_list = load_query_json(cfg.query.path, cfg.dataset.name, scene_id)

    # Get all GT mask indices and create GT instances
    logger.info("Creating GT instances")
    all_gt_mask_indices = get_all_gt_mask_indices(obj_ids)
    gt_instances = create_gt_instances(all_gt_mask_indices, query_list)

    # Align predicted masks to the GT mesh using the chosen mode:
    logger.info(
        f"Aligning predicted masks to GT mesh with method {cfg.masks.alignment_mode} and dist threshold {cfg.masks.alignment_threshold}"
    )
    aligned_pred_mask_indices = get_pred_mask_indices_gt_aligned(
        cfg.masks, pred_pcd, pred_mask_indices, gt_pcd_points
    )

    logger.info(
        f"Creating predicted instances with criteria {cfg.eval.criteria} ({cfg.eval.clip_threshold} or {cfg.eval.top_k})"
    )
    pred_instances = create_pred_instances(
        cfg.eval, pred_features, aligned_pred_mask_indices, query_list, cfg.model
    )

    # Build the matches dictionary for a single scene
    logger.info("Assigning instances")
    matches, match_stats = assign_instances_for_scene(
        scene_id, gt_instances, pred_instances
    )

    logger.info("Saving matches")
    # For visualization
    viz_path = (
        Path(cfg.output_path)
        / "viz"
        / cfg.dataset.name
        / scene_id
        / f"{cfg.pred.method}_{cfg.masks.alignment_mode}_{cfg.masks.alignment_threshold}_{cfg.eval.criteria}_{cfg.eval.clip_threshold}_{cfg.eval.top_k}"
    )

    viz_path.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(viz_path / "point_cloud.pcd"), gt_pcd.to_legacy())

    pickle.dump(matches, open(viz_path / "matches.pkl", "wb"))

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

    logger.info(
        f"Starting evaluation with {cfg.eval.metric} metric "
        f"for method {cfg.pred.method} on dataset {cfg.dataset.name}"
    )

    # Prepare dictionaries to hold per‐scene metrics you care about
    per_scene_results = {}

    for scene_id in scenes:
        matches, _ = get_matches_for_scene(cfg, scene_id)
        all_matches.update(matches)

        if cfg.eval.metric == "rank":
            avg_inverse_rank, scene_query_ranks = evaluate_rank(
                matches, cfg.eval.iou_threshold
            )
            logger.info(f"Scene {scene_id} - Average inverse rank: {avg_inverse_rank}")
            per_scene_results[scene_id] = {"avg_inverse_rank": avg_inverse_rank}

        elif cfg.eval.metric == "ap":
            ap_score, metric_dict = evaluate_matches(matches)
            avg_results = compute_averages(ap_score)
            logger.info(f"Scene {scene_id} - Average Precision: {avg_results}")
            per_scene_results[scene_id] = avg_results

    # Compute overall (all scenes)
    overall_results = {}

    if cfg.eval.metric == "rank":
        avg_inverse_rank, scene_query_ranks = evaluate_rank(
            all_matches, cfg.eval.iou_threshold
        )
        logger.info(f"IoU used: {cfg.eval.iou_threshold}")
        logger.info(f"Overall average inverse rank: {avg_inverse_rank}")

        # Save the pickled ranks (if needed)
        ranks_output_path = (
            Path(cfg.output_path)
            / "rank_metric"
            / cfg.dataset.name
            / f"{cfg.pred.method}_{cfg.masks.alignment_mode}_"
            f"{cfg.masks.alignment_threshold}_{cfg.eval.top_k}_{cfg.eval.iou_threshold}"
        )
        ranks_output_path.mkdir(parents=True, exist_ok=True)
        pickle.dump(
            scene_query_ranks, open(ranks_output_path / "query_ranks.pkl", "wb")
        )

        overall_results["avg_inverse_rank"] = avg_inverse_rank

    elif cfg.eval.metric == "ap":
        ap_score, metric_dict = evaluate_matches(all_matches)
        avg_results = compute_averages(ap_score)
        logger.info(f"Overall Average Precision: {avg_results}")

        # Save the pickled metrics (if needed)
        ap_output_path = (
            Path(cfg.output_path)
            / "ap_metric"
            / cfg.dataset.name
            / f"{cfg.pred.method}_{cfg.masks.alignment_mode}_{cfg.masks.alignment_threshold}_{cfg.eval.criteria}_{cfg.eval.clip_threshold}_{cfg.eval.top_k}"
        )
        ap_output_path.mkdir(parents=True, exist_ok=True)
        pickle.dump(metric_dict, open(ap_output_path / "ap_metrics.pkl", "wb"))

        overall_results.update(avg_results)

    # Finally, save JSON containing all metadata + results
    json_output_dir = hydra.core.hydra_config.HydraConfig().get()["runtime"][
        "output_dir"
    ]
    save_query_results_json(cfg, per_scene_results, overall_results, json_output_dir)


if __name__ == "__main__":
    main()
