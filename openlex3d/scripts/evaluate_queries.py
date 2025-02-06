import numpy as np
from uuid import uuid4
from copy import deepcopy
import hydra
from omegaconf import DictConfig
import logging
import pickle
from pathlib import Path

from openlex3d import get_path
from openlex3d.core.io import load_raw_predictions, load_query_json
from openlex3d.core.average_precision import evaluate_matches, compute_averages
from openlex3d.datasets.scannetpp import load_dataset_gt_files
from openlex3d.core.transfer_masks import (
    get_pred_mask_indices_gt_aligned_global,
    get_pred_mask_indices_gt_aligned_per_mask,
)
from openlex3d.core.cosine_similarity import compute_normalized_cosine_similarities

logger = logging.getLogger(__name__)


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


# ----------------------------
# Get All GT Mask Indices
# ----------------------------
def get_all_gt_mask_indices(seg_indices, seg_anno):
    """
    Reads segmentation annotations and returns a dictionary mapping each object id to the
    corresponding indices (positions) in the mesh (seg_indices) where the object is present.
    """
    gt_dict = {}
    obj_ids, _, segments = seg_anno
    if obj_ids is None:
        raise ValueError("Segmentation annotations could not be loaded.")
    for i, obj_req in enumerate(obj_ids):
        seg_inds_for_obj = segments[i]
        gt_mask = np.isin(seg_indices, seg_inds_for_obj)
        mask_indices = np.nonzero(gt_mask)[0]
        gt_dict[int(obj_req)] = mask_indices
    return gt_dict


# ----------------------------
# Updated: Create GT Instances (using the precomputed dictionary)
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
# Updated: Create Predicted Instances (using aligned indices)
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
    )

    pred_instances = []
    n_instances = pred_features.shape[0]
    pred_id_counter = 0
    for i in range(n_instances):
        for j, qid in enumerate(query_ids):
            conf = norm_sim[i, j]
            if conf > threshold:
                mask_indices = aligned_pred_mask_indices.get(i, None)
                if mask_indices is None or len(mask_indices) < opt["min_region_sizes"]:
                    continue
                pred_inst = {
                    "uuid": str(uuid4()),
                    "pred_id": i,
                    "label_id": 1,
                    "vert_count": len(mask_indices),
                    "confidence": conf,
                    "query_id": qid,
                    "mask_indices": mask_indices,  # already aligned indices on GT mesh
                    "matched_gt": [],
                }
                pred_instances.append(pred_inst)
                pred_id_counter += 1
    return pred_instances


# ----------------------------
# Matching function: assign predictions to GT only if query_id matches
# ----------------------------
def assign_instances_for_scene(scene_id, gt_instances, pred_instances):
    """
    For each predicted instance and each GT instance (of the same query and label),
    compute the intersection (using the stored mask indices) and store the matching information.
    Returns a dictionary in the “matches” format.
    """
    for gt in gt_instances:
        gt["matched_pred"] = []
    for pred in pred_instances:
        pred["matched_gt"] = []
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
    matches = {
        scene_id: {"gt": {"object": gt_instances}, "pred": {"object": pred_instances}}
    }
    return matches


def print_matched_pred_ids_for_query(matches, query_id):
    for m in matches:
        gt_instances = matches[m]["gt"]["object"]
        for gt_instance in gt_instances:
            if gt_instance["query_id"] == query_id:
                print("GT instance:", gt_instance["instance_id"])
                for pred in gt_instance["matched_pred"]:
                    print("  Matched pred:", pred["pred_id"])


# ----------------------------
# Main evaluation pipeline (for one scene)
# ----------------------------
@hydra.main(
    version_base=None,
    config_path=f"{get_path()}/config",
    config_name="eval_query_config",
)
def main(cfg: DictConfig):
    print("Evaluating scene:", cfg.scene_id)

    # Load prediction files
    pred_pcd, pred_mask_indices, pred_features = load_raw_predictions(
        cfg.pred.path, cfg.scene_id
    )
    print("Loaded prediction files.")

    # Load ground truth: mesh and segindices
    mesh_vertices, seg_indices, seg_anno = load_dataset_gt_files(
        cfg.gt.base_path, cfg.scene_id
    )
    print("Loaded ground truth files.")

    # Load queries
    query_list = load_query_json(cfg.query.json_file)
    print("Loaded query file.")

    # Get all GT mask indices and create GT instances
    all_gt_mask_indices = get_all_gt_mask_indices(seg_indices, seg_anno)
    gt_instances = create_gt_instances(all_gt_mask_indices, query_list)
    print("Created GT instances.")

    # Align predicted masks to the GT mesh using the chosen mode:
    if cfg.masks.alignment_mode == "global":
        aligned_pred_mask_indices = get_pred_mask_indices_gt_aligned_global(
            pred_pcd, pred_mask_indices, mesh_vertices, cfg.masks.alignment_threshold
        )
    elif cfg.masks.alignment_mode == "per_mask":
        aligned_pred_mask_indices = get_pred_mask_indices_gt_aligned_per_mask(
            pred_pcd, pred_mask_indices, mesh_vertices, cfg.masks.alignment_threshold
        )
    else:
        raise ValueError("Invalid alignment_mode: choose 'global' or 'per_mask'")
    print("Aligned predicted masks to GT mesh.")

    # save aligned pred mask indices with pickle
    with open(
        str(
            Path(cfg.masks.output_path)
            / f"pred_masks_aligned_{cfg.scene_id}_{cfg.masks.alignment_mode}.pkl"
        ),
        "wb",
    ) as f:
        pickle.dump(aligned_pred_mask_indices, f)

    with open(
        str(Path(cfg.masks.output_path) / f"gt_masks_{cfg.scene_id}.pkl"), "wb"
    ) as f:
        pickle.dump(all_gt_mask_indices, f)

    # Create predicted instances using the aligned mask indices
    pred_instances = create_pred_instances(
        pred_features,
        aligned_pred_mask_indices,
        query_list,
        cfg.model,
        threshold=cfg.eval.threshold,
    )
    print("Created predicted instances.")

    # Build the matches dictionary for a single scene
    matches = assign_instances_for_scene(cfg.scene_id, gt_instances, pred_instances)
    print("Assigned instances.")

    # pdb.set_trace()

    print_matched_pred_ids_for_query(matches, "level0_weight")

    # Evaluate to compute AP
    ap_score = evaluate_matches(matches)
    avg_results = compute_averages(ap_score)
    print("Average Precision results:", avg_results)


if __name__ == "__main__":
    main()
