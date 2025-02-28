#!/usr/bin/env python3
import sys
import json
import csv
from pathlib import Path


def gather_json_data(root_dir):
    """
    Recursively find all `results.json` files under `root_dir`.
    Return two lists of rows:
      1. overall_rows   -> data about overall metrics, one row per JSON file
      2. per_scene_rows -> data about per‐scene metrics, possibly multiple rows per JSON file
    """
    overall_rows = []
    per_scene_rows = []

    for json_file in Path(root_dir).rglob("*.json"):
        # Only process files named "results.json" to skip other JSONs
        if (json_file.name != "results.json"):
            continue

        with open(json_file, "r") as f:
            data = json.load(f)

        # The Hydra config is now under data["cfg"]
        cfg_data = data.get("cfg", {})
        dataset_cfg = cfg_data.get("dataset", {})
        pred_cfg = cfg_data.get("pred", {})
        masks_cfg = cfg_data.get("masks", {})
        eval_cfg = cfg_data.get("eval", {})
        query_cfg = cfg_data.get("query", {})

        # Extract relevant info for CSV
        dataset_name = dataset_cfg.get("name")
        pred_method = pred_cfg.get("method")
        alignment_mode = masks_cfg.get("alignment_mode")
        alignment_threshold = masks_cfg.get("alignment_threshold")

        eval_metric = eval_cfg.get("metric")
        criteria = eval_cfg.get("criteria")
        clip_threshold = eval_cfg.get("clip_threshold")
        top_k = eval_cfg.get("top_k")
        iou_threshold = eval_cfg.get("iou_threshold")
        query_level = query_cfg.get("level")

        # The results for overall and per‐scene are under data["results"]
        results_data = data.get("results", {})
        overall = results_data.get("overall", {})  # dict like {"all_ap": 0.123, ...}
        per_scene_data = results_data.get(
            "per_scene", {}
        )  # e.g. {"room0": {...}, "room1": {...}}

        # Build a row for overall metrics
        overall_row = {
            "dataset_name": dataset_name,
            "pred_method": pred_method,
            "alignment_mode": alignment_mode,
            "alignment_threshold": alignment_threshold,
            "eval_metric": eval_metric,
            "criteria": criteria,
            "clip_threshold": clip_threshold,
            "top_k": top_k,
            "iou_threshold": iou_threshold,
            "query_level": query_level,
            # For AP metrics (may be absent if metric is rank):
            "all_ap": overall.get("all_ap"),
            "all_ap_50%": overall.get("all_ap_50%"),
            "all_ap_25%": overall.get("all_ap_25%"),
            # For rank metrics (may be absent if metric is AP):
            "avg_inverse_rank": overall.get("avg_inverse_rank"),
        }
        overall_rows.append(overall_row)

        # For each scene, we create a row in the "per scene" CSV
        for scene_id, scene_metrics in per_scene_data.items():
            row = {
                "dataset_name": dataset_name,
                "pred_method": pred_method,
                "alignment_mode": alignment_mode,
                "alignment_threshold": alignment_threshold,
                "eval_metric": eval_metric,
                "criteria": criteria,
                "clip_threshold": clip_threshold,
                "top_k": top_k,
                "iou_threshold": iou_threshold,
                "query_level": query_level,
                # Add the scene ID:
                "scene_id": scene_id,
                # For AP metrics:
                "all_ap": scene_metrics.get("all_ap"),
                "all_ap_50%": scene_metrics.get("all_ap_50%"),
                "all_ap_25%": scene_metrics.get("all_ap_25%"),
                # For rank metrics:
                "avg_inverse_rank": scene_metrics.get("avg_inverse_rank"),
            }
            per_scene_rows.append(row)

    return overall_rows, per_scene_rows


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_csv.py <root_directory> [<output_directory>]")
        sys.exit(1)

    root_dir = Path(sys.argv[1])
    if len(sys.argv) == 3:
        output_dir = Path(sys.argv[2])
    else:
        output_dir = root_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    overall_rows, per_scene_rows = gather_json_data(root_dir)

    # Define the columns for the "overall" CSV
    overall_fieldnames = [
        "dataset_name",
        "pred_method",
        "alignment_mode",
        "alignment_threshold",
        "eval_metric",
        "criteria",
        "clip_threshold",
        "top_k",
        "iou_threshold",
        "query_level",
        "all_ap",
        "all_ap_50%",
        "all_ap_25%",
        "avg_inverse_rank",
    ]

    # Define the columns for the "per_scene" CSV
    per_scene_fieldnames = [
        "dataset_name",
        "pred_method",
        "alignment_mode",
        "alignment_threshold",
        "eval_metric",
        "criteria",
        "clip_threshold",
        "top_k",
        "iou_threshold",
        "query_level",
        "scene_id",
        "all_ap",
        "all_ap_50%",
        "all_ap_25%",
        "avg_inverse_rank",
    ]

    # Write the overall CSV
    overall_csv_path = output_dir / "all_results_overall.csv"
    with open(overall_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=overall_fieldnames)
        writer.writeheader()
        for row in overall_rows:
            writer.writerow(row)
    print(f"Overall CSV saved to: {overall_csv_path}")

    # Write the per‐scene CSV
    per_scene_csv_path = output_dir / "all_results_per_scene.csv"
    with open(per_scene_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_scene_fieldnames)
        writer.writeheader()
        for row in per_scene_rows:
            writer.writerow(row)
    print(f"Per-scene CSV saved to: {per_scene_csv_path}")


if __name__ == "__main__":
    main()
