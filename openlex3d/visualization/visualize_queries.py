from pathlib import Path
import json
import open3d as o3d
import argparse




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize a query, the predictions from the method and the ground truth objects matched with the query."
    )
    parser.add_argument(
        "viz_path",
        type=str,
        help="Path to viz directory as saved by evaluate_queries.py",
    )
    parser.add_argument(
        "query",
        type=str,
        help="Query",
    )
    args = parser.parse_args()

    viz_path = Path(args.viz_path) 

    pcd = o3d.io.read_point_cloud(viz_path / "point_cloud.pcd")
    mask_indices = json.load(open(viz_path / "query_mask_indices.json"))

    if args.query not in mask_indices:
        print(f"Query {args.query} not found in mask indices")
        exit(1)

    gt = mask_indices[args.query]['gt']
    pred = mask_indices[args.query]['pred']
    inter = mask_indices[args.query]['inter']

    # Color the points
    gt_pcd = pcd.select_by_index(list(gt))
    gt_pcd.paint_uniform_color([0, 0, 1])
    pred_pcd = pcd.select_by_index(list(pred))
    pred_pcd.paint_uniform_color([1, 0, 0])
    inter_pcd = pcd.select_by_index(list(inter))
    inter_pcd.paint_uniform_color([0, 1, 0])

    # Draw
    o3d.visualization.draw_geometries([gt_pcd, pred_pcd, inter_pcd])




        



