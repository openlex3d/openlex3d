import open3d as o3d
import numpy as np
import argparse

def load_point_cloud(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    return pcd

def visualize_point_cloud(pcd, labels, gt_to_pred):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    
    picked_indices = vis.get_picked_points()
    for i, idx in enumerate(picked_indices):
        print(f"\n\n\nClicked point {i}")
        pred_idx = gt_to_pred[idx]
        if pred_idx == -1:
            print("No labels :(")
        else:
            for j, label in enumerate(labels[pred_idx]):
                print(f"{j:<3}: {label}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a point cloud and click on points to get their index.")
    parser.add_argument("pcd_path", type=str, help="Path to the point cloud file.")
    parser.add_argument("labels_path", type=str, help="Path to the labels file.")
    parser.add_argument("gt_to_pred_path", type=str, help="Array mapping ground truth/point cloud indices to predicted labels indices.")
    args = parser.parse_args()
    
    pcd = load_point_cloud(args.pcd_path)
    labels = np.load(args.labels_path)
    gt_to_pred = np.load(args.gt_to_pred_path)

    while True:
        print("\n\nShift + Click on the points you want to inspect then close the window!")
        visualize_point_cloud(pcd, labels, gt_to_pred)
        print("\n\nPress 'q' to quit, or any other key to continue.")
        if input() == 'q':
            break
