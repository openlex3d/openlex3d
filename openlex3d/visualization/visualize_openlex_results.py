import open3d as o3d
import numpy as np
import argparse

def load_point_cloud(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    return pcd

def visualize_point_cloud(pcd, labels):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    
    picked_indices = vis.get_picked_points()
    for i, idx in enumerate(picked_indices):
        print(f"\n\n\nClicked point {i}")
        for j, label in enumerate(labels[idx]):
            print(f"{j:<3}: {label}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a point cloud and click on points to get their index.")
    parser.add_argument("pcd_path", type=str, help="Path to the point cloud file.")
    parser.add_argument("labels_path", type=str, help="Path to the labels file.")
    args = parser.parse_args()
    
    pcd = load_point_cloud(args.pcd_path)
    labels = np.load(args.labels_path)
    import pdb; pdb.set_trace()
    visualize_point_cloud(pcd, labels)
