import open3d as o3d
import numpy as np
import argparse

def load_point_cloud(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    return pcd

def visualize_point_cloud(pcd, labels, categories):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    
    picked_indices = vis.get_picked_points()
    for i, idx in enumerate(picked_indices):
        print(f"\n\n\nClicked point {i}")
        for j, (label, cat) in enumerate(zip(labels[idx], categories[idx])):
            print(f"{j:<3} |{cat:<10} |{label}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a point cloud and click on points to get their index.")
    parser.add_argument("pcd_path", type=str, help="Path to the point cloud file.")
    parser.add_argument("labels_path", type=str, help="Path to the point labels file.")
    parser.add_argument("cat_path", type=str, help="Path to the point label categories file.")
    args = parser.parse_args()
    
    pcd = load_point_cloud(args.pcd_path)
    labels = np.load(args.labels_path)
    categories = np.load(args.cat_path)

    while True:
        print("\n\nShift + Click on the points you want to inspect then close the window!")
        visualize_point_cloud(pcd, labels, categories)
        print("\n\nPress 'q' to quit, or any other key to continue.")
        if input() == 'q':
            break
