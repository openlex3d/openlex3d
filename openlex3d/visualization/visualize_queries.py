from pathlib import Path
import json
import open3d as o3d
import argparse
import numpy as np
import open3d.visualization.gui as gui


class QueryViewer:

    def __init__(self, viz_path: Path):
        self.viz_path = viz_path

        self.main_pcd_name = "pcd_main"
        self.main_pcd = o3d.io.read_point_cloud(viz_path / "point_cloud.pcd")
        self.mask_indices = json.load(open(viz_path / "query_mask_indices.json"))


        self.current_query = None
        self.query_pcd_rgb_name = "pcd_query_rgb"
        self.query_pcd_rgb = None
        self.query_pcd_pred_gt_name = "pcd_query_pred_gt"
        self.query_pcd_pred_gt = None


    def add_geometries(self, viewer, geometry_names, geometries):
        for name, geometry in zip(geometry_names, geometries):
                viewer.add_geometry(name, geometry)

    def remove_geometries(self, viewer, geometry_names, geometries):
        for name in geometry_names:
            viewer.remove_geometry(name)

    def update(self, vis, main_pcd: bool, query_pcd_rgb: bool, query_pcd_pred_gt: bool):
        if self.current_query is None:
            print("No query loaded. Showing full pcd")
            main_pcd, query_pcd_rgb, query_pcd_pred_gt = True, False, False

        if main_pcd:
            self.add_geometries(vis, [self.main_pcd_name], [self.main_pcd])
        else:
            self.remove_geometries(vis, [self.main_pcd_name], [self.main_pcd])

        if query_pcd_rgb:
            self.add_geometries(vis, [self.query_pcd_rgb_name], [self.query_pcd_rgb])
        else:
            self.remove_geometries(vis, [self.query_pcd_rgb_name], [self.query_pcd_rgb])

        if query_pcd_pred_gt:
            self.add_geometries(vis, [self.query_pcd_pred_gt_name], [self.query_pcd_pred_gt])
        else:
            self.remove_geometries(vis, [self.query_pcd_pred_gt_name], [self.query_pcd_pred_gt])

    def update_geometries(self, viewer, geometry_names, geometries):
        self.remove_geometries(viewer, geometry_names, geometries)
        self.add_geometries(viewer, geometry_names, geometries)


    def load_query(self, query):
        gt = self.mask_indices[query]['gt']
        pred = self.mask_indices[query]['pred']
        inter = self.mask_indices[query]['inter']

        gt_pcd = self.main_pcd.select_by_index(gt)
        pred_pcd = self.main_pcd.select_by_index(pred)
        inter_pcd = self.main_pcd.select_by_index(inter)

        self.query_pcd_rgb = gt_pcd + pred_pcd + inter_pcd

        gt_pcd.paint_uniform_color([0, 0, 1])
        pred_pcd.paint_uniform_color([1, 0, 0])
        inter_pcd.paint_uniform_color([0, 1, 0])

        self.query_pcd_pred_gt = gt_pcd + pred_pcd + inter_pcd
        self.current_query = query


    def query(self, vis):
        query = input("Enter query: ")

        if query not in self.mask_indices:
            print(f"Query {query} not found in mask indices")
            return

        self.load_query(query)

        self.toggle_query_pred_gt(vis)

    def toggle_main(self, vis):
        self.update(vis, True, False, False)

    def toggle_query_rgb(self, vis):
        self.update(vis, False, True, False)

    def toggle_query_pred_gt(self, vis):
        self.update(vis, False, False, True)

    def print_available_queries(self, vis):
        print("Available queries:")
        queries = list(self.mask_indices.keys())
        queries = sorted(queries)

        for q in queries:
            print(q)


    def register_callbacks(self, vis):
        vis.add_action("Main", self.toggle_main)
        vis.add_action("Query RGB", self.toggle_query_rgb)
        vis.add_action("Query Pred GT", self.toggle_query_pred_gt)
        vis.add_action("New Query", self.query)
        vis.add_action("List Queries", self.print_available_queries)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize a query, the predictions from the method and the ground truth objects matched with the query."
    )
    parser.add_argument(
        "viz_path",
        type=str,
        help="Path to viz directory as saved by evaluate_queries.py",
    )
    args = parser.parse_args()

    viz_path = Path(args.viz_path) 


    # App
    viewer = QueryViewer(viz_path)
    app = gui.Application.instance
    app.initialize()

    vis = o3d.visualization.O3DVisualizer("Open3D", 1024, 768)
    vis.set_background([1.0, 1.0, 1.0, 1.0], bg_image=None)
    vis.show_settings = True
    vis.show_skybox(False)
    vis.enable_raw_mode(True)
    viewer.add_geometries(vis, [viewer.main_pcd_name], [viewer.main_pcd])
    viewer.register_callbacks(vis)
    vis.reset_camera_to_default()

    app.add_window(vis)
    app.run()




        



