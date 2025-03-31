from pathlib import Path
import pickle
import open3d as o3d
import argparse
import open3d.visualization.gui as gui

from collections import defaultdict


def matches_to_per_query_mask_indices(matches, target_query):
    # Aggregate mask indices per query for both ground truth and predictions
    mask_indices = defaultdict(lambda: {"gt": set(), "pred": set(), "inter": set()})

    scene = list(matches.keys())[0]

    for gt in matches[scene]["gt"]["object"]:
        query = gt["query_id"].split("_")[1]
        if query != target_query:
            continue

        mask_indices[query]["gt"] = mask_indices[query]["gt"].union(gt["mask_indices"])

    for pred in matches[scene]["pred"]["object"]:
        query = pred["query_id"].split("_")[1]
        if query != target_query:
            continue
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


def get_unique_queries(matches):
    queries = set()
    scene = list(matches.keys())[0]

    for gt in matches[scene]["gt"]["object"]:
        query = gt["query_id"].split("_")[1]
        queries.add(query)

    return sorted(list(queries))


class QueryViewer:
    def __init__(self, viz_path: Path):
        self.viz_path = viz_path

        self.main_pcd_name = "pcd_main"
        self.main_pcd = o3d.io.read_point_cloud(str(viz_path / "point_cloud.pcd"))
        self.matches = pickle.load(open(viz_path / "matches.pkl", "rb"))
        self.unique_queries = get_unique_queries(self.matches)

        self.current_query = None
        self.query_pcd_rgb_name = "pcd_query_rgb"
        self.query_pcd_rgb = None
        self.query_pcd_pred_gt_name = "pcd_query_pred_gt"
        self.query_pcd_pred_gt = None

        self.print_info()

    def print_info(self):
        print(
            "Instructions:\n"
            "1. Use the List Queries button to see the queries available.\n"
            "2. Enter the query you want to visualize using the New Query button.\n"
            "3. Use the Main button to visualize the entire point cloud.\n"
            "4. Use the Query RGB and Query Pred GT buttons to toggle between RGB and semantic views for the point cloud for that query.\n"
            "5. Semantic color info: Green is for the intersection between the ground truth and the prediction. Blue is for the ground truth points with no corresponding prediction. Red is for the prediction points with no corresponding ground truth.\n"
        )

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

        self.remove_geometries(vis, [self.main_pcd_name], [self.main_pcd])
        self.remove_geometries(vis, [self.query_pcd_rgb_name], [self.query_pcd_rgb])
        self.remove_geometries(
            vis, [self.query_pcd_pred_gt_name], [self.query_pcd_pred_gt]
        )

        if main_pcd:
            self.add_geometries(vis, [self.main_pcd_name], [self.main_pcd])

        if query_pcd_rgb:
            self.add_geometries(vis, [self.query_pcd_rgb_name], [self.query_pcd_rgb])

        if query_pcd_pred_gt:
            self.add_geometries(
                vis, [self.query_pcd_pred_gt_name], [self.query_pcd_pred_gt]
            )

    def update_geometries(self, viewer, geometry_names, geometries):
        self.remove_geometries(viewer, geometry_names, geometries)
        self.add_geometries(viewer, geometry_names, geometries)

    def load_query(self, query):
        mask_indices = matches_to_per_query_mask_indices(self.matches, query)

        gt = mask_indices[query]["gt"]
        pred = mask_indices[query]["pred"]
        inter = mask_indices[query]["inter"]

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

        if query not in self.unique_queries:
            print(f"Query {query} not found in unique queries")
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
        for q in self.unique_queries:
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
