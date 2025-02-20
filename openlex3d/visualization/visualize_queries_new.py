from pathlib import Path
import pickle
import open3d as o3d
import argparse
import open3d.visualization.gui as gui
from collections import defaultdict

# ------------------  ADDED IMPORTS FOR COLORMAP  ------------------ #
import matplotlib.cm as cm

# ------------------------------------------------------------------- #


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


# ------------------  HELPER FOR VIRIDIS COLOR BY RANK  ------------------ #
def get_viridis_color(rank, max_rank):
    """
    Returns an RGB color from [0,1], based on the ratio rank/max_rank,
    using the 'viridis' colormap.
    """
    # Safeguard: if max_rank == 0, just return a default color
    if max_rank <= 0:
        return [0.5, 0.5, 0.5]

    norm_val = float(rank) / float(max_rank)
    # matplotlib returns RGBA, we only need RGB
    c = cm.get_cmap("viridis")(norm_val)
    return [c[0], c[1], c[2]]


# ------------------------------------------------------------------------ #


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

        # ------------------  NEW GEOMETRY FOR PRED INSTANCES  ------------------ #
        self.query_pred_instances_name = "pcd_query_pred_instances"
        self.query_pred_instances = None
        # ----------------------------------------------------------------------- #

    def add_geometries(self, viewer, geometry_names, geometries):
        for name, geometry in zip(geometry_names, geometries):
            if geometry is not None:
                viewer.add_geometry(name, geometry)

    def remove_geometries(self, viewer, geometry_names, geometries):
        for name, geometry in zip(geometry_names, geometries):
            if geometry is not None:
                viewer.remove_geometry(name)

    # ------------------  EXTEND update() TO HANDLE NEW GEOMETRY  ------------------ #
    def update(
        self,
        vis,
        main_pcd: bool,
        query_pcd_rgb: bool,
        query_pcd_pred_gt: bool,
        query_pred_instances: bool = False,
    ):
        if self.current_query is None:
            print("No query loaded. Showing full pcd")
            main_pcd, query_pcd_rgb, query_pcd_pred_gt, query_pred_instances = (
                True,
                False,
                False,
                False,
            )

        # Remove old geometries first
        self.remove_geometries(
            vis,
            [
                self.main_pcd_name,
                self.query_pcd_rgb_name,
                self.query_pcd_pred_gt_name,
                self.query_pred_instances_name,
            ],
            [
                self.main_pcd,
                self.query_pcd_rgb,
                self.query_pcd_pred_gt,
                self.query_pred_instances,
            ],
        )

        # Add them back selectively
        if main_pcd:
            self.add_geometries(vis, [self.main_pcd_name], [self.main_pcd])

        if query_pcd_rgb:
            self.add_geometries(vis, [self.query_pcd_rgb_name], [self.query_pcd_rgb])

        if query_pcd_pred_gt:
            self.add_geometries(
                vis, [self.query_pcd_pred_gt_name], [self.query_pcd_pred_gt]
            )

        if query_pred_instances:
            self.add_geometries(
                vis, [self.query_pred_instances_name], [self.query_pred_instances]
            )

    # ------------------------------------------------------------------------------ #

    def update_geometries(self, viewer, geometry_names, geometries):
        self.remove_geometries(viewer, geometry_names, geometries)
        self.add_geometries(viewer, geometry_names, geometries)

    def load_query(self, query):
        """
        Load the query into:
          1) query_pcd_rgb
          2) query_pcd_pred_gt
          3) (NEW) query_pred_instances
        """
        mask_indices = matches_to_per_query_mask_indices(self.matches, query)

        gt = mask_indices[query]["gt"]
        pred = mask_indices[query]["pred"]
        inter = mask_indices[query]["inter"]

        gt_pcd = self.main_pcd.select_by_index(gt)
        pred_pcd = self.main_pcd.select_by_index(pred)
        inter_pcd = self.main_pcd.select_by_index(inter)

        # For "Query RGB" view:
        self.query_pcd_rgb = gt_pcd + pred_pcd + inter_pcd

        # Colors for "Query Pred GT" view
        gt_pcd.paint_uniform_color([0, 0, 1])  # Blue
        pred_pcd.paint_uniform_color([1, 0, 0])  # Red
        inter_pcd.paint_uniform_color([0, 1, 0])  # Green
        self.query_pcd_pred_gt = gt_pcd + pred_pcd + inter_pcd

        # ------------------  BUILD QUERY_PRED_INSTANCES COLORED BY RANK  ------------------ #
        scene = list(self.matches.keys())[0]
        pred_instances = o3d.geometry.PointCloud()

        # Find max rank for normalization if you want a consistent scale
        all_ranks = [
            p["rank"]
            for p in self.matches[scene]["pred"]["object"]
            if p["query_id"].split("_")[1] == query
        ]
        max_rank = max(all_ranks) if len(all_ranks) > 0 else 1

        for pred_obj in self.matches[scene]["pred"]["object"]:
            q = pred_obj["query_id"].split("_")[1]
            if q != query:
                continue

            rank = int(pred_obj["rank"])  # ensure integer
            indices = pred_obj["mask_indices"]
            sub_pcd = self.main_pcd.select_by_index(indices)

            # Get color from viridis
            color = get_viridis_color(rank, max_rank)
            sub_pcd.paint_uniform_color(color)

            # Accumulate
            pred_instances += sub_pcd

        self.query_pred_instances = pred_instances
        # -------------------------------------------------------------------- #

        self.current_query = query

    def query(self, vis):
        query = input("Enter query: ")

        if query not in self.unique_queries:
            print(f"Query {query} not found in unique queries")
            return

        self.load_query(query)

        self.toggle_query_pred_gt(vis)

    def toggle_main(self, vis):
        self.update(vis, True, False, False, False)

    def toggle_query_rgb(self, vis):
        self.update(vis, False, True, False, False)

    def toggle_query_pred_gt(self, vis):
        self.update(vis, False, False, True, False)

    # ------------------  NEW TOGGLE FOR QUERY PRED INSTANCES  ------------------ #
    def toggle_query_pred_instances(self, vis):
        self.update(vis, False, False, False, True)

    # --------------------------------------------------------------------------- #

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

        # ------------- ADD THE NEW BUTTON / ACTION FOR PRED INSTANCES  ------------- #
        vis.add_action("Query Pred Instances", self.toggle_query_pred_instances)
        # --------------------------------------------------------------------------- #


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
