import open3d as o3d
import numpy as np
import pickle

# ------------------------------------------------------------------------------
# Adjust these paths before running
PCD_FILE_PATH = "/data/concept-graphs/scannetpp_openlex_v2/data/49a82360aa/scans/mesh_aligned_0.05.ply"
MASK_DICT_PATH = (
    "/home/kumaraditya/openlex3d/pred_masks_aligned_49a82360aa_per_mask.pkl"
)
# ------------------------------------------------------------------------------


def pc_estimate_normals(pcd, radius=0.1, max_nn=16):
    """
    Estimates the normals for a point cloud using a hybrid KDTree search.
    """
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    return pcd


class MaskViewer:
    def __init__(self, pcd_file_path, mask_dict_path):
        """
        Initializes the MaskViewer by loading the point cloud and the
        dictionary mapping obj_id -> array of point indices.
        """
        # 1. Load the point cloud
        self.pcd = o3d.io.read_point_cloud(pcd_file_path)
        self.pcd = pc_estimate_normals(self.pcd)
        if not self.pcd.has_points():
            raise ValueError(f"No points found in PCD at {pcd_file_path}")

        # 2. Load the mask dictionary (obj_id -> np.array of indices)
        with open(mask_dict_path, "rb") as f:
            self.mask_dict = pickle.load(f)
        if not isinstance(self.mask_dict, dict):
            raise ValueError("Loaded mask_dict is not a dictionary.")

        # 3. Store original colors
        #    If the PCD has no colors, we'll initialize all points as gray
        self.original_colors = np.asarray(self.pcd.colors).copy()
        if len(self.original_colors) == 0:
            point_count = len(self.pcd.points)
            gray = np.full((point_count, 3), 0.5)  # mid-gray
            self.pcd.colors = o3d.utility.Vector3dVector(gray)
            self.original_colors = np.asarray(self.pcd.colors).copy()

    def highlight_obj(self, obj_id):
        """
        Highlights the points belonging to the given obj_id in their
        original color, while making all other points gray.
        """
        # 1. Prepare an all-gray array
        point_count = len(self.pcd.points)
        gray = np.full((point_count, 3), 0.5)

        # 2. Get the indices for the chosen object
        if obj_id not in self.mask_dict:
            print(
                f"Object ID {obj_id} not found in dictionary. No highlighting applied."
            )
            self.pcd.colors = o3d.utility.Vector3dVector(gray)
            return

        indices_to_highlight = self.mask_dict[obj_id]
        print(indices_to_highlight)

        # 3. Restore red color for the highlighted object
        # gray[indices_to_highlight] = self.original_colors[indices_to_highlight]
        gray[indices_to_highlight] = np.array([1, 0, 0])

        # 4. Update the point cloud colors
        self.pcd.colors = o3d.utility.Vector3dVector(gray)

    def reset_colors(self):
        """
        Resets the point cloud to its original colors.
        """
        self.pcd.colors = o3d.utility.Vector3dVector(self.original_colors)

    # --- CALLBACKS ---

    def highlight_callback(self, vis):
        """
        A callback that prompts the user (via terminal) to input an object ID,
        then highlights that object's points.
        """
        obj_id_str = input("Enter object ID to highlight: ")
        try:
            obj_id = int(obj_id_str)
        except ValueError:
            print("Invalid integer input.")
            return False

        self.highlight_obj(obj_id)
        vis.update_geometry(self.pcd)
        return False  # Continue the visualizer loop

    def reset_callback(self, vis):
        """
        A callback that resets the PCD to the original colors.
        """
        self.reset_colors()
        vis.update_geometry(self.pcd)
        return False  # Continue the visualizer loop

    # --- MAIN LOOP ---

    def run(self):
        """
        Creates the Open3D visualization window, loads the point cloud,
        registers the key callbacks, and starts the interactive loop.
        """
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="Mask Viewer", width=1280, height=720)

        # Add geometry once
        vis.add_geometry(self.pcd)

        # Register a callback to prompt the user for an obj_id when 'H' is pressed
        vis.register_key_callback(ord("H"), self.highlight_callback)

        # Register a callback to reset the colors when 'R' is pressed
        vis.register_key_callback(ord("R"), self.reset_callback)

        print("Instructions:")
        print("  - Press 'H' to highlight an object.")
        print("  - Then enter the object ID in the terminal.")
        print("  - Press 'R' to reset all colors.")
        print("  - Press ESC or close window to exit.\n")

        vis.run()
        vis.destroy_window()


class MaskViewerObjects:
    def __init__(self, pcd_file_path, mask_dict_path):
        """
        Initializes the MaskViewer by loading the point cloud,
        loading the dictionary mapping obj_id -> array of point indices,
        and setting up normal estimation.
        """
        # 1. Load the full point cloud
        full_pcd = o3d.io.read_point_cloud(pcd_file_path)
        if not full_pcd.has_points():
            raise ValueError(f"No points found in PCD at {pcd_file_path}")

        # 2. Load the mask dictionary (obj_id -> np.array of indices)
        with open(mask_dict_path, "rb") as f:
            self.mask_dict = pickle.load(f)
        if not isinstance(self.mask_dict, dict):
            raise ValueError("Loaded mask_dict is not a dictionary.")

        # 3. Estimate normals on the full point cloud
        pc_estimate_normals(full_pcd)

        # Store this original geometry (all points)
        # We'll always select subsets from here
        self.full_pcd = full_pcd

        # 4. Prepare the 'display' point cloud (initially the same as full)
        self.display_pcd = o3d.geometry.PointCloud(self.full_pcd)

    def refresh_display_pcd(self, indices, vis):
        """
        Replace the currently displayed geometry with a subset of
        the full point cloud (as selected by 'indices').
        """
        # Remove the old display geometry from the viewer
        vis.remove_geometry(self.display_pcd, reset_bounding_box=False)

        # Create a new geometry for the subset
        # (only the points corresponding to 'indices')
        new_pcd = self.full_pcd.select_by_index(indices, invert=False)

        # Estimate normals again for the subset
        pc_estimate_normals(new_pcd)

        # Update the display_pcd reference
        self.display_pcd = new_pcd

        # Add new geometry back to the viewer
        vis.add_geometry(self.display_pcd)
        vis.update_renderer()

    # ----------- CALLBACKS ----------- #

    def highlight_callback(self, vis):
        """
        Callback that prompts the user for an obj_id, then displays
        only those points (all others become hidden).
        """
        obj_id_str = input("Enter object ID to highlight: ")
        try:
            obj_id = int(obj_id_str)
        except ValueError:
            print("Invalid integer input.")
            return False  # keep visualizer running

        if obj_id not in self.mask_dict:
            print(f"Object ID {obj_id} not found in dictionary. Nothing to show.")
            return False

        # Get the indices for the chosen object
        highlight_indices = self.mask_dict[obj_id]

        # Refresh the displayed geometry to only these indices
        self.refresh_display_pcd(highlight_indices, vis)
        return False

    def reset_callback(self, vis):
        """
        Callback that shows all points again (resets the display).
        """
        # Use all indices
        all_indices = np.arange(len(self.full_pcd.points))
        self.refresh_display_pcd(all_indices, vis)
        return False

    # ----------- MAIN LOOP ----------- #

    def run(self):
        """
        Creates the Open3D visualization window, registers key callbacks,
        and starts the interactive loop.
        """
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="Mask Viewer", width=1280, height=720)

        # Add the initial geometry (all points)
        vis.add_geometry(self.display_pcd)

        # Register callbacks:
        #   'H' -> highlight an object (only show its points)
        vis.register_key_callback(ord("H"), self.highlight_callback)
        #   'R' -> reset to show all points
        vis.register_key_callback(ord("R"), self.reset_callback)

        print("Instructions:")
        print(
            "  - Press 'H' to highlight an object (terminal will prompt for an object ID)."
        )
        print("  - Press 'R' to reset (show all points).")
        print("  - Press ESC or close the window to exit.\n")

        vis.run()
        vis.destroy_window()


if __name__ == "__main__":
    viewer = MaskViewerObjects(PCD_FILE_PATH, MASK_DICT_PATH)
    viewer.run()
