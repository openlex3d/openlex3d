import argparse
from pathlib import Path

import open3d as o3d
import open3d.visualization.gui as gui
import numpy as np


class MaskViewer:
    def __init__(self, viz_path: Path):
        """
        Initialize the MaskViewer with the paths to:
          - point_cloud.pcd
          - index.npy
        We store the main point cloud, the mask index array, and any derived
        sub-point-clouds here.
        """

        # Load the main point cloud
        self.main_pcd_name = "pcd_main"
        self.main_pcd = o3d.io.read_point_cloud(str(viz_path / "point_cloud.pcd"))
        num_points = len(self.main_pcd.points)

        # Load the mask index array, shape should be (N,)
        self.index = np.load(str(viz_path / "index.npy"))
        assert len(self.index.shape) == 1, "index.npy must be 1D"
        assert self.index.shape[0] == num_points, (
            f"index.npy has {self.index.shape[0]} elements, "
            f"but point cloud has {num_points} points"
        )

        # Get all unique mask IDs
        self.unique_masks = np.unique(self.index)

        # We'll keep references for geometry that shows just a specific mask
        self.mask_pcd_name = "pcd_mask"
        self.mask_pcd = None

        # Keep track of which mask is being displayed
        self.current_mask_id = None

    def add_geometries(self, viewer, geometry_names, geometries):
        """Helper to add multiple geometries to the viewer."""
        for name, geometry in zip(geometry_names, geometries):
            viewer.add_geometry(name, geometry)

    def remove_geometries(self, viewer, geometry_names):
        """Helper to remove multiple geometries from the viewer."""
        for name in geometry_names:
            # Must handle if geometry does not exist in the scene
            viewer.remove_geometry(name)
            # try:
            #     viewer.remove_geometry(name)
            # except:
            #     print(f"Geometry {name} not found in the scene.")

    def update(self, vis, show_main=False, show_mask=False):
        """
        Decides which geometry (main_pcd, mask_pcd) should be
        currently displayed in the viewer.
        """
        # Remove any existing geometry from the viewer
        self.remove_geometries(vis, [self.main_pcd_name, self.mask_pcd_name])

        if show_main:
            self.add_geometries(vis, [self.main_pcd_name], [self.main_pcd])

        if show_mask and self.mask_pcd is not None:
            self.add_geometries(vis, [self.mask_pcd_name], [self.mask_pcd])

    def toggle_main(self, vis):
        """Action to display all points in the point cloud."""
        # We hide the mask display and show the main pcd
        self.update(vis, show_main=True, show_mask=False)

    def toggle_mask(self, vis):
        """
        Action to display only the selected mask's points.
        If no mask has been chosen yet, prompt the user.
        """
        if self.current_mask_id is None:
            self.select_mask(vis)  # Will call update inside if needed
        else:
            self.update(vis, show_main=False, show_mask=True)

    def select_mask(self, vis):
        """
        A callback that prompts the user for a mask ID,
        then creates a sub-point-cloud of only the points
        belonging to that mask.
        """
        user_input = input("Enter mask ID (integer): ")
        try:
            mask_id = int(user_input)
        except ValueError:
            print("Please enter a valid integer for the mask ID.")
            return

        if mask_id not in self.unique_masks:
            print(f"Mask ID {mask_id} not found. Available IDs: {self.unique_masks}")
            return

        # Construct the sub-point-cloud for this mask
        indices = np.where(self.index == mask_id)[0]
        self.mask_pcd = self.main_pcd.select_by_index(indices)

        # Optionally, give the mask points a uniform color
        # self.mask_pcd.paint_uniform_color([1, 0, 0])  # Red

        self.current_mask_id = mask_id

        # Update the scene to only show the mask
        self.update(vis, show_main=False, show_mask=True)

    def list_masks(self, vis):
        """A callback to list all available mask IDs."""
        print("Available mask IDs:")
        for m in self.unique_masks:
            print(m)

    def register_callbacks(self, vis):
        """
        Register actions (buttons in the O3DVisualizer UI).
        These can be seen in the menu of the visualization window.
        """
        vis.add_action("Show All Points", self.toggle_main)
        vis.add_action("Show Current Mask", self.toggle_mask)
        vis.add_action("Select New Mask", self.select_mask)
        vis.add_action("List Masks", self.list_masks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize points from a .pcd file by mask (as per index.npy)."
    )
    parser.add_argument(
        "viz_path",
        type=str,
        help="Path to directory with point_cloud.pcd and index.npy.",
    )
    args = parser.parse_args()

    viz_path = Path(args.viz_path)

    # Create our viewer
    viewer = MaskViewer(viz_path)

    # Set up the Open3D application and O3DVisualizer
    app = gui.Application.instance
    app.initialize()

    vis = o3d.visualization.O3DVisualizer("Mask Visualization", 1024, 768)
    vis.set_background([1.0, 1.0, 1.0, 1.0], bg_image=None)
    vis.show_settings = True
    vis.show_skybox(False)
    vis.enable_raw_mode(True)

    # Initially, add the full point cloud to the scene
    viewer.add_geometries(vis, [viewer.main_pcd_name], [viewer.main_pcd])

    # Register the menu callbacks
    viewer.register_callbacks(vis)

    # Reset the camera
    vis.reset_camera_to_default()

    # Launch the viewer
    app.add_window(vis)
    app.run()
