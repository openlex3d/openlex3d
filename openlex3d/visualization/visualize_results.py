from pathlib import Path
import open3d as o3d
import numpy as np
import argparse
import colorama
from colorama import init, Style
from collections import defaultdict

from openlex3d.core.categories import (
    SYNONYMS,
    DEPICTIONS,
    VISUALLY_SIMILAR,
    INCORRECT,
    CLUTTER,
)

init()

color_map = defaultdict(lambda: colorama.Fore.WHITE)
color_map[SYNONYMS] = colorama.Fore.GREEN
color_map[DEPICTIONS] = colorama.Fore.CYAN
color_map[VISUALLY_SIMILAR] = colorama.Fore.YELLOW
color_map[CLUTTER] = colorama.Fore.BLUE
color_map[INCORRECT] = colorama.Fore.RED


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
            color = color_map[cat]
            print(color + f"{j:<3} |{cat:<10} |{label}" + Style.RESET_ALL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize a point cloud and click on points to get their index."
    )
    parser.add_argument(
        "results_path",
        type=str,
        help="Path to the results from the evaluate script. The path should include the desired algorithm, dataset and scene.",
    )
    args = parser.parse_args()

    results_path = Path(args.results_path)

    pcd = load_point_cloud(results_path / "point_cloud.pcd")
    labels = np.load(results_path / "point_labels.npy")
    categories = np.load(results_path / "point_categories.npy")

    while True:
        print(
            "\n\nShift + Click on the points you want to inspect then close the window!"
        )
        visualize_point_cloud(pcd, labels, categories)
        print("\n\nPress 'q' to quit, or any other key to continue.")
        if input() == "q":
            break
