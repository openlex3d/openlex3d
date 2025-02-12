from pathlib import Path
import hydra
from omegaconf import DictConfig
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

from openlex3d import get_path


def plot_rank_barchart(ranks, title, save_path):
    """Plots and saves the rank distribution as a bar chart."""
    plt.figure(figsize=(10, 6))

    # Count occurrences of each rank
    rank_counts = Counter(ranks)
    sorted_ranks = sorted(rank_counts.keys())  # Ensure order is preserved
    counts = [rank_counts[r] for r in sorted_ranks]

    # Convert to NumPy arrays (ensures integer casting)
    sorted_ranks = np.array(sorted_ranks, dtype=int)
    counts = np.array(counts, dtype=int)

    # Create a bar plot
    sns.barplot(
        x=sorted_ranks, y=counts, hue=sorted_ranks, palette="viridis", legend=False
    )

    plt.xlabel("Rank")
    plt.ylabel("Count")
    plt.title(f"Rank Distribution - {title}")
    plt.xticks(rotation=45)  # Rotate x-axis labels for readability if needed
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


@hydra.main(
    version_base=None,
    config_path=f"{get_path()}/config",
    config_name="eval_query_config",
)
def main(cfg: DictConfig):
    iou_threshold = cfg.eval.iou_threshold

    ranks_output_path = (
        Path(cfg.output_path)
        / "rank_metric"
        / cfg.dataset.name
        / f"{cfg.pred.method}_{cfg.masks.alignment_mode}_{cfg.masks.alignment_threshold}_{cfg.eval.top_k}_{cfg.eval.iou_threshold}"
    )
    with open(ranks_output_path / "query_ranks.pkl", "rb") as f:
        query_ranks = pickle.load(f)

    charts_path = ranks_output_path / "barcharts"
    charts_path.mkdir(parents=True, exist_ok=True)

    all_ranks = []

    for scene_id, queries in query_ranks.items():
        scene_ranks = []
        for query_id, ranks in queries.items():
            scene_ranks.extend(ranks)

        all_ranks.extend(scene_ranks)

        # Plot bar chart for the scene
        plot_rank_barchart(
            scene_ranks,
            f"Method {cfg.pred.method} - Dataset {cfg.dataset.name} Scene {scene_id} - IoU {iou_threshold}",
            charts_path / f"scene_{scene_id}_bar.png",
        )

    # Plot overall bar chart
    plot_rank_barchart(
        all_ranks,
        f"Method {cfg.pred.method} - Dataset {cfg.dataset.name} All Scenes - IoU {iou_threshold}",
        charts_path / "all_scenes_bar.png",
    )


if __name__ == "__main__":
    main()
