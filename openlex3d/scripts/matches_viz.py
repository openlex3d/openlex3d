import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from openlex3d import get_path
from omegaconf import DictConfig
import hydra
from pathlib import Path


def plot_match_statistics(pickle_path, save_path):
    """
    Reads match statistics from a pickle file and plots histograms for the entire scene.
    Saves the plots as a single image.
    """
    # Load match statistics
    with open(pickle_path, "rb") as f:
        match_stats = pickle.load(f)

    # Extract data for plotting
    before_intersection = list(match_stats["before_intersection"].values())
    after_intersection = list(match_stats["after_intersection"].values())
    total_gt_instances = list(match_stats["total_gt_instances"].values())

    # Determine common x-axis range
    min_val = min(
        min(before_intersection), min(after_intersection), min(total_gt_instances)
    )
    max_val = max(
        max(before_intersection), max(after_intersection), max(total_gt_instances)
    )
    bins = list(np.arange(min_val - 1, max_val + 1))
    print(bins)

    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sns.histplot(before_intersection, bins=bins, kde=True, ax=axes[0], color="blue")
    axes[0].set_title("Pred Matches Before Intersection")
    axes[0].set_xlabel("Number of Matches")
    axes[0].set_ylabel("Frequency")

    sns.histplot(after_intersection, bins=bins, kde=True, ax=axes[1], color="green")
    axes[1].set_title("Pred Matches After Intersection")
    axes[1].set_xlabel("Number of Matches")
    axes[1].set_ylabel("Frequency")

    sns.histplot(total_gt_instances, bins=bins, kde=True, ax=axes[2], color="red")
    axes[2].set_title("Total GT Instances per Query ID")
    axes[2].set_xlabel("Number of GT Instances")
    axes[2].set_ylabel("Frequency")

    # Ensure consistent x-axis across all plots
    for ax in axes:
        ax.set_xlim(min_val, max_val + 1)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


@hydra.main(
    version_base=None,
    config_path=f"{get_path()}/config",
    config_name="eval_query_config",
)
def main(cfg: DictConfig):
    if cfg.eval.criteria == "clip_threshold":
        file_save_string = f"{cfg.scene_id}_{cfg.method}_{cfg.masks.alignment_mode}_{cfg.masks.alignment_threshold}_{cfg.eval.criteria}_{cfg.eval.clip_threshold}"
    elif cfg.eval.criteria == "top_k":
        file_save_string = f"{cfg.scene_id}_{cfg.method}_{cfg.masks.alignment_mode}_{cfg.masks.alignment_threshold}_{cfg.eval.criteria}_{cfg.eval.top_k}"
    else:
        raise ValueError(
            "Invalid evaluation criteria: choose 'clip_threshold' or 'top_k'"
        )

    pickle_path = str(Path(cfg.output_path) / f"match_stats_{file_save_string}.pkl")
    save_path = pickle_path.replace(".pkl", ".png")

    plot_match_statistics(pickle_path, save_path)


if __name__ == "__main__":
    main()
