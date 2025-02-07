import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

from openlex3d import get_path
from omegaconf import DictConfig
import hydra
from pathlib import Path


def load_metrics(pkl_path):
    with open(pkl_path, "rb") as f:
        metric_dict = pickle.load(f)
    return metric_dict


def plot_precision_recall(metric_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    sns.set_style("whitegrid")

    for label, overlaps in metric_dict.items():
        for overlap_th, metrics in overlaps.items():
            plt.figure(figsize=(8, 6))
            precision = metrics["precision"]
            recall = metrics["recall"]

            # remove last prec and recall value
            precision = precision[:-1]
            recall = recall[:-1]

            sns.lineplot(
                x=recall,
                y=precision,
                label=f"{label} (AP={metrics['ap']:.2f}, Overlap={overlap_th})",
            )

            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve: {label}, Overlap {overlap_th}")
            plt.legend()
            save_path = os.path.join(
                save_dir, f"precision_recall_curve_{label}_overlap_{overlap_th}.png"
            )
            plt.savefig(save_path, dpi=300)
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

    pickle_path = str(Path(cfg.output_path) / f"metric_dict_{file_save_string}.pkl")
    save_dir = str(Path(cfg.output_path) / f"ap_plots_{file_save_string}.pkl")

    metric_dict = load_metrics(pickle_path)
    plot_precision_recall(metric_dict, save_dir)


if __name__ == "__main__":
    main()
