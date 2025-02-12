import os
from pathlib import Path
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
from openlex3d import get_path


def load_metrics(pkl_path):
    with open(pkl_path, "rb") as f:
        metric_dict = pickle.load(f)
    return metric_dict


def plot_precision_recall(title, metric_dict, save_dir):
    sns.set_style("whitegrid")

    # Plot all overlap thresholds for each label
    for label, overlaps in metric_dict.items():
        plt.figure(figsize=(8, 6))

        for overlap_th, metrics in overlaps.items():
            precision = metrics["precision"][:-1]  # Remove last precision value
            recall = metrics["recall"][:-1]  # Remove last recall value

            sns.lineplot(
                x=recall,
                y=precision,
                label=f"Overlap {overlap_th:.2f} (AP={metrics['ap']:.2f})",
            )

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve: (All Overlaps) - {title}")
        plt.legend()
        save_path = os.path.join(save_dir, "precision_recall_curve_all_overlaps.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

    # Plot only for overlap thresholds 0.25 and 0.5
    for specific_overlap in [0.25, 0.5]:
        plt.figure(figsize=(8, 6))

        for label, overlaps in metric_dict.items():
            if specific_overlap in overlaps:
                precision = overlaps[specific_overlap]["precision"][:-1]
                recall = overlaps[specific_overlap]["recall"][:-1]
                ap_value = overlaps[specific_overlap]["ap"]

                sns.lineplot(
                    x=recall,
                    y=precision,
                    label=f"{label} (AP={ap_value:.2f})",
                )

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve: Overlap {specific_overlap} - {title}")
        plt.legend()
        save_path = os.path.join(
            save_dir, f"precision_recall_curve_overlap_{specific_overlap}.png"
        )
        plt.savefig(save_path, dpi=300)
        plt.close()


@hydra.main(
    version_base=None,
    config_path=f"{get_path()}/config",
    config_name="eval_query_config",
)
def main(cfg: DictConfig):
    ap_output_path = (
        Path(cfg.output_path)
        / "ap_metric"
        / cfg.dataset.name
        / f"{cfg.pred.method}_{cfg.masks.alignment_mode}_{cfg.masks.alignment_threshold}_{cfg.eval.criteria}_{cfg.eval.clip_threshold}_{cfg.eval.top_k}"
    )
    metric_dict_path = str(ap_output_path / "ap_metrics.pkl")

    """
    metric_dict[label_name][overlap_th] = {
                        "precision": precision,
                        "recall": recall,
                        "ap": ap_current,
                    }
    """
    metric_dict = load_metrics(metric_dict_path)

    title = f"Method {cfg.pred.method} - Dataset {cfg.dataset.name}"

    prec_recall_curves_path = ap_output_path / "prec_recall_curves"
    prec_recall_curves_path.mkdir(parents=True, exist_ok=True)
    plot_precision_recall(title, metric_dict, prec_recall_curves_path)


if __name__ == "__main__":
    main()
