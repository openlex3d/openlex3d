import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os


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


if __name__ == "__main__":
    pkl_path = "/home/kumaraditya/openlex3d/metric_dict_49a82360aa_0.01_0.9.pkl"  # Change this to your actual file path
    save_dir = "/home/kumaraditya/openlex3d/ap_plots_49a82360aa_0.01_0.9"  # Change this to your desired folder path
    metric_dict = load_metrics(pkl_path)
    plot_precision_recall(metric_dict, save_dir)
