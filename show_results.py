import os
from pathlib import Path
import pandas as pd
import yaml
from src.utils.utils import read_dir
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import numpy as np


def get_preds_result():
    preds_list = read_dir(
        "logs/train/multiruns", lambda x: x.endswith(".xlsx"), recursive=True
    )
    model_names = [
        "resnet",
        "senet",
        "convnext",
        "vit",
        "gcvit",
        "swint",
        "medmamba",
        "nnmamba",
        "vim",
        "mshm",
    ]
    results_dict = {
        "lbl": {i: {} for i in model_names},
        "bra": {i: {} for i in model_names},
    }
    for excel in preds_list:
        with open(
            f"{Path(excel).parent}/.hydra/config.yaml", "r", encoding="utf-8"
        ) as f:
            res_df = pd.read_excel(excel)
            res_df["predictions"]
            res_df["targets"]
            res_df["probabilities"]
            cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
            if cfg["data"]["dataset_name"] == "LBL_all_reg_resample_2d":
                results_dict["lbl"][cfg["model"]["model_name"].lower()][
                    cfg["data"]["fold"]
                ] = {}
                results_dict["lbl"][cfg["model"]["model_name"].lower()][
                    cfg["data"]["fold"]
                ]["preds"] = res_df["predictions"].tolist()
                results_dict["lbl"][cfg["model"]["model_name"].lower()][
                    cfg["data"]["fold"]
                ]["targets"] = res_df["targets"]
                results_dict["lbl"][cfg["model"]["model_name"].lower()][
                    cfg["data"]["fold"]
                ]["probs"] = res_df["probabilities"].tolist()
            else:
                results_dict["bra"][cfg["model"]["model_name"].lower()][
                    cfg["data"]["fold"]
                ] = {}
                results_dict["bra"][cfg["model"]["model_name"].lower()][
                    cfg["data"]["fold"]
                ]["preds"] = res_df["predictions"].tolist()
                results_dict["bra"][cfg["model"]["model_name"].lower()][
                    cfg["data"]["fold"]
                ]["targets"] = res_df["targets"].tolist()
                results_dict["bra"][cfg["model"]["model_name"].lower()][
                    cfg["data"]["fold"]
                ]["probs"] = res_df["probabilities"].tolist()

    # Call the function to compute and save the metrics table
    compute_and_save_metrics(results_dict["lbl"], "lbl")
    compute_and_save_metrics(results_dict["bra"], "bra")

    # Plot ROC curves
    plot_roc_curve(results_dict["lbl"], "lbl")
    plot_roc_curve(results_dict["bra"], "bra")


def compute_and_save_metrics(results_dict, save_name):
    model_names_map = {
        "resnet": "ResNet",
        "senet": "SENet",
        "convnext": "ConvNeXt",
        "vit": "ViT",
        "gcvit": "GCViT",
        "swint": "SwinT",
        "medmamba": "MedMamba",
        "nnmamba": "nnMamba",
        "vim": "ViM",
        "mshm": "MSHM",
    }

    # Prepare a dictionary to store average values and std for each model
    avg_metrics_dict = {model: [] for model in model_names_map.keys()}
    fold_metrics_dict = {model: [] for model in model_names_map.keys()}

    for model_name, folds in results_dict.items():
        for fold_name, data in folds.items():
            preds = data["preds"]
            probs = data["probs"]
            target = data["targets"]

            # Calculate performance metrics for each fold
            auc_value, acc, f1, precision, recall = calculate_metrics(
                target, preds, probs
            )

            # Store per-fold metrics
            fold_metrics_dict[model_name].append(
                [auc_value, acc, f1, precision, recall]
            )

            # Store metrics for averaging
            avg_metrics_dict[model_name].append([auc_value, acc, f1, precision, recall])

    # Calculate mean and std for each model across all folds (without sensitivity)
    avg_df = pd.DataFrame(
        {
            key: [
                f"{np.mean(np.array(val), axis=0)[0] * 100:.2f} ± {np.std(np.array(val), axis=0)[0] * 100:.2f}",  # AUC
                f"{np.mean(np.array(val), axis=0)[1] * 100:.2f} ± {np.std(np.array(val), axis=0)[1] * 100:.2f}",  # ACC
                f"{np.mean(np.array(val), axis=0)[2] * 100:.2f} ± {np.std(np.array(val), axis=0)[2] * 100:.2f}",  # F1
                f"{np.mean(np.array(val), axis=0)[3] * 100:.2f} ± {np.std(np.array(val), axis=0)[3] * 100:.2f}",  # PRE
                f"{np.mean(np.array(val), axis=0)[4] * 100:.2f} ± {np.std(np.array(val), axis=0)[4] * 100:.2f}",  # REC
            ]
            for key, val in avg_metrics_dict.items()
        }
    ).transpose()

    avg_df.columns = ["AUC", "ACC", "F1", "PRE", "REC"]

    avg_df.to_excel(f"{output_dir}/{save_name}_average_metrics.xlsx", index=True)

    # Convert per-fold metrics to DataFrame
    fold_df = pd.DataFrame(fold_metrics_dict).transpose()
    fold_df.columns = ["AUC", "ACC", "F1", "PRE", "REC"]
    fold_df.to_excel(f"{output_dir}/{save_name}_fold_metrics.xlsx", index=True)


def calculate_metrics(targets, preds, probs):
    # Compute AUC, Accuracy, F1, Precision, Recall
    fpr, tpr, _ = roc_curve(targets, probs)
    auc_value = auc(fpr, tpr)

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds)
    precision = precision_score(targets, preds)
    recall = recall_score(targets, preds)

    return auc_value, acc, f1, precision, recall


def plot_roc_curve(results_dict, save_name):
    model_names_map = {
        "resnet": "ResNet",
        "senet": "SENet",
        "convnext": "ConvNeXt",
        "vit": "ViT",
        "gcvit": "GCViT",
        "swint": "SwinT",
        "medmamba": "MedMamba",
        "nnmamba": "nnMamba",
        "vim": "ViM",
        "mshm": "MSHM",
    }

    # Set font and style
    plt.rcParams.update(
        {
            "font.size": 14,
            "font.family": "Times New Roman",
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    # Initialize the figure
    plt.figure(figsize=(10, 8))

    # Color list for models
    colors = ["b", "g", "r", "c", "m", "y", "k", "orange", "purple", "brown"]

    for i, (model_name, folds) in enumerate(results_dict.items()):
        all_fpr = []
        all_tpr = []
        auc_values = []

        for fold_name, data in folds.items():
            preds = data["preds"]
            probs = data["probs"]
            target = data["targets"]

            fpr, tpr, _ = roc_curve(target, probs)
            roc_auc = auc(fpr, tpr)

            all_fpr.append(fpr)
            all_tpr.append(tpr)
            auc_values.append(roc_auc)

        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.mean(
            [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(all_fpr, all_tpr)],
            axis=0,
        )
        mean_auc = np.mean(auc_values)

        color = colors[i % len(colors)]
        plt.plot(
            mean_fpr,
            mean_tpr,
            color=color,
            lw=2,
            label=f"{model_names_map[model_name]} (AUC = {mean_auc * 100:.2f})",
        )

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{save_name}_roc.jpg", dpi=300)


if __name__ == "__main__":
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    get_preds_result()
