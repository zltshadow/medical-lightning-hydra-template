from logging import Logger
import os
from pathlib import Path
from typing import List
import PIL
import hydra
import monai
import omegaconf
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch
import yaml
from src.data.lbl_datamodule import LBLDataModule
from src.models.components.resnet import ResNet
from src.models.lbl_module import LBLLitModule
from src.utils.instantiators import instantiate_loggers
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
import lightning as L
import torch.nn.functional as F


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


def compute_and_save_metrics(results_dict, save_name, ablation=False):
    model_names_map = (
        {
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
        if not ablation
        else {
            "mshm_000": "mshm_000",
            "mshm_001": "mshm_001",
            "mshm_010": "mshm_010",
            "mshm_011": "mshm_011",
            "mshm_100": "mshm_100",
            "mshm_101": "mshm_101",
            "mshm_110": "mshm_110",
            "mshm_111": "mshm_111",
        }
    )

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


def get_ablation_result():
    preds_list = read_dir(
        "logs/train/ablation", lambda x: x.endswith(".xlsx"), recursive=True
    )
    model_names = [
        "mshm_000",
        "mshm_001",
        "mshm_010",
        "mshm_011",
        "mshm_100",
        "mshm_101",
        "mshm_110",
        "mshm_111",
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
                results_dict["lbl"][cfg["tags"][0].lower()][cfg["data"]["fold"]] = {}
                results_dict["lbl"][cfg["tags"][0].lower()][cfg["data"]["fold"]][
                    "preds"
                ] = res_df["predictions"].tolist()
                results_dict["lbl"][cfg["tags"][0].lower()][cfg["data"]["fold"]][
                    "targets"
                ] = res_df["targets"]
                results_dict["lbl"][cfg["tags"][0].lower()][cfg["data"]["fold"]][
                    "probs"
                ] = res_df["probabilities"].tolist()
            else:
                results_dict["bra"][cfg["tags"][0].lower()][cfg["data"]["fold"]] = {}
                results_dict["bra"][cfg["tags"][0].lower()][cfg["data"]["fold"]][
                    "preds"
                ] = res_df["predictions"].tolist()
                results_dict["bra"][cfg["tags"][0].lower()][cfg["data"]["fold"]][
                    "targets"
                ] = res_df["targets"].tolist()
                results_dict["bra"][cfg["tags"][0].lower()][cfg["data"]["fold"]][
                    "probs"
                ] = res_df["probabilities"].tolist()

    # Call the function to compute and save the metrics table
    compute_and_save_metrics(results_dict["lbl"], "lbl_ablation", ablation=True)
    compute_and_save_metrics(results_dict["bra"], "bra_ablation", ablation=True)


def get_cam():
    pth_list = preds_list = read_dir(
        "logs/train/multiruns",
        lambda x: x.endswith(".ckpt") and "last" not in x,
        recursive=True,
    )
    for checkpoint_path in pth_list:
        model_name = Path(checkpoint_path).stem.lower()
        from omegaconf import DictConfig

        # 指定 Hydra 生成的 config.yaml 文件的路径
        hydra_config_path = f"{Path(checkpoint_path).parent.parent}/.hydra/config.yaml"
        # 加载配置文件
        cfg = omegaconf.OmegaConf.load(hydra_config_path)

        # if model_name in ["mshm"]:
        if cfg.data.fold == 1:
            # checkpoint = torch.load(checkpoint_path)

            # # 定义一个函数来修改键值
            # def rename_keys(state_dict):
            #     new_state_dict = {}
            #     for key, value in state_dict.items():
            #         # 检查键名中是否包含 'net.mamba_fusion_block.'
            #         if "net.mamba_fusion_block." in key:
            #             # 替换键名
            #             new_key = key.replace(
            #                 "net.mamba_fusion_block.", "net.fusion_block."
            #             )
            #             new_state_dict[new_key] = value
            #         else:
            #             new_state_dict[key] = value
            #     return new_state_dict

            # # 修改state dictionary中的键值
            # checkpoint["state_dict"] = rename_keys(checkpoint["state_dict"])
            # # 保存修改后的checkpoint到原始路径
            # torch.save(checkpoint, checkpoint_path)

            if cfg.get("seed"):
                L.seed_everything(cfg.seed, workers=True)
                monai.utils.set_determinism(cfg.seed)

            datamodule = hydra.utils.instantiate(cfg.data)
            model = hydra.utils.instantiate(cfg.model)
            # logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
            logger: List[Logger] = instantiate_loggers([])
            cfg.trainer.default_root_dir = ""
            trainer = hydra.utils.instantiate(
                cfg.trainer,
                logger=logger,
                deterministic=False,
                benchmark=True,
            )
            # test_res = trainer.test(
            #     model=model, datamodule=datamodule, ckpt_path=checkpoint_path
            # )

            predictions = trainer.predict(
                model=model,
                dataloaders=datamodule.test_dataloader(),
                ckpt_path=checkpoint_path,
            )
            probs, preds, targets = [], [], []
            for t in predictions:
                probs = probs + t[0].tolist()
                preds = preds + t[1].tolist()
                targets = targets + t[2].tolist()
            test_results = pd.DataFrame(
                {
                    "predictions": preds,
                    "targets": targets,
                    "probabilities": probs,
                }
            )
            # cfg.paths.output_dir = "output"
            # # Save the results to an Excel file
            # test_results.to_excel(
            #     os.path.join(
            #         cfg.paths.output_dir,
            #         f"{cfg.model.model_name.lower()}_fold{cfg.data.fold}_test_results.xlsx",
            #     ),
            #     index=False,
            # )
            # metric_dict = trainer.callback_metrics
            test_metrics = calculate_metrics(targets, preds, probs)


if __name__ == "__main__":
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    get_preds_result()
    get_ablation_result()
    get_cam()
