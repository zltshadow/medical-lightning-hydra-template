from logging import Logger
import os
from pathlib import Path
import shutil
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
from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    Resize,
    ToTensor,
)
from monai.data import DataLoader, PILReader
import cv2
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import torch


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
            if cfg["data"]["dataset_name"] == "LBL_all_reg_resample_2d" and "lbl_ablation" not in cfg["tags"]:
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
            elif "bra_ablation" not in cfg["tags"]:
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


def get_cam(dataset_name):
    output_prefix = dataset_name.split("_")[0][:3].lower()
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
        if cfg.data.fold == 0 and cfg.data.dataset_name == dataset_name:
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

    shutil.move("outputs/cam", f"outputs/{output_prefix}_cam")


class LoadMulImage:
    def __init__(self, reader, resize=False):
        self.reader = reader()
        self.resize = resize

    def __call__(self, filename):
        """
        Load image file and metadata from the given filename(s).
        If `reader` is not specified, this class automatically chooses readers based on the
        reversed order of registered readers `self.readers`.

        Args:
            filename: path file or file-like object or a list of files.
                will save the filename to meta_data with key `filename_or_obj`.
                if provided a list of files, use the filename of first file to save,
                and will stack them together as multi-channels data.
                if provided directory path instead of file path, will treat it as
                DICOM images series and read.
            reader: runtime reader to load image file and metadata.

        """
        t1_img = torch.from_numpy(np.array(self.reader.read(filename[0]))).unsqueeze(
            dim=0
        )
        t2_img = torch.from_numpy(np.array(self.reader.read(filename[1]))).unsqueeze(
            dim=0
        )
        t1c_img = torch.from_numpy(np.array(self.reader.read(filename[2]))).unsqueeze(
            dim=0
        )
        if self.resize:
            processor = Resize((224, 224))
            t1_img = processor(t1_img)
            t2_img = processor(t2_img)
            t1c_img = processor(t1c_img)
        img = torch.concat((t1_img, t2_img, t1c_img), dim=0)
        return img


class LBLDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms
        self.load_data = LoadMulImage(reader=PILReader, resize=False)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img = self.load_data(self.image_files[index])
        return self.transforms(img), self.labels[index]


def get_cam_gt_mask(dataset_name):
    # dataset_name = "BraTs_TCGA_2d"
    # dataset_name = "LBL_all_reg_resample_2d"
    output_prefix = dataset_name.split("_")[0][:3].lower()
    # 获取数据目录
    data_dir = "../data"
    seed = 42
    set_determinism(seed=seed)
    batch_size = 32
    num_workers = 0

    # 获取数据目录
    data_dir = os.path.join(data_dir, dataset_name, "Nii")

    # 获取类别名称
    class_names = sorted(
        x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))
    )
    num_class = len(class_names)

    # 获取每个类别的图像文件路径
    t1_image_files = [
        [x for x in read_dir(os.path.join(data_dir, class_names[i])) if "T1_" in x]
        for i in range(num_class)
    ]
    t2_image_files = [
        [x for x in read_dir(os.path.join(data_dir, class_names[i])) if "T2_" in x]
        for i in range(num_class)
    ]
    t1c_image_files = [
        [x for x in read_dir(os.path.join(data_dir, class_names[i])) if "T1C_" in x]
        for i in range(num_class)
    ]

    print(len(t1_image_files[0]), len(t1_image_files[1]))
    print(len(t2_image_files[0]), len(t2_image_files[1]))
    print(len(t1c_image_files[0]), len(t1c_image_files[1]))
    image_files_list = []
    image_class = []
    for i in range(num_class):
        # 使用 zip 函数按顺序将 t1、t2、t1c 图像路径打包成元组
        for t1_img, t2_img, t1c_img in zip(
            t1_image_files[i], t2_image_files[i], t1c_image_files[i]
        ):
            image_files_list.append((t1_img, t2_img, t1c_img))
            image_class.append(i)

    # # 获取图像尺寸
    # image_width, image_height = PIL.Image.open(image_files_list[0]).size

    # print(f"Total image count: {len(image_class)}")
    # print(f"Image dimensions: {image_width} x {image_height}")
    # print(f"Label names: {class_names}")
    # print(f"Label counts: {[len(image_files[i]) for i in range(num_class)]}")

    # 设置交叉验证的折数
    n_splits = 5
    fold = 0

    # 使用StratifiedKFold进行五折交叉验证
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # 创建一个分割对象，存储所有的训练集和验证集的索引
    splits = list(skf.split(image_files_list, image_class))

    if fold in range(n_splits):
        train_index, val_index = splits[fold]

        # 根据索引分割数据集
        train_x, val_x = [image_files_list[i] for i in train_index], [
            image_files_list[i] for i in val_index
        ]
        train_y, val_y = [image_class[i] for i in train_index], [
            image_class[i] for i in val_index
        ]

        final_test_x = val_x
        final_test_y = val_y
    elif fold == "train_val":
        # 使用StratifiedShuffleSplit进行分层抽样
        split = StratifiedShuffleSplit(
            n_splits=1, test_size=0.4, train_size=0.6, random_state=seed
        )
        for train_index, test_index in split.split(image_files_list, image_class):
            train_x, test_x = [image_files_list[i] for i in train_index], [
                image_files_list[i] for i in test_index
            ]
            train_y, test_y = [image_class[i] for i in train_index], [
                image_class[i] for i in test_index
            ]

        # 进一步将测试集分为验证集和测试集，比例为1:1
        test_split = StratifiedShuffleSplit(
            n_splits=1, test_size=0.5, train_size=0.5, random_state=seed
        )
        for val_index, final_test_index in test_split.split(test_x, test_y):
            val_x, final_test_x = [test_x[i] for i in val_index], [
                test_x[i] for i in final_test_index
            ]
            val_y, final_test_y = [test_y[i] for i in val_index], [
                test_y[i] for i in final_test_index
            ]

    print(
        f"Training count: {len(train_x)}, Validation count: {len(val_x)}, Test count: {len(final_test_x)}"
    )

    # 计算并打印每个数据集中每个类别的数量
    def print_class_counts(data_x, data_y, name):
        class_counts = {i: 0 for i in range(num_class)}
        for label in data_y:
            class_counts[label] += 1
        print(f"\n{name} set class counts:")
        for class_name, count in zip(class_names, class_counts.values()):
            print(f"{class_name}: {count}")

    print_class_counts(train_x, train_y, "Training")
    print_class_counts(val_x, val_y, "Validation")
    print_class_counts(final_test_x, final_test_y, "Test")

    train_transforms = Compose(
        [
            # EnsureChannelFirst(),
            ScaleIntensity(),
            # CropForeground(),
            # Resize((224, 224)),
            RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
            ToTensor(
                track_meta=False,
            ),
        ],
        lazy=True,
    )

    val_transforms = Compose(
        [
            # EnsureChannelFirst(),
            ScaleIntensity(),
            # CropForeground(),
            # Resize((224, 224)),
            ToTensor(
                track_meta=False,
            ),
        ]
    )
    for data in train_x:
        # data[0].replace("T1_AX_nFS", ""), data[1].replace("T2_AX_nFS", "").replace("T2_AX_FS", ""), data[2].replace("T1C_AX_FS", "")
        # 检查三个元素是否相等
        if (
            not data[0].replace("T1_AX_nFS", "")
            == data[1].replace("T2_AX_nFS", "").replace("T2_AX_FS", "")
            == data[2].replace("T1C_AX_FS", "")
        ):
            print("Not all elements are equal.")
    for data in final_test_x:
        # data[0].replace("T1_AX_nFS", ""), data[1].replace("T2_AX_nFS", "").replace("T2_AX_FS", ""), data[2].replace("T1C_AX_FS", "")
        # 检查三个元素是否相等
        if (
            not data[0].replace("T1_AX_nFS", "")
            == data[1].replace("T2_AX_nFS", "").replace("T2_AX_FS", "")
            == data[2].replace("T1C_AX_FS", "")
        ):
            print("Not all elements are equal.")

    # 打开文件用于写入 train_x 的数据
    with open(f"outputs/{output_prefix}_train_x_data.txt", "w") as train_file:
        for data in train_x:
            # 将 data 写入文件，每个元组占一行，元素之间用空格分隔
            train_file.write(f"{data[0]} {data[1]} {data[2]}\n")

    # 打开文件用于写入 final_test_x 的数据
    with open(f"outputs/{output_prefix}_final_test_x_data.txt", "w") as test_file:
        for data in final_test_x:
            # 将 data 写入文件，每个元组占一行，元素之间用空格分隔
            test_file.write(f"{data[0]} {data[1]} {data[2]}\n")

    train_ds = LBLDataset(train_x, train_y, train_transforms)
    val_ds = LBLDataset(val_x, val_y, val_transforms)
    test_ds = LBLDataset(final_test_x, final_test_y, val_transforms)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)
    for batch_idx, batch in enumerate(test_loader):
        image = batch[0]

        for i in range(batch[0].shape[0]):
            for seq_idx in range(3):
                # 获取灰度图像并转换为 RGB
                gray_img = image[i, seq_idx].cpu().detach().numpy()
                rgb_img = np.stack([gray_img] * 3, axis=-1)

                # 输出目录
                output_dir = f"outputs/{output_prefix}_cam/{batch_idx}/{i}"
                os.makedirs(output_dir, exist_ok=True)

                rgb_img = np.uint8(255 * rgb_img)
                # 保存原图（灰度图）
                cv2.imwrite(f"{output_dir}/{seq_idx}_0_gt.jpg", rgb_img)

                # 构建标签路径
                seg_path = (
                    final_test_x[batch_idx * batch_size + i][seq_idx].replace(
                        "Nii", "Label"
                    )[:-4]
                    + "_Label"
                    + ".png"
                )

                # 读取标签图像并进行轮廓提取
                label_img = cv2.imread(
                    seg_path, cv2.IMREAD_GRAYSCALE
                )  # 假设标签是灰度图

                if label_img is not None:
                    # 二值化标签图像：假设标签是0和1，或者是其他类别值
                    _, binary_label = cv2.threshold(
                        label_img, 1, 255, cv2.THRESH_BINARY
                    )

                    # 找到标签图像中的轮廓
                    contours, _ = cv2.findContours(
                        binary_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    # 在原图上绘制轮廓
                    cv2.drawContours(
                        rgb_img, contours, -1, (128, 174, 128), 2
                    )  # 绿色轮廓，线宽为2

                    # 保存包含轮廓的图像
                    cv2.imwrite(
                        f"{output_dir}/{seq_idx}_0_mask.jpg",
                        rgb_img,
                    )

                else:
                    print(f"Warning: Label image not found at {seg_path}")


if __name__ == "__main__":
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    get_preds_result()
    get_ablation_result()

    get_cam("LBL_all_reg_resample_2d")
    get_cam_gt_mask("LBL_all_reg_resample_2d")

    get_cam("BraTs_TCGA_2d")
    get_cam_gt_mask("BraTs_TCGA_2d")
