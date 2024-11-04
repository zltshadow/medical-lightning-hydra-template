import json
import os
from typing import Any, Dict, Optional, Tuple, Union
from matplotlib import pyplot as plt
from natsort import natsorted
import numpy as np
from lightning import LightningDataModule
from monai.data import CacheDataset, DataLoader, ITKReader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    NormalizeIntensityd,
    CropForegroundd,
    Resized,
    Spacingd,
    CenterSpatialCropd,
    SpatialCropd,
    SpatialPadd,
    RandRotated,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandZoomd,
    ToTensord,
)
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import compute_class_weight
import torch
import yaml

# # 测试模块导入的代码
# import os
# import sys
# print(os.getcwd())
# print(sys.path)
# # sys.path.append(os.getcwd())
# # print(sys.path)
from src.utils import utils


class LBLDataModule(LightningDataModule):
    """`LightningDataModule` for the LBL dataset.

    The LBL database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        fold: Union[int, str] = 0,
        sequence_idx: str = "0000",
        dataset_json: str = "",
        splits_final_json: str = "",
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
        input_size: list = [128, 128, 32],
    ) -> None:
        """Initialize a `LBLDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()
        self.sequences: Dict[str, Any] = {
            "0000": "T1_AX_nFS",
            "0001": "T2_AX_nFS",
            "0002": "T1C_AX_FS",
        }
        self.fold = fold
        self.data_dir = data_dir
        self.sequence_idx = sequence_idx
        self.sequence_name = self.sequences[sequence_idx]
        self.input_size = input_size

        with open(dataset_json, "r") as f:
            self.dataset_json_content = json.load(f)
        with open(splits_final_json, "r") as f:
            self.splits_final_json_content = json.load(f)

        if self.fold in ["all", "extest"]:
            # 使用全部数据训练，查看网络是否能过拟合
            self.train_images = [
                os.path.join(self.data_dir, i["image"])
                for i in self.dataset_json_content["training"]
                + self.dataset_json_content["test"]
            ]
            self.train_segs = [
                os.path.join(self.data_dir, i["label"])
                for i in self.dataset_json_content["training"]
                + self.dataset_json_content["test"]
            ]
            self.train_labels = [
                i["flag"]
                for i in self.dataset_json_content["training"]
                + self.dataset_json_content["test"]
            ]
            self.val_images = self.train_images
            self.val_segs = self.train_segs
            self.val_labels = self.train_labels

            extest_dir = "/data/zlt/projects/data/LBL_raw"
            extest_df = pd.read_excel(f"{extest_dir}/LBL_raw.xlsx")
            extest_df = extest_df[extest_df["来源医院"].isin(["南昌附二"])]
            first_sequence_name = f"{self.sequence_name}"
            extest_images = [
                f"{extest_dir}/{i}/{first_sequence_name}/Nii/{first_sequence_name}.nii.gz"
                for i in extest_df["输出文件夹"]
            ]
            extest_segs = [
                f"{extest_dir}/{i}/{first_sequence_name}/Label/{first_sequence_name}_Label.nii.gz"
                for i in extest_df["输出文件夹"]
            ]
            extest_labels = np.array([i for i in extest_df["病理级别"]])
            # 不需要变为one_hot模式
            # extest_labels = torch.nn.functional.one_hot(
            #     torch.as_tensor(extest_labels).to(torch.int64)
            # ).float()
            self.test_images = extest_images
            self.test_segs = extest_segs
            self.test_labels = extest_labels
        elif self.fold == "train_val":
            all_data_df = pd.read_excel(
                "/data/zlt/projects/data/LBL_raw_BJTR/LBL_raw_BJTR.xlsx"
            )
            # 划分训练_验证180例、测试集44例
            split_ratio = 180
            train_val_df, test_df = train_test_split(
                all_data_df,
                train_size=split_ratio,
                stratify=all_data_df["病理级别"].values,
                random_state=66,
            )

            train_val_df = train_val_df.iloc[
                natsorted(
                    range(len(train_val_df)),
                    key=lambda i: train_val_df["输出文件夹"].iloc[i],
                )
            ].reset_index(drop=True)
            test_df = test_df.iloc[
                natsorted(
                    range(len(test_df)), key=lambda i: test_df["输出文件夹"].iloc[i]
                )
            ].reset_index(drop=True)

            self.train_images = [
                os.path.join(
                    "/data/zlt/projects/data/LBL_raw_BJTR",
                    i,
                    (
                        f"{self.sequence_name }"
                        if "+" not in self.sequence_name
                        else self.sequence_name.split("+")[0]
                    ),
                    f"Nii",
                    (
                        f"{self.sequence_name }.nii.gz"
                        if "+" not in self.sequence_name
                        else self.sequence_name.split("+")[0] + ".nii.gz"
                    ),
                )
                for i in train_val_df["输出文件夹"]
            ]
            self.train_segs = [
                os.path.join(
                    "/data/zlt/projects/data/LBL_raw_BJTR",
                    i,
                    (
                        f"{self.sequence_name}"
                        if "+" not in self.sequence_name
                        else self.sequence_name.split("+")[0]
                    ),
                    f"Label",
                    (
                        f"{self.sequence_name}_Label.nii.gz"
                        if "+" not in self.sequence_name
                        else self.sequence_name.split("+")[0] + ".nii.gz"
                    ),
                )
                for i in train_val_df["输出文件夹"]
            ]
            self.train_labels = train_val_df["病理级别"].values

            self.val_images = [
                os.path.join(
                    "/data/zlt/projects/data/LBL_raw_BJTR",
                    i,
                    (
                        f"{self.sequence_name }"
                        if "+" not in self.sequence_name
                        else self.sequence_name.split("+")[0]
                    ),
                    f"Nii",
                    (
                        f"{self.sequence_name }.nii.gz"
                        if "+" not in self.sequence_name
                        else self.sequence_name.split("+")[0] + ".nii.gz"
                    ),
                )
                for i in test_df["输出文件夹"]
            ]
            self.val_segs = [
                os.path.join(
                    "/data/zlt/projects/data/LBL_raw_BJTR",
                    i,
                    (
                        f"{self.sequence_name}"
                        if "+" not in self.sequence_name
                        else self.sequence_name.split("+")[0]
                    ),
                    f"Label",
                    (
                        f"{self.sequence_name}_Label.nii.gz"
                        if "+" not in self.sequence_name
                        else self.sequence_name.split("+")[0] + ".nii.gz"
                    ),
                )
                for i in test_df["输出文件夹"]
            ]
            self.val_labels = test_df["病理级别"].values

            self.test_images = [
                os.path.join(
                    "/data/zlt/projects/data/LBL_raw_BJTR",
                    i,
                    (
                        f"{self.sequence_name }"
                        if "+" not in self.sequence_name
                        else self.sequence_name.split("+")[0]
                    ),
                    f"Nii",
                    (
                        f"{self.sequence_name }.nii.gz"
                        if "+" not in self.sequence_name
                        else self.sequence_name.split("+")[0] + ".nii.gz"
                    ),
                )
                for i in test_df["输出文件夹"]
            ]
            self.test_segs = [
                os.path.join(
                    "/data/zlt/projects/data/LBL_raw_BJTR",
                    i,
                    (
                        f"{self.sequence_name}"
                        if "+" not in self.sequence_name
                        else self.sequence_name.split("+")[0]
                    ),
                    f"Label",
                    (
                        f"{self.sequence_name}_Label.nii.gz"
                        if "+" not in self.sequence_name
                        else self.sequence_name.split("+")[0] + ".nii.gz"
                    ),
                )
                for i in test_df["输出文件夹"]
            ]
            self.test_labels = test_df["病理级别"].values

            # # 使用训练验证集训练，查看网络是否能过拟合
            # self.train_images = [
            #     os.path.join(self.data_dir, i["image"])
            #     for i in self.dataset_json_content["training"]
            # ]
            # self.train_segs = [
            #     os.path.join(self.data_dir, i["label"])
            #     for i in self.dataset_json_content["training"]
            # ]
            # self.train_labels = [
            #     i["flag"] for i in self.dataset_json_content["training"]
            # ]
            # self.val_images = self.extract_data("image", "test")
            # self.val_segs = self.extract_data("label", "test")
            # self.val_labels = self.extract_data("flag", "test")
            # 提取测试数据
            # self.test_images = self.extract_data("image", "test")
            # self.test_segs = self.extract_data("label", "test")
            # self.test_labels = self.extract_data("flag", "test")
        elif self.fold == "reg":
            all_data_df = pd.read_excel(
                "/data/zlt/projects/data/LBL_raw_BJTR/LBL_raw_BJTR.xlsx"
            )
            # 划分训练_验证180例、测试集44例
            split_ratio = 180
            train_val_df, test_df = train_test_split(
                all_data_df,
                train_size=split_ratio,
                stratify=all_data_df["病理级别"].values,
                random_state=66,
            )

            train_val_df = train_val_df.iloc[
                natsorted(
                    range(len(train_val_df)),
                    key=lambda i: train_val_df["输出文件夹"].iloc[i],
                )
            ].reset_index(drop=True)
            test_df = test_df.iloc[
                natsorted(
                    range(len(test_df)), key=lambda i: test_df["输出文件夹"].iloc[i]
                )
            ].reset_index(drop=True)
            data_root = "/data/zlt/projects/OrbitalMamba/outputs/antspy_reg/"
            self.train_images = [
                os.path.join(data_root, i, f"{self.sequence_name}_antspy_reg.nii.gz")
                for i in train_val_df["输出文件夹"]
            ]
            self.train_segs = [
                os.path.join(data_root, i, f"{self.sequence_name}_antspy_reg_label.nii.gz")
                for i in train_val_df["输出文件夹"]
            ]
            self.train_labels = train_val_df["病理级别"].values

            self.val_images = [
                os.path.join(data_root, i, f"{self.sequence_name}_antspy_reg.nii.gz")
                for i in test_df["输出文件夹"]
            ]
            self.val_segs = [
                os.path.join(data_root, i, f"{self.sequence_name}_antspy_reg_label.nii.gz")
                for i in test_df["输出文件夹"]
            ]
            self.val_labels = test_df["病理级别"].values

            self.test_images = [
                os.path.join(data_root, i, f"{self.sequence_name}_antspy_reg.nii.gz")
                for i in test_df["输出文件夹"]
            ]
            self.test_segs = [
                os.path.join(data_root, i, f"{self.sequence_name}_antspy_reg_label.nii.gz")
                for i in test_df["输出文件夹"]
            ]
            self.test_labels = test_df["病理级别"].values
        else:
            # 提取训练数据
            self.train_images = self.extract_data("image", "train")
            self.train_segs = self.extract_data("label", "train")
            self.train_labels = self.extract_data("flag", "train")
            # 提取验证数据
            self.val_images = self.extract_data("image", "val")
            self.val_segs = self.extract_data("label", "val")
            self.val_labels = self.extract_data("flag", "val")
            # 提取测试数据
            self.test_images = self.extract_data("image", "test")
            self.test_segs = self.extract_data("label", "test")
            self.test_labels = self.extract_data("flag", "test")

        self.dataset_func = CacheDataset
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # 定义变换, lazy在不用Orientation时是可以的
        self.train_transforms = Compose(
            [
                LoadImaged(keys=["image", "seg"], reader=ITKReader),
                EnsureChannelFirstd(keys=["image", "seg"]),
                Orientationd(keys=["image", "seg"], axcodes="RAS"),
                NormalizeIntensityd(
                    keys=["image", "seg"],
                ),
                CropForegroundd(
                    keys=["image", "seg"], allow_smaller=False, source_key="image"
                ),
                Resized(keys=["image", "seg"], spatial_size=self.input_size),
                # CenterSpatialCropd(
                #     keys=["image", "seg"],
                #     roi_size=(
                #         self.input_size[1],
                #         -1,
                #         self.input_size[2],
                #     ),
                # ),
                # SpatialCropd(
                #     keys=["image", "seg"],
                #     roi_start=(0, 0, 0),
                #     roi_end=tuple(self.input_size),
                # ),
                # SpatialPadd(keys=["image", "seg"], spatial_size=tuple(self.input_size)),
                # RandRotated(
                #     keys=["image", "seg"],
                #     range_x=0.1,
                #     range_y=0.0,
                #     range_z=0.0,
                #     prob=1,
                # ),
                # RandFlipd(
                #     keys=["image", "seg"],
                #     prob=0.1,
                #     spatial_axis=[0],
                # ),
                # RandFlipd(
                #     keys=["image", "seg"],
                #     prob=0.1,
                #     spatial_axis=[1],
                # ),
                # RandScaleIntensityd(
                #     keys=["image"],
                #     factors=0.1,
                #     prob=0.1,
                # ),
                # RandShiftIntensityd(
                #     keys=["image"],
                #     offsets=0.1,
                #     prob=0.1,
                # ),
                # RandZoomd(
                #     keys=["image", "seg"],
                #     min_zoom=0.9,
                #     max_zoom=1.1,
                #     prob=0.1,
                # ),
                ToTensord(keys=["image", "seg", "label"], track_meta=False),
            ],
            lazy=True,
        )
        self.val_transforms = Compose(
            [
                LoadImaged(keys=["image", "seg"], reader=ITKReader),
                EnsureChannelFirstd(
                    keys=["image", "seg"],
                ),
                Orientationd(keys=["image", "seg"], axcodes="RAS"),
                NormalizeIntensityd(
                    keys=["image", "seg"],
                ),
                CropForegroundd(
                    keys=["image", "seg"], allow_smaller=False, source_key="image"
                ),
                Resized(keys=["image", "seg"], spatial_size=self.input_size),
                # CenterSpatialCropd(
                #     keys=["image", "seg"],
                #     roi_size=(
                #         self.input_size[1],
                #         -1,
                #         self.input_size[2],
                #     ),
                # ),
                # SpatialCropd(
                #     keys=["image", "seg"],
                #     roi_start=(0, 0, 0),
                #     roi_end=tuple(self.input_size),
                # ),
                SpatialPadd(keys=["image", "seg"], spatial_size=tuple(self.input_size)),
                ToTensord(keys=["image", "seg", "label"], track_meta=False),
            ],
            lazy=True,
        )
        self.test_transforms = Compose(
            [
                LoadImaged(keys=["image", "seg"], reader=ITKReader),
                EnsureChannelFirstd(
                    keys=["image", "seg"],
                ),
                Orientationd(keys=["image", "seg"], axcodes="RAS"),
                NormalizeIntensityd(
                    keys=["image", "seg"],
                ),
                CropForegroundd(
                    keys=["image", "seg"], allow_smaller=False, source_key="image"
                ),
                Resized(keys=["image", "seg"], spatial_size=self.input_size),
                # CenterSpatialCropd(
                #     keys=["image", "seg"],
                #     roi_size=(
                #         self.input_size[1],
                #         -1,
                #         self.input_size[2],
                #     ),
                # ),
                # SpatialCropd(
                #     keys=["image", "seg"],
                #     roi_start=(0, 0, 0),
                #     roi_end=tuple(self.input_size),
                # ),
                # SpatialPadd(keys=["image", "seg"], spatial_size=tuple(self.input_size)),
                ToTensord(keys=["image", "seg", "label"], track_meta=False),
            ],
            lazy=True,
        )

        # 处理外部数据集
        if fold == "extest":
            self.test_transforms = Compose(
                [
                    LoadImaged(keys=["image", "seg"], reader=ITKReader),
                    EnsureChannelFirstd(
                        keys=["image", "seg"],
                    ),
                    Orientationd(keys=["image", "seg"], axcodes="RAS"),
                    NormalizeIntensityd(
                        keys=["image", "seg"],
                    ),
                    # Spacingd(
                    #     keys=["image", "seg"],
                    #     pixdim=(1, 1, 1),
                    # ),
                    CropForegroundd(
                        keys=["image", "seg"], allow_smaller=False, source_key="image"
                    ),
                    Resized(keys=["image", "seg"], spatial_size=self.input_size),
                    # CenterSpatialCropd(
                    #     keys=["image", "seg"],
                    #     roi_size=(
                    #         self.input_size[1],
                    #         -1,
                    #         self.input_size[2],
                    #     ),
                    # ),
                    # SpatialCropd(
                    #     keys=["image", "seg"],
                    #     roi_start=(0, 0, 0),
                    #     roi_end=tuple(self.input_size),
                    # ),
                    # SpatialPadd(
                    #     keys=["image", "seg"], spatial_size=tuple(self.input_size)
                    # ),
                    ToTensord(keys=["image", "seg", "label"], track_meta=False),
                ],
                lazy=True,
            )

        self.data_train: Optional[CacheDataset] = None
        self.data_val: Optional[CacheDataset] = None
        self.data_test: Optional[CacheDataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of LBL classes (10).
        """
        return 2

    def extract_data(self, key, dataset_type="train"):
        """Extract data from the dataset based on the given key, dataset type, and fold."""
        splits_key_mapping = {
            "train": "training",
            "val": "training",
            "test": "test",
        }

        if dataset_type == "test":
            data_list = [
                (
                    i[key]
                    if key != "image"
                    else i[key].replace(f"_0000.", f"_{self.sequence_idx}.")
                )
                for i in self.dataset_json_content[splits_key_mapping[dataset_type]]
            ]
        else:
            data_list = [
                (
                    i[key]
                    if key != "image"
                    else i[key].replace(f"_0000.", f"_{self.sequence_idx}.")
                )
                for i in self.dataset_json_content[splits_key_mapping[dataset_type]]
                if i["image"][9:].replace(
                    f"_0000{self.dataset_json_content['file_ending']}",
                    "",
                )
                in self.splits_final_json_content[self.fold][dataset_type]
            ]
        # 拼接数据集目录
        if key != "flag":
            data_list = [os.path.join(self.data_dir, i) for i in data_list]

        return data_list

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        train_data_list = []
        for data in zip(self.train_images, self.train_segs, self.train_labels):
            train_data_list.append(
                {
                    "image": data[0],
                    "seg": data[1],
                    "label": data[2],
                }
            )

        val_data_list = []
        for data in zip(self.val_images, self.val_segs, self.val_labels):
            val_data_list.append(
                {
                    "image": data[0],
                    "seg": data[1],
                    "label": data[2],
                }
            )

        test_data_list = []
        for data in zip(self.test_images, self.test_segs, self.test_labels):
            test_data_list.append(
                {
                    "image": data[0],
                    "seg": data[1],
                    "label": data[2],
                }
            )

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = self.dataset_func(
                data=train_data_list,
                transform=self.train_transforms,
                cache_num=0,
                cache_rate=0,
                num_workers=0,
            )
            self.data_val = self.dataset_func(
                data=val_data_list,
                transform=self.val_transforms,
                cache_num=0,
                cache_rate=0,
                num_workers=0,
            )
            self.data_test = self.dataset_func(
                data=test_data_list,
                transform=self.test_transforms,
                cache_num=0,
                cache_rate=0,
                num_workers=0,
            )

    def train_dataloader(self):
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    utils.add_torch_shape_forvs()
    with open("configs/data/lbl.yaml", "r", encoding="utf-8") as f:
        data_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    data_dir = data_config["data_dir"]
    dataset_json = data_config["dataset_json"]
    splits_final_json = data_config["splits_final_json"]
    fold = 0
    fold = "extest"
    fold = "train_val"
    # 默认模态是0000，测试0001
    lbl_dataset = LBLDataModule(
        data_dir=data_dir,
        fold=fold,
        sequence_idx="0001",
        dataset_json=dataset_json,
        splits_final_json=splits_final_json,
    )
    lbl_dataset.setup()

    # 获取训练、验证和测试集的第一个样本
    image_train = lbl_dataset.data_train[0]["image"][0]
    seg_train = lbl_dataset.data_train[0]["seg"][0]

    image_val = lbl_dataset.data_val[0]["image"][0]
    seg_val = lbl_dataset.data_val[0]["seg"][0]

    image_test = lbl_dataset.data_test[0]["image"][0]
    seg_test = lbl_dataset.data_test[0]["seg"][0]

    # 打印图像和分割的形状
    print("Train Image Shape:", image_train.shape)
    print("Train Segmentation Shape:", seg_train.shape)
    print("Validation Image Shape:", image_val.shape)
    print("Validation Segmentation Shape:", seg_val.shape)
    print("Test Image Shape:", image_test.shape)
    print("Test Segmentation Shape:", seg_test.shape)

    # 计算中间切片的索引
    mid_idx_train = image_train.shape[-1] // 2
    mid_idx_val = image_val.shape[-1] // 2
    mid_idx_test = image_test.shape[-1] // 2

    # 提取中间切片
    image_slice_train = image_train[:, :, mid_idx_train]
    seg_slice_train = seg_train[:, :, mid_idx_train]

    image_slice_val = image_val[:, :, mid_idx_val]
    seg_slice_val = seg_val[:, :, mid_idx_val]

    image_slice_test = image_test[:, :, mid_idx_test]
    seg_slice_test = seg_test[:, :, mid_idx_test]

    # 创建拼图，2行3列
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # 2行3列

    # 定义旋转函数
    def rotate_image(image):
        return np.rot90(image, k=1, axes=(0, 1))

    # 训练集
    axs[0, 0].imshow(rotate_image(image_slice_train), cmap="gray")
    axs[0, 0].set_title("Train Image Slice")
    axs[0, 0].axis("off")

    axs[1, 0].imshow(rotate_image(seg_slice_train), cmap="jet", alpha=0.5)
    axs[1, 0].set_title("Train Segmentation Slice")
    axs[1, 0].axis("off")

    # 验证集
    axs[0, 1].imshow(rotate_image(image_slice_val), cmap="gray")
    axs[0, 1].set_title("Validation Image Slice")
    axs[0, 1].axis("off")

    axs[1, 1].imshow(rotate_image(seg_slice_val), cmap="jet", alpha=0.5)
    axs[1, 1].set_title("Validation Segmentation Slice")
    axs[1, 1].axis("off")

    # 测试集
    axs[0, 2].imshow(rotate_image(image_slice_test), cmap="gray")
    axs[0, 2].set_title("Test Image Slice")
    axs[0, 2].axis("off")

    axs[1, 2].imshow(rotate_image(seg_slice_test), cmap="jet", alpha=0.5)
    axs[1, 2].set_title("Test Segmentation Slice")
    axs[1, 2].axis("off")

    plt.tight_layout()  # 调整子图间距
    plt.show()
    plt.savefig(f"lbldata_test_fold{fold}.jpg", dpi=300, bbox_inches="tight")
    plt.close()

    # BCEWithLogitsLoss损失函数权重获取
    train_class_weights = compute_class_weight(
        "balanced",
        classes=np.unique([0, 1]),
        y=[int(i["label"]) for i in lbl_dataset.data_train],
    )
    val_class_weights = compute_class_weight(
        "balanced",
        classes=np.unique([0, 1]),
        y=[int(i["label"]) for i in lbl_dataset.data_val],
    )
    test_class_weights = compute_class_weight(
        "balanced",
        classes=np.unique([0, 1]),
        y=[int(i["label"]) for i in lbl_dataset.data_test],
    )
    print(train_class_weights)
    # 将权重转换为 PyTorch 张量
    weight_tensor = torch.tensor(train_class_weights, dtype=torch.float)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight_tensor[1])

    first_data = next(iter(lbl_dataset.train_dataloader()))
    # 输出训练集数据第一个图像跟标签是否都存在
    print(first_data["image"].shape, lbl_dataset.train_images[0])
    print(first_data["seg"].shape, lbl_dataset.train_segs[0])
    print(
        os.path.isfile(lbl_dataset.train_images[0])
        and os.path.isfile(lbl_dataset.train_segs[0])
    )
