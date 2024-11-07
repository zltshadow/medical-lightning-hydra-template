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
    Resize,
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
    LoadImage,
)
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import compute_class_weight
import torch
import yaml

from monai.config import DtypeLike, KeysCollection, NdarrayOrTensor
from monai.data import image_writer
from monai.data.image_reader import ImageReader
from monai.transforms.io.array import LoadImage, SaveImage, WriteFileMapping
from monai.transforms.transform import MapTransform, Transform
from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep
from monai.utils.enums import PostFix

# # 测试模块导入的代码
# import os
# import sys
# print(os.getcwd())
# print(sys.path)
# # sys.path.append(os.getcwd())
# # print(sys.path)
from src.utils import utils

DEFAULT_POST_FIX = PostFix.meta()


class LoadMulImaged(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.LoadImage`,
    It can load both image data and metadata. When loading a list of files in one key,
    the arrays will be stacked and a new dimension will be added as the first dimension
    In this case, the metadata of the first image will be used to represent the stacked result.
    The affine transform of all the stacked images should be same.
    The output metadata field will be created as ``meta_keys`` or ``key_{meta_key_postfix}``.

    If reader is not specified, this class automatically chooses readers
    based on the supported suffixes and in the following order:

        - User-specified reader at runtime when calling this loader.
        - User-specified reader in the constructor of `LoadImage`.
        - Readers from the last to the first in the registered list.
        - Current default readers: (nii, nii.gz -> NibabelReader), (png, jpg, bmp -> PILReader),
          (npz, npy -> NumpyReader), (dcm, DICOM series and others -> ITKReader).

    Please note that for png, jpg, bmp, and other 2D formats, readers by default swap axis 0 and 1 after
    loading the array with ``reverse_indexing`` set to ``True`` because the spatial axes definition
    for non-medical specific file formats is different from other common medical packages.

    Note:

        - If `reader` is specified, the loader will attempt to use the specified readers and the default supported
          readers. This might introduce overheads when handling the exceptions of trying the incompatible loaders.
          In this case, it is therefore recommended setting the most appropriate reader as
          the last item of the `reader` parameter.

    See also:

        - tutorial: https://github.com/Project-MONAI/tutorials/blob/master/modules/load_medical_images.ipynb

    """

    def __init__(
        self,
        keys: KeysCollection,
        reader: type[ImageReader] | str | None = None,
        dtype: DtypeLike = np.float32,
        meta_keys: KeysCollection | None = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        image_only: bool = True,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        prune_meta_pattern: str | None = None,
        prune_meta_sep: str = ".",
        allow_missing_keys: bool = False,
        expanduser: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            reader: reader to load image file and metadata
                - if `reader` is None, a default set of `SUPPORTED_READERS` will be used.
                - if `reader` is a string, it's treated as a class name or dotted path
                (such as ``"monai.data.ITKReader"``), the supported built-in reader classes are
                ``"ITKReader"``, ``"NibabelReader"``, ``"NumpyReader"``.
                a reader instance will be constructed with the `*args` and `**kwargs` parameters.
                - if `reader` is a reader class/instance, it will be registered to this loader accordingly.
            dtype: if not None, convert the loaded image data to this data type.
            meta_keys: explicitly indicate the key to store the corresponding metadata dictionary.
                the metadata is a dictionary object which contains: filename, original_shape, etc.
                it can be a sequence of string, map to the `keys`.
                if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
            meta_key_postfix: if meta_keys is None, use `key_{postfix}` to store the metadata of the nifti image,
                default is `meta_dict`. The metadata is a dictionary object.
                For example, load nifti file for `image`, store the metadata into `image_meta_dict`.
            overwriting: whether allow overwriting existing metadata of same key.
                default is False, which will raise exception if encountering existing key.
            image_only: if True return dictionary containing just only the image volumes, otherwise return
                dictionary containing image data array and header dict per input key.
            ensure_channel_first: if `True` and loaded both image array and metadata, automatically convert
                the image array shape to `channel first`. default to `False`.
            simple_keys: whether to remove redundant metadata keys, default to False for backward compatibility.
            prune_meta_pattern: combined with `prune_meta_sep`, a regular expression used to match and prune keys
                in the metadata (nested dictionary), default to None, no key deletion.
            prune_meta_sep: combined with `prune_meta_pattern`, used to match and prune keys
                in the metadata (nested dictionary). default is ".", see also :py:class:`monai.transforms.DeleteItemsd`.
                e.g. ``prune_meta_pattern=".*_code$", prune_meta_sep=" "`` removes meta keys that ends with ``"_code"``.
            allow_missing_keys: don't raise exception if key is missing.
            expanduser: if True cast filename to Path and call .expanduser on it, otherwise keep filename as is.
            args: additional parameters for reader if providing a reader name.
            kwargs: additional parameters for reader if providing a reader name.
        """
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadImage(
            reader,
            image_only,
            dtype,
            ensure_channel_first,
            simple_keys,
            prune_meta_pattern,
            prune_meta_sep,
            expanduser,
            *args,
            **kwargs,
        )
        if not isinstance(meta_key_postfix, str):
            raise TypeError(
                f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}."
            )
        self.meta_keys = (
            ensure_tuple_rep(None, len(self.keys))
            if meta_keys is None
            else ensure_tuple(meta_keys)
        )
        if len(self.keys) != len(self.meta_keys):
            raise ValueError(
                f"meta_keys should have the same length as keys, got {len(self.keys)} and {len(self.meta_keys)}."
            )
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting

    def register(self, reader: ImageReader):
        self._loader.register(reader)

    def __call__(self, data, reader: ImageReader | None = None):
        """
        Raises:
            KeyError: When not ``self.overwriting`` and key already exists in ``data``.

        """
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(
            d, self.meta_keys, self.meta_key_postfix
        ):
            if key in ["image_path", "seg_path"]:
                t1_data = self._loader(d[key][0], reader)
                t2_data = self._loader(d[key][1], reader)
                t1c_data = self._loader(d[key][2], reader)
                if key == "seg_path":
                    mode = "nearest"
                else:
                    mode = "bilinear"
                preprocessor = Compose([Resize(spatial_size=(512, 512, 16), mode=mode)])
                preprocess_t1_data = preprocessor(t1_data)
                preprocess_t2_data = preprocessor(t2_data)
                preprocess_t1c_data = preprocessor(t1c_data)
                data = torch.concat(
                    (
                        preprocess_t1_data,
                        preprocess_t2_data,
                        preprocess_t1c_data,
                    ),
                    dim=-1,
                )
                if key == "image_path":
                    d["image"] = data
                else:
                    d["seg"] = data
            # data = self._loader(d[key], reader)
            # if self._loader.image_only:
            #     d[key] = data
            # else:
            #     if not isinstance(data, (tuple, list)):
            #         raise ValueError(
            #             f"loader must return a tuple or list (because image_only=False was used), got {type(data)}."
            #         )
            #     d[key] = data[0]
            #     if not isinstance(data[1], dict):
            #         raise ValueError(f"metadata must be a dict, got {type(data[1])}.")
            #     meta_key = meta_key or f"{key}_{meta_key_postfix}"
            #     if meta_key in d and not self.overwriting:
            #         raise KeyError(
            #             f"Metadata with key {meta_key} already exists and overwriting=False."
            #         )
            #     d[meta_key] = data[1]
        return d


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
        seed: int = 42,
        task_name: str = "",
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
        self.seed = seed
        self.task_name = task_name

        with open(dataset_json, "r") as f:
            self.dataset_json_content = json.load(f)
        with open(splits_final_json, "r") as f:
            self.splits_final_json_content = json.load(f)

        self.dataset_func = CacheDataset
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        dataset_name = "LBL_raw_BJTR"
        dataset_name = "LBL_all_tumor"
        dataset_name = "LBL_all"
        all_data_df = pd.read_excel(
            f"/data/zlt/projects/data/{dataset_name}/{dataset_name}.xlsx"
        )
        all_data_df = all_data_df[
            (all_data_df["来源医院"].isin(["北京同仁"]))
            # & (all_data_df["flag"] == 1)
            # & (all_data_df["MagneticFieldStrength"] != 1.5)
        ]

        # 划分训练_验证224*0.8=179例、测试集224*0.2=45例
        split_ratio = 0.8
        train_val_df, test_df = train_test_split(
            all_data_df,
            train_size=split_ratio,
            stratify=all_data_df["病理级别"].values,
            random_state=self.seed,
        )

        train_val_df = train_val_df.iloc[
            natsorted(
                range(len(train_val_df)),
                key=lambda i: train_val_df["输出文件夹"].iloc[i],
            )
        ].reset_index(drop=True)
        test_df = test_df.iloc[
            natsorted(range(len(test_df)), key=lambda i: test_df["输出文件夹"].iloc[i])
        ].reset_index(drop=True)

        # cross-validation
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits)
        splits = list(skf.split(train_val_df, train_val_df["病理级别"].values))
        train_df = train_val_df
        val_df = test_df
        if fold in range(n_splits):
            train_index, val_index = splits[fold]
            train_df = train_val_df.iloc[train_index]
            val_df = train_val_df.iloc[val_index]

        # image_loader = Compose(
        #     [
        #         LoadImage(reader=ITKReader, ensure_channel_first=True),
        #         # CropForegroundd(
        #         #     keys=["image", "seg"], allow_smaller=False, source_key="image"
        #         # ),
        #         Resized(keys=["image", "seg"], spatial_size=self.input_size),
        #     ],
        #     lazy=True,
        # )

        data_root = f"/data/zlt/projects/data/{dataset_name}"
        image_prefix = "Nii"
        seg_prefix = "Label"
        self.image_prefix = "image"
        self.seg_prefix = "seg"
        self.label_prefix = "label"
        self.train_rawlist = {}
        self.val_rawlist = {}
        self.test_rawlist = {}
        # "image"[file1,file2,file3]
        # "seg"[file1,file2,file3]
        # "label" 0,1
        # 这是按患者划分的
        for _, data in train_df.iterrows():
            patient_dir = data["输出文件夹"]
            self.train_rawlist[patient_dir] = {}
            self.train_rawlist[patient_dir][self.image_prefix] = {}
            self.train_rawlist[patient_dir][self.seg_prefix] = {}
            for seq_name in self.sequences.values():
                self.train_rawlist[patient_dir][self.image_prefix][
                    seq_name
                ] = f"{data_root}/{patient_dir}/{seq_name}/{image_prefix}/{seq_name}.nii.gz"
                self.train_rawlist[patient_dir][self.seg_prefix][
                    seq_name
                ] = f"{data_root}/{patient_dir}/{seq_name}/{seg_prefix}/{seq_name}_Label.nii.gz"
                self.train_rawlist[patient_dir][self.label_prefix] = data["病理级别"]
        for _, data in val_df.iterrows():
            patient_dir = data["输出文件夹"]
            self.val_rawlist[patient_dir] = {}
            self.val_rawlist[patient_dir][self.image_prefix] = {}
            self.val_rawlist[patient_dir][self.seg_prefix] = {}
            for seq_name in self.sequences.values():
                self.val_rawlist[patient_dir][self.image_prefix][
                    seq_name
                ] = f"{data_root}/{patient_dir}/{seq_name}/{image_prefix}/{seq_name}.nii.gz"
                self.val_rawlist[patient_dir][self.seg_prefix][
                    seq_name
                ] = f"{data_root}/{patient_dir}/{seq_name}/{seg_prefix}/{seq_name}_Label.nii.gz"
                self.val_rawlist[patient_dir][self.label_prefix] = data["病理级别"]
        for _, data in test_df.iterrows():
            patient_dir = data["输出文件夹"]
            self.test_rawlist[patient_dir] = {}
            self.test_rawlist[patient_dir][self.image_prefix] = {}
            self.test_rawlist[patient_dir][self.seg_prefix] = {}
            for seq_name in self.sequences.values():
                self.test_rawlist[patient_dir][self.image_prefix][
                    seq_name
                ] = f"{data_root}/{patient_dir}/{seq_name}/{image_prefix}/{seq_name}.nii.gz"
                self.test_rawlist[patient_dir][self.seg_prefix][
                    seq_name
                ] = f"{data_root}/{patient_dir}/{seq_name}/{seg_prefix}/{seq_name}_Label.nii.gz"
                self.test_rawlist[patient_dir][self.label_prefix] = data["病理级别"]
        if fold in ["extest", "train_val"]:
            # 外部验证集
            self.extest_rawlist = {}
            # dataset_name = "LBL_raw_extest"
            # extest_dir = f"/data/zlt/projects/data/{dataset_name}/"
            # extest_df = pd.read_excel(f"{extest_dir}/{dataset_name}.xlsx")

            extest_dir = f"/data/zlt/projects/data/{dataset_name}/"
            extest_df = pd.read_excel(
                f"/data/zlt/projects/data/{dataset_name}/{dataset_name}.xlsx"
            )
            extest_df = extest_df[extest_df["来源医院"].isin(["吉大二院", "湘雅二院"])]
            for _, data in extest_df.iterrows():
                patient_dir = data["输出文件夹"]
                self.extest_rawlist[patient_dir] = {}
                self.extest_rawlist[patient_dir][self.image_prefix] = {}
                self.extest_rawlist[patient_dir][self.seg_prefix] = {}
                for seq_name in self.sequences.values():
                    # 特殊处理湘雅二院的T2,只有T2_AX_FS(12，14，22是nFS)
                    if (
                        data["来源医院"] in ["湘雅二院"]
                        and seq_name == "T2_AX_nFS"
                        and not os.path.exists(
                            f"{extest_dir}/{patient_dir}/{seq_name}/{seg_prefix}/{seq_name}_Label.nii.gz"
                        )
                    ):
                        seq_name = "T2_AX_FS"

                    self.extest_rawlist[patient_dir][self.image_prefix][
                        seq_name
                    ] = f"{extest_dir}/{patient_dir}/{seq_name}/{image_prefix}/{seq_name}.nii.gz"
                    self.extest_rawlist[patient_dir][self.seg_prefix][
                        seq_name
                    ] = f"{extest_dir}/{patient_dir}/{seq_name}/{seg_prefix}/{seq_name}_Label.nii.gz"
                    self.extest_rawlist[patient_dir][self.label_prefix] = data[
                        "病理级别"
                    ]

            self.test_rawlist = self.extest_rawlist

        # # 这是按序列划分的
        # for seq_name in self.sequences.values():
        #     self.train_rawlist[seq_name] = {}
        #     self.train_rawlist[seq_name][self.image_prefix] = [
        #         f"{data_root}/{i}/{seq_name}/{image_prefix}/{seq_name}.nii.gz"
        #         for i in train_val_df["输出文件夹"]
        #     ]
        #     self.train_rawlist[seq_name][self.seg_prefix] = [
        #         f"{data_root}/{i}/{seq_name}/{seg_prefix}/{seq_name}_{seg_prefix}.nii.gz"
        #         for i in train_val_df["输出文件夹"]
        #     ]
        #     self.train_rawlist[seq_name][self.label_prefix] = train_val_df[
        #         "病理级别"
        #     ].values

        #     self.val_rawlist[seq_name] = {}
        #     self.val_rawlist[seq_name][self.image_prefix] = [
        #         f"{data_root}/{i}/{seq_name}/{image_prefix}/{seq_name}.nii.gz"
        #         for i in test_df["输出文件夹"]
        #     ]
        #     self.val_rawlist[seq_name][self.seg_prefix] = [
        #         f"{data_root}/{i}/{seq_name}/{seg_prefix}/{seq_name}_{seg_prefix}.nii.gz"
        #         for i in test_df["输出文件夹"]
        #     ]
        #     self.val_rawlist[seq_name][self.label_prefix] = test_df["病理级别"].values

        #     self.test_rawlist[seq_name] = {}
        #     self.test_rawlist[seq_name][self.image_prefix] = [
        #         f"{data_root}/{i}/{seq_name}/{image_prefix}/{seq_name}.nii.gz"
        #         for i in test_df["输出文件夹"]
        #     ]
        #     self.test_rawlist[seq_name][self.seg_prefix] = [
        #         f"{data_root}/{i}/{seq_name}/{seg_prefix}/{seq_name}_{seg_prefix}.nii.gz"
        #         for i in test_df["输出文件夹"]
        #     ]
        #     self.test_rawlist[seq_name][self.label_prefix] = test_df["病理级别"].values

        # 定义变换
        self.train_transforms = Compose(
            [
                LoadMulImaged(
                    keys=["image_path", "seg_path"],
                    reader=ITKReader,
                    ensure_channel_first=True,
                ),
                NormalizeIntensityd(
                    keys=["image"],
                ),
                # range_z 围绕Z轴旋转，0.3弧度大约是17度， 度= 弧度 × π / 180
                RandRotated(
                    keys=["image", "seg"],
                    range_x=0.0,
                    range_y=0.0,
                    range_z=0.3,
                    prob=0.1,
                    mode=["bilinear", "nearest"],
                ),
                # X轴翻转
                RandFlipd(
                    keys=["image", "seg"],
                    prob=0.1,
                    spatial_axis=[0],
                ),
                RandScaleIntensityd(
                    keys=["image"],
                    factors=0.1,
                    prob=0.1,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.1,
                    prob=0.1,
                ),
                RandZoomd(
                    keys=["image", "seg"],
                    min_zoom=0.9,
                    max_zoom=1.1,
                    prob=0.1,
                    mode=["bilinear", "nearest"],
                ),
                ToTensord(
                    keys=["image", "seg", "label"],
                    track_meta=False,
                ),
            ],
            lazy=True,
        )
        self.val_transforms = Compose(
            [
                LoadMulImaged(
                    keys=["image_path", "seg_path"],
                    reader=ITKReader,
                    ensure_channel_first=True,
                ),
                NormalizeIntensityd(
                    keys=["image"],
                ),
                ToTensord(
                    keys=["image", "seg", "label"],
                    track_meta=False,
                ),
            ],
            lazy=True,
        )
        self.test_transforms = Compose(
            [
                LoadMulImaged(
                    keys=["image_path", "seg_path"],
                    reader=ITKReader,
                    ensure_channel_first=True,
                ),
                NormalizeIntensityd(
                    keys=["image"],
                ),
                ToTensord(
                    keys=["image", "seg", "label"],
                    track_meta=False,
                ),
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
        for data in self.train_rawlist.values():
            train_data_list.append(
                {
                    "image_path": [i for i in data[self.image_prefix].values()],
                    "seg_path": [i for i in data[self.seg_prefix].values()],
                    "label": data[self.label_prefix],
                }
            )
        val_data_list = []
        for data in self.val_rawlist.values():
            val_data_list.append(
                {
                    "image_path": [i for i in data[self.image_prefix].values()],
                    "seg_path": [i for i in data[self.seg_prefix].values()],
                    "label": data[self.label_prefix],
                }
            )
        test_data_list = []
        for data in self.test_rawlist.values():
            test_data_list.append(
                {
                    "image_path": [i for i in data[self.image_prefix].values()],
                    "seg_path": [i for i in data[self.seg_prefix].values()],
                    "label": data[self.label_prefix],
                }
            )
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = self.dataset_func(
                data=train_data_list,
                transform=self.train_transforms,
                cache_rate=1 if self.task_name == "train" else 0,
            )
            self.data_val = self.dataset_func(
                data=val_data_list,
                transform=self.val_transforms,
                cache_rate=1 if self.task_name == "train" else 0,
            )
            self.data_test = self.dataset_func(
                data=test_data_list,
                transform=self.test_transforms,
                cache_rate=1 if self.task_name == "eval" else 0,
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
    seg_train = lbl_dataset.data_train[0]["seg"][0].cpu().numpy().astype(np.int8)

    image_val = lbl_dataset.data_val[0]["image"][0]
    seg_val = lbl_dataset.data_val[0]["seg"][0].cpu().numpy().astype(np.int8)

    image_test = lbl_dataset.data_test[0]["image"][0]
    seg_test = lbl_dataset.data_test[0]["seg"][0].cpu().numpy().astype(np.int8)

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
        return image
        # return np.transpose(image, (1, 0))
        # return np.rot90(image, k=1, axes=(0, 1))

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
    plt.savefig(f"lbldata_test_fold_{fold}.jpg", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"save lbldata_test_fold_{fold}.jpg")

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
    # # 输出训练集数据第一个图像跟标签是否都存在
    # print(first_data["image"].shape, lbl_dataset.train_images[0])
    # print(first_data["seg"].shape, lbl_dataset.train_segs[0])
    # print(
    #     os.path.isfile(lbl_dataset.train_images[0])
    #     and os.path.isfile(lbl_dataset.train_segs[0])
    # )
