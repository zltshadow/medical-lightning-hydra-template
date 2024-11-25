import json
import os
from typing import Any, Dict, Optional, Tuple, Union
import PIL
from matplotlib import pyplot as plt
from natsort import natsorted
import numpy as np
from lightning import LightningDataModule
from monai.data import CacheDataset, DataLoader, ITKReader
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    Resize,
    ToTensor,
)
from sklearn.model_selection import (
    StratifiedShuffleSplit,
)
from sklearn.utils import compute_class_weight
import torch
import yaml
from monai.transforms.transform import MapTransform, Transform
from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep
from monai.utils.enums import PostFix
from pydoc import locate

# # 测试模块导入的代码
# import os
# import sys
# print(os.getcwd())
# print(sys.path)
# # sys.path.append(os.getcwd())
# # print(sys.path)
from src.utils import utils
from monai.data.image_reader import (
    PILReader,
)

DEFAULT_POST_FIX = PostFix.meta()


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
        batch_size: int = 64,
        num_workers: int = 0,
        input_size: list = [224, 224],
        seed: int = 42,
        task_name: str = "",
        dataset_name: str = "",
        num_classes: int = 2,
        fold: Union[int, str] = 0,
        pin_memory: bool = False,
        in_channels: int = 3,
        **kwargs,
    ) -> None:
        super().__init__()
        self.fold = fold
        self.data_dir = data_dir
        self.input_size = input_size
        self.seed = seed
        self.task_name = task_name
        self.num_classes = num_classes

        self.dataset_func = CacheDataset
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        # dataset_name = "BraTS_TCGA_resample_2d"
        # dataset_name = "LBL_all_reg_resample_2d"

        # 获取数据目录
        data_dir = os.path.join(data_dir, dataset_name, "Nii")

        # 获取类别名称
        class_names = sorted(
            x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))
        )
        num_class = len(class_names)

        # 获取每个类别的图像文件路径
        t1_image_files = [
            [
                x
                for x in utils.read_dir(os.path.join(data_dir, class_names[i]))
                if "T1_" in x
            ]
            for i in range(num_class)
        ]
        t2_image_files = [
            [
                x
                for x in utils.read_dir(os.path.join(data_dir, class_names[i]))
                if "T2_" in x
            ]
            for i in range(num_class)
        ]
        t1c_image_files = [
            [
                x
                for x in utils.read_dir(os.path.join(data_dir, class_names[i]))
                if "T1C_" in x
            ]
            for i in range(num_class)
        ]
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
                # Resize((224, 224)),
                ToTensor(
                    track_meta=False,
                ),
            ]
        )

        train_ds = LBLDataset(train_x, train_y, train_transforms)
        val_ds = LBLDataset(val_x, val_y, val_transforms)
        test_ds = LBLDataset(final_test_x, final_test_y, val_transforms)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, num_workers=num_workers
        )

        self.data_train = train_ds
        self.data_val = val_ds
        self.data_test = test_ds

        self.batch_size_per_device = batch_size

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
    input_size = data_config["input_size"]
    dataset_name = "BraTs_TCGA_resample_2d"
    dataset_name = "LBL_all_reg_resample_2d"
    fold = 0
    fold = "extest"
    fold = "train_val"
    # 默认模态是0000，测试0001
    lbl_dataset = LBLDataModule(
        data_dir=data_dir,
        fold=fold,
        sequence_idx="0001",
        dataset_name=dataset_name,
        input_size=input_size,
    )
    lbl_dataset.setup()

    # 获取训练、验证和测试集的第一个样本
    image_train = lbl_dataset.data_train[0][0]
    image_val = lbl_dataset.data_val[0][0]
    image_test = lbl_dataset.data_test[0][0]

    # 打印图像和分割的形状
    print("Train Image Shape:", image_train.shape)
    print("Validation Image Shape:", image_val.shape)
    print("Test Image Shape:", image_test.shape)

    image_slice_train = image_train[0, :, :]
    image_slice_val = image_val[0, :, :]
    image_slice_test = image_test[0, :, :]

    # 创建拼图，2行3列
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # 2行3列

    # 定义旋转函数
    def rotate_image(image):
        # return image
        return np.transpose(image, (1, 0))
        # return np.rot90(image, k=1, axes=(0, 1))

    # 训练集
    axs[0, 0].imshow(rotate_image(image_slice_train), cmap="gray")
    axs[0, 0].set_title("Train Image Slice")
    axs[0, 0].axis("off")

    axs[1, 0].imshow(rotate_image(image_slice_train), cmap="jet", alpha=0.5)
    axs[1, 0].set_title("Train Segmentation Slice")
    axs[1, 0].axis("off")

    # 验证集
    axs[0, 1].imshow(rotate_image(image_slice_val), cmap="gray")
    axs[0, 1].set_title("Validation Image Slice")
    axs[0, 1].axis("off")

    axs[1, 1].imshow(rotate_image(image_slice_val), cmap="jet", alpha=0.5)
    axs[1, 1].set_title("Validation Segmentation Slice")
    axs[1, 1].axis("off")

    # 测试集
    axs[0, 2].imshow(rotate_image(image_slice_test), cmap="gray")
    axs[0, 2].set_title("Test Image Slice")
    axs[0, 2].axis("off")

    axs[1, 2].imshow(rotate_image(image_slice_test), cmap="jet", alpha=0.5)
    axs[1, 2].set_title("Test Segmentation Slice")
    axs[1, 2].axis("off")

    plt.tight_layout()  # 调整子图间距
    plt.show()
    plt.savefig(f"lbldata_test_fold_{fold}.jpg", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"save lbldata_test_fold_{fold}.jpg")

    # # BCEWithLogitsLoss损失函数权重获取
    # train_class_weights = compute_class_weight(
    #     "balanced",
    #     classes=np.unique([0, 1]),
    #     y=[int(i[1]) for i in lbl_dataset.data_train],
    # )
    # val_class_weights = compute_class_weight(
    #     "balanced",
    #     classes=np.unique([0, 1]),
    #     y=[int(i[1]) for i in lbl_dataset.data_val],
    # )
    # test_class_weights = compute_class_weight(
    #     "balanced",
    #     classes=np.unique([0, 1]),
    #     y=[int(i[1]) for i in lbl_dataset.data_test],
    # )
    # print(train_class_weights)
    # # 将权重转换为 PyTorch 张量
    # weight_tensor = torch.tensor(train_class_weights, dtype=torch.float)
    # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight_tensor[1])

    first_data = next(iter(lbl_dataset.train_dataloader()))
    # 输出训练集数据第一个数据
    print(first_data[0].shape)
    print(first_data[1].shape)
