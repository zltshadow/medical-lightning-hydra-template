import json
from typing import Any, Dict, Optional, Tuple
import numpy as np
from lightning import LightningDataModule
from monai.data import ImageDataset, DataLoader
from monai.transforms import Compose, EnsureChannelFirst, Orientation, NormalizeIntensity, CropForeground, Resize, \
    RandRotate, RandFlip, RandScaleIntensity, RandShiftIntensity, RandZoom
from src.utils import utils
import pandas as pd


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
            train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
            batch_size: int = 2,
            num_workers: int = 0,
            pin_memory: bool = False,
    ) -> None:
        """Initialize a `LBLDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()
        self.fold = 0
        self.sequences: Dict[str, Any] = {
            "0000": 'T1',
            "0001": 'T2',
            "0002": 'T1C',
        }
        self.dataset_json = json.dumps(r"E:\projects\BIT\data\nnUNet_datasets\nnUNet_raw\Dataset803_LBL_raw_BJTR\dataset.json")
        self.train_images = self.dataset_json['training']
        self.test_images = self.dataset_json['testing']

        self.dataset_func = ImageDataset
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # 定义变换
        self.train_transforms = Compose(
            [
                EnsureChannelFirst(),
                NormalizeIntensity(),
                Orientation(axcodes='RAS', lazy=True, ),
                CropForeground(allow_smaller=False, lazy=True, ),
                Resize(self.hparams.input_size),
                RandRotate(range_x=0.3, range_y=0.0, range_z=0.0, prob=0.1, lazy=True, ),
                RandFlip(prob=0.1, spatial_axis=[0], lazy=True, ),
                RandFlip(prob=0.1, spatial_axis=[1], lazy=True, ),
                RandScaleIntensity(factors=0.1, prob=0.1, ),
                RandShiftIntensity(offsets=0.1, prob=0.1, ),
                RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.1, lazy=True, ),
            ]
        )
        self.val_transforms = Compose(
            [
                EnsureChannelFirst(),
                NormalizeIntensity(),
                Orientation(axcodes='RAS', lazy=True),
                CropForeground(allow_smaller=False, lazy=True),
                Resize(self.hparams.input_size, lazy=True),
            ]
        )
        self.test_transforms = Compose(
            [
                EnsureChannelFirst(),
                NormalizeIntensity(),
                Orientation(axcodes='RAS', lazy=True),
                CropForeground(allow_smaller=False, lazy=True),
                Resize(self.hparams.input_size, lazy=True),
            ]
        )

        self.data_train: Optional[ImageDataset] = None
        self.data_val: Optional[ImageDataset] = None
        self.data_test: Optional[ImageDataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of LBL classes (10).
        """
        return 2

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
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = self.dataset_func(self.train_images, self.train_segs, self.train_labels,
                                                transform=self.train_transforms)
            self.data_val = self.dataset_func(self.val_images, self.val_segs, self.val_labels,
                                              transform=self.val_transforms)
            self.data_val = self.dataset_func(self.test_images, self.test_segs, self.test_labels,
                                              transform=self.test_transforms)

    def train_dataloader(self) -> DataLoader[Any]:
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

    def val_dataloader(self) -> DataLoader[Any]:
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

    def test_dataloader(self) -> DataLoader[Any]:
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
    _ = LBLDataModule()
