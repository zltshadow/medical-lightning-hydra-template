from pathlib import Path

import pytest
import torch

from src.data.lbl_datamodule import LBLDataModule


@pytest.mark.parametrize("batch_size", [2])
def test_LBL_datamodule(batch_size: int) -> None:
    """Tests `LBLDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    dataset_json = r"E:\projects\BIT\data\nnUNet_datasets\nnUNet_raw\Dataset803_LBL_raw_BJTR\dataset.json"

    dm = LBLDataModule(dataset_json=dataset_json, batch_size=batch_size)

    assert not dm.data_train and not dm.data_val and not dm.data_test

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 70_000

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
