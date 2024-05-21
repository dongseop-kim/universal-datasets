from functools import partial
from typing import Any, Literal, Optional

import albumentations as A
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from univdt.components import (JRAIGS, MNIST, NIH, BaseComponent, CamVid,
                               PascalVOC, PublicTuberculosis)
from univdt.transforms import build_transforms

AVAILABLE_COMPONENTS = {'camvid': CamVid, 'mnist': MNIST, 'nih': NIH,
                        'jraigs': JRAIGS, 'pascalvoc': PascalVOC,
                        'publictuberculosis': PublicTuberculosis}


class BaseDataModule(LightningDataModule):
    """
    Base dataloader for all datasets

    Args:
        data_dir : root directory for dataset
        datasets : dataset names to load. if multiple datasets are given, they will be concatenated
        num_workers : number of workers for dataloader
        additional_keys (optional) : additional keys to load dataset
        split_train (optional) : split name for training. default is 'train'. either 'train' or 'trainval'
        split_val (optional) : split name for validation. default is 'val'. either 'val' or 'test'
        split_test (optional) : split name for testing. default is 'test'. either 'val' or 'test'
        transforms_train (optional) : transforms for training dataset
        transforms_val (optional) : transforms for validation dataset
        transforms_test (optional) : transforms for testing dataset
        batch_size_train (optional) : batch size for training dataset
        batch_size_val (optional) : batch size for validation dataset
        batch_size_test (optional) : batch size for testing dataset
    """

    def __init__(self,
                 config_dataset: dict[str, Any],
                 split_train: Optional[str] = 'train',
                 split_val: Optional[str] = 'val',
                 split_test: Optional[str] = 'test',
                 transforms_train: Optional[dict[str, Any]] = None,
                 transforms_val: Optional[dict[str, Any]] = None,
                 transforms_test: Optional[dict[str, Any]] = None,
                 batch_size_train: Optional[int] = None,
                 batch_size_val: Optional[int] = None,
                 batch_size_test: Optional[int] = None,
                 num_workers: Optional[int] = 0):
        super().__init__()
        self.config_dataset = config_dataset

        self.split_train = split_train
        self.split_val = split_val
        self.split_test = split_test

        self.transforms_train = build_transforms(transforms_train)
        self.transforms_val = build_transforms(transforms_val)
        self.transforms_test = build_transforms(transforms_test)

        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test

        self.num_workers = num_workers

        self._prepare_datasets()

    def _prepare_datasets(self):
        dataset_name = self.config_dataset.pop('name')
        dataset = partial(AVAILABLE_COMPONENTS[dataset_name], **self.config_dataset)
        self.dataset_train = dataset(split=self.split_train, transform=self.transforms_train)
        self.dataset_val = dataset(split=self.split_val, transform=self.transforms_val)
        self.dataset_test = dataset(split=self.split_test, transform=self.transforms_test)

    def _build_dataloader(self, dataset, batch_size):
        return DataLoader(dataset, batch_size=batch_size, shuffle=True if dataset is self.dataset_train else False,
                          drop_last=False, collate_fn=None, num_workers=self.num_workers, pin_memory=True)

    def train_dataloader(self):
        return self._build_dataloader(self.dataset_train, self.batch_size_train)

    def val_dataloader(self):
        return self._build_dataloader(self.dataset_val, self.batch_size_val)

    def test_dataloader(self):
        return self._build_dataloader(self.dataset_test, self.batch_size_test)
