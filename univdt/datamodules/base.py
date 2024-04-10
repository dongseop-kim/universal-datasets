from functools import partial
from typing import Any, Optional

import albumentations as A
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader

from univdt.components import (MNIST, NIH, BaseComponent, CamVid, PascalVOC,
                               PublicTuberculosis)
from univdt.transforms import build_transforms

AVAILABLE_COMPONENTS = {'mnist': MNIST,
                        'nih': NIH,
                        'camvid': CamVid,
                        'pascalvoc': PascalVOC,
                        'publictuberculosis': PublicTuberculosis}

DEFAULT_HEIGHT = 256
DEFAULT_WIDTH = 256
DEFAULT_TRANSFORMS = A.Compose([A.Resize(height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH, p=1.0)])


class BaseDataModule(LightningDataModule):
    """
    Base dataloader for all datasets

    Args:
        data_dir : root directory for dataset
        datasets : dataset names to load. if multiple datasets are given, they will be concatenated
        batch_size : batch size for dataloader. if batch_size is given, all batch_size_* will be ignored
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
                 data_dir: str | list[str],
                 datasets: str | list[str],
                 batch_size: int,
                 num_workers: Optional[int] = 0,

                 split_train: Optional[str] = 'train',
                 split_val: Optional[str] = 'val',
                 split_test: Optional[str] = 'test',
                 additional_keys: Optional[list[str]] = [],

                 transforms_train: Optional[dict[str, Any]] = None,
                 transforms_val: Optional[dict[str, Any]] = None,
                 transforms_test: Optional[dict[str, Any]] = None,

                 batch_size_train: Optional[int] = None,
                 batch_size_val: Optional[int] = None,
                 batch_size_test: Optional[int] = None):
        super().__init__()
        self.data_dir = [data_dir] if isinstance(data_dir, str) else data_dir
        self.datasets = [datasets] if isinstance(datasets, str) else datasets

        # get splits and check
        self.split_train = split_train if split_train is not None else 'train'
        self.split_val = split_val if split_val is not None else 'val'
        self.split_test = split_test if split_test is not None else 'test'
        assert self.split_train in ['train', 'trainval'], f'Invalid split for training: {self.split_train}'
        assert self.split_val in ['val', 'test'], f'Invalid split for validation: {self.split_val}'
        assert self.split_test in ['val', 'test'], f'Invalid split for testing: {self.split_test}'
        self.additional_keys = additional_keys

        # set hyperparameters for dataloader
        self.batch_size_train = batch_size_train or batch_size
        self.batch_size_val = batch_size_val or batch_size
        self.batch_size_test = batch_size_test or batch_size
        self.num_workers = num_workers if num_workers is not None else 0
        self.persistent_workers = True if self.num_workers > 0 else False
        self.pin_memory = True  # if self.num_workers > 0 else False

        # get transforms
        self.transforms_train = build_transforms(transforms_train) \
            if transforms_train is not None else DEFAULT_TRANSFORMS
        self.transforms_val = build_transforms(transforms_val) \
            if transforms_val is not None else DEFAULT_TRANSFORMS
        self.transforms_test = build_transforms(transforms_test) \
            if transforms_test is not None else DEFAULT_TRANSFORMS

        self.dataset_train: BaseComponent = None
        self.dataset_val: BaseComponent = None
        self.dataset_test: BaseComponent = None

    def _load_datasets(self, split: str, transforms: dict[str, Any]):
        loaded_datasets = []
        for data_dir, dataset in zip(self.data_dir, self.datasets):
            loaded_datasets.append(AVAILABLE_COMPONENTS[dataset](data_dir, split, transforms))
        return ConcatDataset(loaded_datasets)

    def setup(self, stage: str = 'fit'):
        assert stage in ['fit', 'validate', 'test', 'predict'], \
            f"Invalid stage: {stage}. Must be in ['fit', 'validate', 'test', 'predict']"
        match stage:
            case 'fit':
                self.dataset_train = self._load_datasets(self.split_train, self.transforms_train)
                self.dataset_val = self._load_datasets(self.split_val, self.transforms_train)
            case 'validate':
                self.dataset_val = self._load_datasets(self.split_val, self.transforms_train)
            case 'test':
                self.dataset_test = self._load_datasets(self.split_test, self.transforms_test)
            case 'predict':
                pass

    def teardown(self, stage: str):
        """Called at the end of fit (train + validate), validate, test, or predict."""
        super().teardown(stage)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size_train, shuffle=True, num_workers=self.num_workers,
                          drop_last=True, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers,
                          collate_fn=None)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size_val, shuffle=False, num_workers=self.num_workers,
                          drop_last=False, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers,
                          collate_fn=None)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size_test, shuffle=False, num_workers=self.num_workers,
                          drop_last=False, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers,
                          collate_fn=None)

    def predict_dataloader(self):
        # TODO: Implement predict dataloader!
        pass
