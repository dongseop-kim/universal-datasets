from functools import partial
from typing import Any, Literal, Optional

import albumentations as A
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader

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
                 data_dir: str | list[str],
                 datasets: str | list[str],
                 dataset_kwargs: dict[str, Any] = {},

                 split_train: Optional[str] = 'train',
                 split_val: Optional[str] = 'val',
                 split_test: Optional[str] = 'test',
                 additional_keys: Optional[list[str]] = [],

                 transforms_train: Optional[dict[str, Any]] = None,
                 transforms_val: Optional[dict[str, Any]] = None,
                 transforms_test: Optional[dict[str, Any]] = None,

                 batch_size_train: Optional[int] = None,
                 batch_size_val: Optional[int] = None,
                 batch_size_test: Optional[int] = None,
                 num_workers: Optional[int] = 0,
                 init_set: Literal['fit', 'validate', 'test', 'predict'] = 'fit'):

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
        self.dataset_kwargs = dataset_kwargs
        self.additional_keys = additional_keys

        # set hyperparameters for dataloader
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test
        self.num_workers = num_workers if num_workers is not None else 0
        self.persistent_workers = True if self.num_workers > 0 else False
        self.pin_memory = True

        # get transforms
        self.transforms_train = build_transforms(transforms_train) \
            if transforms_train is not None else None
        self.transforms_val = build_transforms(transforms_val) \
            if transforms_val is not None else None
        self.transforms_test = build_transforms(transforms_test) \
            if transforms_test is not None else None

        self.dataset_train: BaseComponent = None
        self.dataset_val: BaseComponent = None
        self.dataset_test: BaseComponent = None

        self.common_loader_settings = {'num_workers': self.num_workers,
                                       'pin_memory': self.pin_memory,
                                       'persistent_workers': self.persistent_workers}
        self.setup(init_set)
        self.num_classes = self.dataset_train.num_classes

    def _load_datasets(self, split: str, transform: dict[str, Any], dataset_kwargs):
        loaded_datasets = []
        for data_dir, dataset in zip(self.data_dir, self.datasets):
            loaded_datasets.append(AVAILABLE_COMPONENTS[dataset](root_dir=data_dir,
                                                                 split=split,
                                                                 transform=transform, **dataset_kwargs))
        return ConcatDataset(loaded_datasets) if len(loaded_datasets) > 1 else loaded_datasets[0]

    def setup(self, stage: str = 'fit'):
        assert stage in ['fit', 'validate', 'test', 'predict'], \
            f"Invalid stage: {stage}. Must be in ['fit', 'validate', 'test', 'predict']"

        match stage:
            case 'fit':
                self.dataset_train = self._load_datasets(self.split_train, self.transforms_train,
                                                         self.dataset_kwargs)
                self.dataset_val = self._load_datasets(self.split_val, self.transforms_val,
                                                       self.dataset_kwargs)
            case 'validate':
                self.dataset_val = self._load_datasets(self.split_val, self.transforms_val,
                                                       self.dataset_kwargs)
            case 'test':
                self.dataset_test = self._load_datasets(self.split_test, self.transforms_test,
                                                        self.dataset_kwargs)
            case 'predict':
                pass

    def teardown(self, stage: str):
        """Called at the end of fit (train + validate), validate, test, or predict."""
        super().teardown(stage)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size_train, shuffle=True, drop_last=True,
                          collate_fn=None,
                          **self.common_loader_settings)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size_val, shuffle=False, drop_last=False,
                          collate_fn=None,
                          **self.common_loader_settings)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size_test, shuffle=False, drop_last=False,
                          collate_fn=None,
                          **self.common_loader_settings)

    def predict_dataloader(self):
        # TODO: Implement predict dataloader!
        pass
