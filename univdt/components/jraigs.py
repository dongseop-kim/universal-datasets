import random
from pathlib import Path
from typing import Any
from collections.abc import Sequence
import numpy as np
import torch

from univdt.components.base import BaseComponent
from univdt.utils.image import load_image
from univdt.utils.retrieve import load_file


class JRAIGS(BaseComponent):
    """
    Justified Referral in AI Glaucoma Screening dataset for glaucoma screening
    https://justraigs.grand-challenge.org/justraigs

    Args:
        root : root folder for dataset
        split : 'train', 'val', 'test' and 'trainval'
        transform : Composed transforms
        fold : int or list of int for validation fold. Default is 1.
        normal_ratio : float for balancing normal and abnormal samples. Default is -1.0.
        additional_keys : list of additional keys to be included in the output. Default is None.
    """

    _AVAILABLE_SPLITS = ['train', 'val']
    _AVAILABLE_KEYS = ['difficulty']

    def __init__(self, root_dir: str, split: str, transform=None, fold: int | list[int] = 1,
                 normal_ratio: float = -1.0, additional_keys: list[str] | None = None):
        super().__init__(root_dir=root_dir, split=split, transform=transform)
        self._check_split(self._AVAILABLE_SPLITS)

        self.fold = fold if isinstance(fold, Sequence) else [fold]
        if self.split == 'train':
            self.fold = list(set(range(10)) - set(self.fold))

        # additional keys
        additional_keys = additional_keys if additional_keys is not None else []
        additional_keys = additional_keys if isinstance(additional_keys, list) else [additional_keys]
        self.additional_keys = additional_keys
        if self.additional_keys:
            assert all([key in self._AVAILABLE_KEYS for key in self.additional_keys]), \
                f'Invalid additional keys: {self.additional_keys}'

        self.num_classes = 1

        self.annots = self._load_paths()
        self.normal_ratio = normal_ratio

        self.len_total = None
        if normal_ratio > 0 and self.split == 'train':
            self._balancing_annots()

    def __getitem__(self, index) -> Any:
        data = self._load_data(index)
        image: np.ndarray = data['image']
        label: int = 1 if data['label'] == 'rg' else 0
        label = np.array(label, dtype=np.int64)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']

        image = self._to_tensor(image)  # convert image to pytorch tensor
        output = {'image': image, 'label': label, 'path': data['path']}
        output.update({key: data[key] for key in self.additional_keys})
        return output

    def __len__(self) -> int:
        return len(self.annots) if self.len_total is None else self.len_total

    def _load_data(self, index: int) -> dict[str, Any]:
        data = self.annots[index]

        if self.len_total is not None:
            if index < self.len_abnormal:
                data = self.annots_abnormal[index]
            else:
                # index = index - self.len_abnormal
                # normal 전체 길이에서 index를 유니폼하게 랜덤하게 뽑아서 사용
                index = random.randint(0, self.len_normal - 1)
                data = self.annots_normal[index]

        path: str = data['path']
        label: str = data['label']
        difficulty: int = int(data['difficult'])  # 1 or 0 for difficult or not
        image: np.ndarray = load_image(path, out_channels=3)  # 8bit 3 channel image
        return {'image': image, 'label': label, 'path': path, 'difficulty': difficulty}

    def _load_paths(self) -> list[dict[str, Any]]:
        dummy = load_file(Path(self.root_dir) / 'just_raigs_train_folded.json')
        dummy = [d for d in dummy if d['fold'] in self.fold]
        annots = [img for d in dummy for img in d['images']]  # flatten
        for annot in annots:
            annot['label'] = annot['label'][0].lower()
            path = Path(self.root_dir) / 'images' / annot['path']
            annot['path'] = str(path.with_suffix('.jpg'))
        return annots

    def _balancing_annots(self):
        cnt_abnormal = sum([1 for annot in self.annots if annot['label'] == 'rg'])
        cnt_normal = sum([1 for annot in self.annots if annot['label'] == 'nrg'])
        assert len(self.annots) == cnt_abnormal + cnt_normal, 'Invalid annotations'
        self.annots_abnormal = [annot for annot in self.annots if annot['label'] == 'rg']
        self.annots_normal = [annot for annot in self.annots if annot['label'] == 'nrg']

        self.len_abnormal = len(self.annots_abnormal)
        self.len_normal = int(self.len_abnormal * self.normal_ratio)
        self.len_total = self.len_abnormal + self.len_normal
