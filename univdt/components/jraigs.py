# Justified Referral in AI Glaucoma Screening
# https://justraigs.grand-challenge.org/justraigs

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import torch

from univdt.components.base import BaseComponent
from univdt.utils.image import load_image
from univdt.utils.retrieve import load_file


class JRAIGS(BaseComponent):
    """
    JRAIGS dataset for glaucoma screening

    Args:
        root : root folder for dataset
        split : 'train', 'val', 'test' and 'trainval'
        transform : Composed transforms
        fold : int or list of int for validation fold. Default is 1.
    """

    _AVAILABLE_SPLITS = ['train', 'val']
    _AVAILABLE_KEYS = ['difficulty']

    def __init__(self, root_dir: str, split: str, transform=None, fold: int | list[int] = 1,
                 additional_keys: list[str] | None = None):
        super().__init__(root_dir, split, transform)
        self._check_split(self._AVAILABLE_SPLITS)

        self.fold_val = fold if isinstance(fold, list) else [fold]
        self.fold_train = list(set(range(10)) - set(self.fold_val))

        # additional keys
        additional_keys = additional_keys if additional_keys is not None else []
        additional_keys = additional_keys if isinstance(additional_keys, list) else [additional_keys]
        self.additional_keys = additional_keys
        if self.additional_keys:
            assert all([key in self._AVAILABLE_KEYS for key in self.additional_keys]), \
                f'Invalid additional keys: {self.additional_keys}'

        self.num_classes = 1

        self.annots = self._load_paths()

    def __getitem__(self, index) -> Any:
        data = self._load_data(index)
        image: np.ndarray = data['image']
        label: int = 1 if data['label'] == 'rg' else 0
        label = np.array(label, dtype=np.int64)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']

        # convert image to pytorch tensor
        image = image.transpose(2, 0, 1)
        image = image.astype('float32') / 255.0
        image = torch.Tensor(image)
        output = {'image': image, 'label': label, 'path': data['path']}
        output.update({key: data[key] for key in self.additional_keys})
        return output

    def __len__(self) -> int:
        return len(self.annots)

    def _load_data(self, index: int) -> dict[str, Any]:
        data = self.annots[index]
        path: str = data['path']
        label: str = data['label'][0].lower()  # main label
        difficulty: int = int(data['difficult'])  # 1 or 0 for difficult or not
        image: np.ndarray = load_image(path, out_channels=3)  # 8bit 3 channel image
        return {'image': image, 'label': label, 'path': path, 'difficulty': difficulty}

    def _load_paths(self) -> list[dict[str, Any]]:
        dummy = load_file(Path(self.root_dir) / 'just_raigs_train_folded.json')
        dummy = [ann for ann in dummy if ann['fold'] in
                 (self.fold_train if self.split == 'train' else self.fold_val)]
        annots = [img for annot in dummy for img in annot['images']]
        for annot in annots:
            path = Path(self.root_dir) / 'images' / annot['path']
            for ext in ['JPG', 'JPEG', 'PNG']:
                new_path = path.with_suffix(f'.{ext}')
                if new_path.exists():
                    annot['path'] = str(new_path)
                    break
        return annots

    # def _load_paths(self) -> list[dict[str, Any]]:
    #     dummy = load_file(Path(self.root_dir) / 'just_raigs_train_folded.json')
    #     dummy = [ann for ann in dummy if ann['fold'] in
    #              (self.fold_train if self.split == 'train' else self.fold_val)]
    #     annots = [img for annot in dummy for img in annot['images']]

    #     if self.balance_classes:
    #         # normal과 abnormal의 샘플을 분리
    #         normal_samples = [annot for annot in annots if annot['label'].lower() == 'normal']
    #         abnormal_samples = [annot for annot in annots if annot['label'].lower() != 'normal']
    #         # normal과 abnormal 중 적은 수를 따라감
    #         min_samples = min(len(normal_samples), len(abnormal_samples))
    #         # 각 클래스에서 동일한 수의 샘플을 선택하여 리스트로 만듦
    #         balanced_samples = list(itertools.chain.from_iterable(itertools.zip_longest(normal_samples[:min_samples],
    #                                                                                     abnormal_samples[:min_samples])))
    #         # None 값 제거
    #         balanced_samples = [sample for sample in balanced_samples if sample is not None]
    #         annots = balanced_samples

    #     return annots
