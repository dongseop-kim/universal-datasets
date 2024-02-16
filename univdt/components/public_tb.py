from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from univdt.components.base import BaseComponent
from univdt.utils.image import load_image

# 'both' means both active and inactive tb
# 2~4 are only available in tbx11k dataset
LABEL_TO_TRAINID = {'normal': 0, 'activetb': 1,
                    'inactivetb': 2, 'both': 3, 'others': 4}


class PublicTuberculosis(BaseComponent):
    """
    Public Tuberculosis dataset, which is a collection of 4 datasets:
    - DADB 
    - Shenzhen 
    - Montgomery
    - TBX11K 

    Args:
        root : root folder for dataset
        split : 'train', 'val', 'test' and 'trainval'
        transform : Composed transforms
        dataset : dataset name to load
    """
    AVAILABLE_DATASETS = ['tbxpredict', 'shenzhen', 'montgomery', 'tbx11k']
    AVAILABLE_KEYS = ['age', 'gender', 'report']

    def __init__(self, root_dir: str, split: str,
                 transform=None, dataset: str = 'shenzhen',
                 additional_keys: Optional[list[str]] = None):
        super().__init__(root_dir, split, transform, additional_keys)
        self.check_split(['train', 'val', 'trainval', 'test'])
        assert dataset in self.AVAILABLE_DATASETS, \
            f'Invalid dataset: {dataset}, must be one of tbxpredict, shenzhen, montgomery, tbx11k'

        if self.additional_keys:
            assert all([key in self.AVAILABLE_KEYS for key in self.additional_keys]), \
                f'Invalid additional keys: {self.additional_keys}'

        self.dataset = dataset
        self.num_classes = 4  # 4 classes: normal, active, inactive, others
        self.void_class = -1  # class labeled as -1 is the 'void'

        self.raw_data = self._load_paths()

    def __getitem__(self, index) -> dict[str, Any]:
        data = self.load_data(index)
        image: np.ndarray = data['image']
        label: np.ndarray = data['label']
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']

        # convert image to pytorch tensor
        image = image.transpose(2, 0, 1)
        image = image.astype('float32') / 255.0
        image = torch.Tensor(image)
        output = {'image': image, 'label': label,
                  'dataset': self.dataset, 'path': data['path']}
        output.update({key: data[key] for key in self.additional_keys})
        return output

    def __len__(self) -> int:
        return len(self.raw_data)

    def load_data(self, index) -> dict[str, Any]:
        raw_data = self.raw_data[index]

        # load image
        image_path = Path(self.root_dir) / raw_data['name']
        assert image_path.exists(), f'Image {image_path} does not exist'
        image = load_image(image_path, out_channels=1)  # normalized to [0, 255]

        # load label
        label = LABEL_TO_TRAINID[raw_data['label']] if raw_data['label'] != -1 else self.void_class
        label = np.array(label, dtype=np.int64)

        # load etc data
        age = raw_data['age']
        gender = raw_data['gender']
        report = raw_data['report']

        # include dataset name for dataset concatenation
        return {'image': image, 'label': label, 'age': age, 'gender': gender,
                'report': report, 'dataset': self.dataset, 'path': str(image_path)}

    def _load_paths(self) -> list[dict[str, Any]]:
        import pandas as pd
        # name, split, label, age, gender, report
        df = pd.read_csv(Path(self.root_dir) / f'{self.dataset}.csv')
        df = df[df['split'].isin([self.split])] if self.split != 'trainval' \
            else df[df['split'].isin(['train', 'val'])]
        if self.split == 'test':
            # all df['label'] are -1 in test split
            df['label'] = -1
            df['report'] = ''

        return [dict(row) for _, row in df.iterrows()]
