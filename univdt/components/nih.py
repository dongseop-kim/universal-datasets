from pathlib import Path
from typing import Any

import numpy as np
import torch

from univdt.components.base import BaseComponent
from univdt.utils.image import load_image

MAPPER = {'normal': 0,
          'effusion': 1,
          'emphysema': 2,
          'atelectasis': 3,
          'edema': 4,
          'consolidation': 5,
          'pleural_thickening': 6,
          'hernia': 7,
          'mass': 8,
          'cardiomegaly': 9,
          'nodule': 10,
          'pneumothorax': 11,
          'pneumonia': 12,
          'fibrosis': 13,
          'infiltration': 14}


class NIH(BaseComponent):
    """
    NIH Chest X-ray 14 dataset    
    Args:
        root : root folder for dataset
        split : 'train', 'val', 'test' and 'trainval'
        transform : Composed transforms

    """

    _AVAILABLE_KEYS = ['age', 'gender', 'view_position', 'patient_id', 'follow_up']

    def __init__(self, root_dir: str, split: str, transform=None,
                 additional_keys: list[str] | None = None):
        super().__init__(root_dir, split, transform, additional_keys)
        self._check_split(['train', 'val', 'trainval', 'test'])

        if self.additional_keys:
            assert all([key in self._AVAILABLE_KEYS for key in self.additional_keys]), \
                f'Invalid additional keys: {self.additional_keys}'

        self.raw_data = self._load_paths()

    def __getitem__(self, index: int) -> dict[str, Any]:
        data = self._load_data(index)
        image: np.ndarray = data['image']
        label: np.ndarray = data['label']
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
        return len(self.raw_data)

    def _load_data(self, index: int) -> dict[str, Any]:
        raw_data = self.raw_data[index]
        # load image
        image_path = Path(self.root_dir) / raw_data['path']
        assert image_path.exists(), f'Image {image_path} does not exist'
        image = load_image(image_path, out_channels=1)  # normalized to [0, 255]

        label = raw_data['findings']
        label = np.array(label, dtype=np.int64)

        # load etc data
        age = raw_data['age']
        gender = raw_data['gender']
        view_position = raw_data['view-position']
        patient_id = raw_data['pid']
        follow_up = raw_data['follow-up']

        return {'image': image, 'label': label, 'path': str(image_path),
                'age': age, 'gender': gender, 'view_position': view_position,
                'pid': patient_id, 'fup': follow_up}

    def _load_paths(self):
        import pandas as pd
        # path, split, findings, age, gender, view-position, pid, follow-up
        df = pd.read_csv(self.root_dir / 'nih.csv')
        df = df[df['split'].isin([self.split])] if self.split != 'trainval' \
            else df[df['split'].isin(['train', 'val'])]
        for i, row in df.iterrows():
            # string list to list in findings
            findings = str(row['findings'])
            findings = findings.strip('[]').split(',')
            findings = [int(f.strip()) for f in findings]
            df.at[i, 'findings'] = findings
        # add 'images/' to path
        df['path'] = df['path'].apply(lambda x: 'images/' + x)
        return [dict(row) for _, row in df.iterrows()]
