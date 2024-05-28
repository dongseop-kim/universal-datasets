from pathlib import Path
from typing import Any

import numpy as np
import random

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
         root_dir (str): Root directory of the dataset.
         split (str): One of {'train', 'val', 'test', 'trainval'} for dataset split.
         transform (callable, optional): A function/transform to apply to the image.
         additional_keys (list[str], optional): List of additional keys to be included in the output.
        target_findings (list[str], optional): List of target findings to be included in the output.
        normal_ratio (float, optional): Ratio of normal to abnormal samples. Default is 0.0.
    """
    _AVAILABLE_SPLITS = ['train', 'val', 'test', 'trainval']
    _AVAILABLE_KEYS = ['age', 'gender', 'view_position', 'patient_id', 'follow_up']

    def __init__(self, root_dir: str, split: str, transform=None,
                 additional_keys: list[str] | None = None,
                 target_findings: list[str] | None = None,
                 normal_ratio: float = 0.0):
        super().__init__(root_dir, split, transform)

        # set additional keys and check validity
        self.additional_keys: list[str] = additional_keys if additional_keys is not None else []
        assert self._check_additional_keys(self.additional_keys), f"Invalid additional keys"

        # set target findings and check validity, exclude 'normal' from target findings
        self.target_findings: list[str] = target_findings if target_findings is not None else list(MAPPER.keys())[1:]
        assert self._check_target_findings(self.target_findings), "Invalid target findings."
        self.target_finding_ids: list[int] = [MAPPER[f] for f in self.target_findings]

        self.num_classes = len(self.target_findings)

        self.normal_ratio = normal_ratio
        self.annots, self.annots_abnormal, self.annots_normal = self._load_annoations()

    def __getitem__(self, index: int) -> dict[str, Any]:

        if self.normal_ratio > 0.0 and index >= len(self.annots_abnormal):
            index = len(self.annots_abnormal) + random.randint(0, len(self.annots_normal) - 1)

        data = self._load_data(index)  # load image, label, path, and additional keys
        image: np.ndarray = data['image']
        label: list[int] = data['label']
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']

        image = self._to_tensor(image)  # convert image to pytorch tensor
        label = self._to_onehot(label, len(MAPPER))  # [1:]  # one-hot encoding except for normal
        label = label[self.target_finding_ids]
        output = {'image': image, 'label': label, 'path': data['path']}
        output.update({key: data[key] for key in self.additional_keys})
        return output

    def __len__(self) -> int:
        if self.normal_ratio > 0.0:
            # balanced dataset with abnormal to normal ratio
            return len(self.annots_abnormal) + int(self.normal_ratio * len(self.annots_abnormal))
        else:
            return len(self.annots)

    def _load_data(self, index: int) -> dict[str, Any]:
        if self.normal_ratio > 0.0:
            if index < len(self.annots_abnormal):
                annot = self.annots_abnormal[index]
            else:
                annot = random.choice(self.annots_normal)
        else:
            annot = self.annots[index]

        # load image
        image_path = Path(self.root_dir) / annot['path']
        assert image_path.exists(), f'Image {image_path} does not exist'
        image = load_image(image_path, out_channels=1)  # normalized to [0, 255]

        label: list[int] = annot['findings']

        # load etc data
        age: int = int(annot['age'])
        gender: str = str(annot['gender']).lower()
        view_position: str = str(annot['view-position']).lower()
        patient_id: int = int(annot['pid'])
        follow_up: int = int(annot['follow-up'])
        return {'image': image, 'label': label, 'path': str(image_path),
                'age': age, 'gender': gender, 'view_position': view_position,
                'pid': patient_id, 'fup': follow_up}

    def _load_annoations(self):
        import pandas as pd
        # path, split, findings, age, gender, view-position, pid, follow-up
        df = pd.read_csv(str(Path(self.root_dir) / 'nih.csv'))
        df = df[df['split'].isin([self.split])] if self.split != 'trainval' \
            else df[df['split'].isin(['train', 'val'])]
        df['path'] = df['path'].apply(lambda x: 'images/' + x)

        annots = [dict(row) for _, row in df.iterrows()]
        for ann in annots:
            ann['findings'] = eval(ann['findings'])  # list[int]

        # filter with target findings for balanced dataset

        def is_abnormal(findings): return any(f in self.target_finding_ids for f in findings)
        annots_abnormal = [ann for ann in annots if is_abnormal(ann['findings'])]
        annots_normal = [ann for ann in annots if not is_abnormal(ann['findings'])]
        return annots_abnormal + annots_normal, annots_abnormal, annots_normal

    def _check_target_findings(self, findings: list[str]) -> bool:
        return all([finding in MAPPER.keys() for finding in findings])

    def _check_additional_keys(self, keys: list[str]) -> bool:
        return all([key in self._AVAILABLE_KEYS for key in keys])
