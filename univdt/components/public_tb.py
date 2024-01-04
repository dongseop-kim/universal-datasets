from pathlib import Path
from typing import Any, Dict, List

from univdt.components.base import BaseComponent
from univdt.utils.image import load_image

# 'both' means both active and inactive tb
# 2~4 are only available in tbx11k dataset
_LABEL_TO_TRAINID = {'normal': 0, 'activetb': 1,
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
    TASK = ['classification']

    def __init__(self,
                 root_dir: str,
                 split: str,
                 transform=None,
                 dataset: str = 'shenzhen'):
        assert split in ['train', 'val', 'trainval', 'test']
        assert dataset in ['tbxpredict', 'shenzhen', 'montgomery', 'tbx11k']
        self.dataset = dataset
        # if self.dataset == 'tbx11k':
        #     self.TASK.append('detection')
        super().__init__(root_dir, split, transform)

        self.num_classes = 4  # 4 classes: normal, active, inactive, others
        self.void_class = -1  # class labeled as -1 is the 'void'

        self.raw_data = self._load_paths()

    def __getitem__(self, index):
        pass

    def __len__(self) -> int:
        return len(self.raw_data)

    def load_data(self, index) -> Dict[str, Any]:
        raw_data = self.raw_data[index]

        # load image
        image_path = Path(self.root_dir) / raw_data['name']
        assert image_path.exists(), f'Image {image_path} does not exist'
        image = load_image(image_path, out_channels=1)  # normalized to [0, 255]

        # load label
        label = _LABEL_TO_TRAINID[raw_data['label']]

        # load etc data
        age = raw_data['age']
        gender = raw_data['gender']
        report = raw_data['report']

        # include dataset name for dataset concatenation
        return {'image': image, 'label': label, 'age': age, 'gender': gender,
                'report': report, 'dataset': self.dataset}

    def _load_paths(self) -> List[Dict[str, Any]]:
        import pandas as pd

        # name, split, label, age, gender, report
        df = pd.read_csv(Path(self.root_dir) / f'{self.dataset}.csv')
        df = df[df['split'] == self.split] if self.split != 'trainval' \
            else df[df['split'] == 'train'] + df[df['split'] == 'val']

        output = [dict(row) for _, row in df.iterrows()]
        return output
