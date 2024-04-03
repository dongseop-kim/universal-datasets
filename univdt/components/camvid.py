from collections import namedtuple
from pathlib import Path
from typing import Any, List, Tuple

import cv2
import numpy as np

from univdt.components.base import BaseComponent
from univdt.utils.image import load_image

CAMVID_CLASS = namedtuple('camvid_class', ['name', 'id', 'train_id',  'color'])
CAMVID_CLASSES = [CAMVID_CLASS('sky', 0, 0,  (128, 128, 128)),
                  CAMVID_CLASS('building', 1, 1,  (128, 0, 0)),
                  CAMVID_CLASS('column-pole', 2, 2,  (192, 192, 128)),
                  CAMVID_CLASS('road', 3, 3,  (128, 64, 128)),
                  CAMVID_CLASS('sidewalk', 4, 4,  (0, 0, 192)),
                  CAMVID_CLASS('tree', 5, 5,  (128, 128, 0)),
                  CAMVID_CLASS('sign-symbol', 6, 6,  (192, 128, 128)),
                  CAMVID_CLASS('fence', 7, 7,  (64, 64, 128)),
                  CAMVID_CLASS('car', 8, 8,  (64, 0, 128)),
                  CAMVID_CLASS('pedestrian', 9, 9,  (64, 64, 0)),
                  CAMVID_CLASS('bicyclist', 10, 10,  (0, 128, 192)),
                  CAMVID_CLASS('void', 11, 255,  (0, 0, 0))]


class CamVid(BaseComponent):
    """
    CamVid dataset for semantic segmentation

    Args:
        root : root folder for dataset
        split : 'train', 'val', 'test' and 'trainval'
        transform : Composed transforms
    """
    COLORMAP = np.array([c.color for c in CAMVID_CLASSES], dtype=np.uint8)
    _AVAILABLE_SPLITS = ['train', 'val', 'trainval', 'test']

    def __init__(self, root_dir: str, split: str, transform=None):
        super().__init__(root_dir, split, transform)

        self.num_classes = 11
        self.void_class = 255  # class labeled as 11 is the 'void'

        self.paths: list[tuple[str, str]] = self._load_paths()

    def __getitem__(self, index: int) -> dict[str, Any]:
        data: dict[str, Any] = self._load_data(index)
        image: np.ndarray = data['image']
        mask: np.ndarray = self._mask_to_array(data['mask'])
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        return {'image': image, 'mask': mask, 'path': data['path']}

    def __len__(self) -> int:
        return len(self.paths)

    def _mask_to_array(self, mask: np.ndarray) -> np.ndarray:
        mask[mask == 11] = self.void_class  # set void class to 255
        return mask

    def _load_data(self, index: int) -> dict[str, Any]:
        path_image, path_mask = self.paths[index]
        image = load_image(path_image, out_channels=3)
        mask: np.ndarray = cv2.imread(path_mask, cv2.IMREAD_UNCHANGED).astype(np.uint8)
        return {'image': image, 'mask': mask, 'path': path_image}

    def _load_paths(self) -> list[tuple[str, str]]:

        def load_paths_by_split(split: str) -> Tuple[List[Path], List[Path]]:
            with open(Path(self.root_dir) / f'{split}.txt') as f:
                data = [line.strip().split(" ") for line in f]
            paths_image = [Path(self.root_dir) / line[0] for line in data]
            paths_masks = [Path(self.root_dir) / line[1] for line in data]
            return paths_image, paths_masks

        if self.split == 'trainval':
            paths_image, paths_masks = [], []
            for split in ['train', 'val']:
                split_image, split_masks = load_paths_by_split(split)
                paths_image += split_image
                paths_masks += split_masks
        else:
            paths_image, paths_masks = load_paths_by_split(self.split)

        # check
        assert len(paths_image) == len(paths_masks)
        assert any([p.exists() for p in paths_image])
        assert any([p.exists() for p in paths_masks])
        assert any(p1.stem == p2.stem for p1, p2 in zip(paths_image, paths_masks))

        # sort and convert paths to strings
        paths_image = sorted(map(str, paths_image))
        paths_masks = sorted(map(str, paths_masks))

        return list(zip(paths_image, paths_masks))
