from collections import namedtuple
from pathlib import Path
from typing import List, Tuple, Union

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

    def __init__(self,
                 root_dir: str,
                 split: str,
                 transform=None):
        super().__init__(root_dir, split, transform)
        self.check_split(['train', 'val', 'trainval', 'test'])

        self.num_classes = 11
        self.void_class = 11  # class labeled as 11 is the 'void'
        self.paths_image, self.paths_masks = self._load_paths()

    def __getitem__(self, index):
        # TODO: implement this
        pass

    def __len__(self) -> int:
        return len(self.paths_image)

    def _load_mask(self, path_mask: Union[Path, str]) -> np.ndarray:
        mask: np.ndarray = cv2.imread(str(path_mask), cv2.IMREAD_UNCHANGED)
        mask = mask.astype(np.uint8)
        mask[mask == self.void_class] = 255  # set void class to 255
        return mask

    def draw_mask(self, mask: np.ndarray) -> np.ndarray:
        mask[mask == 255] = self.void_class
        return self.COLORMAP[mask]

    def overlay_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        overlay = self.draw_mask(mask)
        return cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    def load_data(self, index) -> dict[str, np.ndarray]:
        path_image = self.paths_image[index]
        path_mask = self.paths_masks[index]

        image = load_image(path_image, out_channels=3)
        mask = self._load_mask(path_mask)
        return {'image': image, 'mask': mask, 'path': str(path_image)}

    def _load_paths_by_split(self, split: str) -> Tuple[List[Path], List[Path]]:
        with open(self.root_dir / f'{split}.txt') as f:
            data = [line.strip().split(" ") for line in f]
        paths_image = [self.root_dir / line[0] for line in data]
        paths_masks = [self.root_dir / line[1] for line in data]
        return paths_image, paths_masks

    def _load_paths(self) -> Tuple[List[Path], List[Path]]:
        if self.split in ['train', 'val', 'test']:
            paths_image, paths_masks = self._load_paths_by_split(self.split)
        elif self.split == 'trainval':
            paths_image, paths_masks = self._load_paths_by_split('train')
            paths_image_val, paths_masks_val = self._load_paths_by_split('val')
            paths_image.extend(paths_image_val)
            paths_masks.extend(paths_masks_val)
        # sort
        paths_image = sorted(paths_image)
        paths_masks = sorted(paths_masks)
        # check
        assert len(paths_image) == len(paths_masks)
        assert any([p.exists() for p in paths_image])
        assert any([p.exists() for p in paths_masks])
        assert any([p1.stem == p2.stem for p1, p2 in zip(paths_image, paths_masks)])
        return paths_image, paths_masks
