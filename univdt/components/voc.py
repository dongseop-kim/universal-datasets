
from collections import namedtuple
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from univdt.components.base import BaseComponent
from univdt.utils.image import load_image

VOC_CLASS = namedtuple('camvid_class', ['name', 'id', 'train_id',  'color'])
VOC_CLASSES = [VOC_CLASS('background', 0, 255,  (0, 0, 0)),
               VOC_CLASS('aeroplane', 1, 0,  (128, 0, 0)),
               VOC_CLASS('bicycle', 2, 1,  (0, 128, 0)),
               VOC_CLASS('bird', 3, 2,  (128, 128, 0)),
               VOC_CLASS('boat', 4, 3,  (0, 0, 128)),
               VOC_CLASS('bottle', 5, 4,  (128, 0, 128)),
               VOC_CLASS('bus', 6, 5,  (0, 128, 128)),
               VOC_CLASS('car', 7, 6,  (128, 128, 128)),
               VOC_CLASS('cat', 8, 7,  (64, 0, 0)),
               VOC_CLASS('chair', 9, 8,  (192, 0, 0)),
               VOC_CLASS('cow', 10, 9,  (64, 128, 0)),
               VOC_CLASS('diningtable', 11, 10,  (192, 128, 0)),
               VOC_CLASS('dog', 12, 11,  (64, 0, 128)),
               VOC_CLASS('horse', 13, 12,  (192, 0, 128)),
               VOC_CLASS('motorbike', 14, 13,  (64, 128, 128)),
               VOC_CLASS('person', 15, 14,  (192, 128, 128)),
               VOC_CLASS('pottedplant', 16, 15,  (0, 64, 0)),
               VOC_CLASS('sheep', 17, 16,  (128, 64, 0)),
               VOC_CLASS('sofa', 18, 17,  (0, 192, 0)),
               VOC_CLASS('train', 19, 18,  (128, 192, 0)),
               VOC_CLASS('tvmonitor', 20, 19,  (0, 64, 128))]


class PascalVOC(BaseComponent):
    """
    Pascal VOC dataset for visual object detection, instance segmentation, and semantic segmentation.
    Currently, only segmentation is supported.

    Args:
        root : root folder for dataset
        split : 'train', 'val', 'test' and 'trainval'
        transform : Composed transforms
    """
    COLORMAP = np.array([c.color for c in VOC_CLASSES], dtype=np.uint8)
    _AVAILABLE_SPLITS = ['train', 'val', 'trainval', 'test']

    def __init__(self, root_dir: str, split: str, transform=None):
        super().__init__(root_dir, split, transform)

        self.num_classes = 20  # excluding background
        self.void_class = 255

        self.paths: list[tuple[str, str]] = self._load_paths()

    def __getitem__(self, index: int) -> dict[str, Any]:
        data: dict[str, Any] = self._load_data(index)
        image: np.ndarray = data['image']
        mask: Image.Image = data['mask']
        mask: np.ndarray = self._mask_to_array(mask)  # convert to numpy array mapped to train_id

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        image = self._to_tensor(image)  # convert image to pytorch tensor
        mask = torch.from_numpy(mask).long()  # convert mask to pytorch tensor
        return {'image': image, 'mask': mask, 'path': data['path']}

    def __len__(self) -> int:
        return len(self.paths)

    def _load_paths(self) -> list[tuple[str, str]]:
        # NOTE : currently only for semantic segmentation.
        # read file names
        root_dir = Path(self.root_dir) / 'VOC2012'
        with open(str(root_dir / 'ImageSets' / 'Segmentation' / f'{self.split}.txt'), 'r') as f:
            file_names = [x.strip() for x in f.readlines()]

        def get_paths(image_dir: Path, fnames: list[str], suffix: str) -> list[str]:
            paths = [(image_dir / f'{fn}').with_suffix(suffix) for fn in fnames]
            return [str(p) for p in paths]

        paths_image = get_paths(root_dir / 'JPEGImages', file_names, '.jpg')
        paths_mask = get_paths(root_dir / 'SegmentationClass', file_names, '.png')

        return list(zip(paths_image, paths_mask))

    def _mask_to_array(self, mask: Image.Image) -> np.ndarray:
        """
        Convert the mask to numpy array and map the class ids to train ids.
        """
        mask = np.array(mask).astype(np.uint8)
        mask[mask == self.void_class] = 0
        mask -= 1
        return mask

    def _load_data(self, index: int) -> dict[str, Any]:
        # Get the paths of the image and mask.
        path_image, path_mask = self.paths[index]

        # Load the image and mask.
        image: np.ndarray = load_image(path_image, out_channels=3)
        if self.split == 'test':
            mask = Image.new('L', (image.shape[1], image.shape[0]))  # 'L' mode for grayscale
        else:
            mask = Image.open(path_mask)

        return {'image': image, 'mask': mask, 'path': path_image}
