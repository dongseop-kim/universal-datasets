from pathlib import Path
from typing import Union

import cv2
import numpy as np
from torch.utils.data import Dataset


class BaseComponent(Dataset):
    TASK = None

    def __init__(self, root_dir: str, split: str, transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        assert self.TASK is not None, "task must be defined"

    def _load_image(self, path_image: Union[str, Path], out_channels: int = 3) -> np.ndarray:
        """ load image from path 
        Args:
            path_image: path to image
            out_channels: number of channels of image (1 or 3)
        Returns:
            image (np.ndarray, uint8) : loaded image (H x W x 1 or 3) if channels=3, image is RGB
        """
        assert out_channels in [1, 3], "channels must be 1 or 3"
        image: np.ndarray = cv2.imread(str(path_image), cv2.IMREAD_UNCHANGED)
        image = np.expand_dims(image, axis=2) if image.shape == 2 else image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image
        if image.shape[2] != out_channels:
            conversion = cv2.COLOR_GRAY2RGB if image.shape[2] == 1 else cv2.COLOR_BGR2GRAY
            image = cv2.cvtColor(image, conversion)
        return image.astype(np.uint8)
