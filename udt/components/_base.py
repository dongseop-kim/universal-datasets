from torch.utils.data import Dataset
import numpy as np
import cv2
from pathlib import Path
from typing import Union


def _check_image(image: np.ndarray):
    # check 3 channels
    if len(image.shape) != 3 and image.shape[2] != 3:
        raise ValueError("image must be 3 channels")
    # check dtype
    if image.dtype != np.uint8:
        raise ValueError("image must be uint8")


class BaseComponent(Dataset):
    TASK = None

    def __init__(self, root_dir: str, split: str, transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        # self.task = task

    def _load_image(self, path_image: Union[str, Path]) -> np.ndarray:
        """ load image from path 
        Args:
            path_image: path to image
        Returns:
            image (np.ndarray, uint8) : loaded image (H x W x 3), RGB
        """
        image: np.ndarray = cv2.imread(str(path_image), cv2.IMREAD_UNCHANGED)
        image = np.expand_dims(image, axis=2) if image.shape == 2 else image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.shape[2] == 1 else image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.uint8)
        return image
