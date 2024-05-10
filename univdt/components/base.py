from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Data:
    image: Any
    raw_data: Any
    path: str


class BaseComponent(Dataset):

    _AVAILABLE_SPLITS: list[str] = None

    def __init__(self, root_dir: str, split: str, transform=None):
        self.root_dir = root_dir
        self.split = split
        self._check_split(self._AVAILABLE_SPLITS)
        self.transform = transform
        self.collate_fn = None

    def load_data(self, index) -> dict[str, Any]:
        """
        Load raw image, raw data and path from the given index.

        Args:
            index: Index to load data from

        Returns:
            dict: Dictionary containing image, raw data and path
        """
        return self._load_data(index)

    @abstractmethod
    def _load_data(self, index) -> dict[str, Any]:
        pass

    def _check_split(self, split: list[str]) -> bool:
        assert self.split in split, f'Invalid split: {self.split}, must be one of {split}'

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert image to tensor. 
        """
        assert image.ndim == 3, f'Image must have 3 dimensions, got {image.ndim}'
        image = image.transpose(2, 0, 1)
        image = image.astype('float32') / 255.0
        return torch.Tensor(image)
