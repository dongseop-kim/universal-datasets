from typing import Any

import numpy as np
from torchvision.datasets import MNIST as TorchMNIST

from univdt.components.base import BaseComponent


class MNIST(TorchMNIST, BaseComponent):
    _AVAILABLE_SPLITS = ['train', 'test']

    def __init__(self, root_dir: str, split: str, transform=None):
        BaseComponent.__init__(self, root_dir, split, transform=transform)
        TorchMNIST.__init__(self, root_dir, train=(split == 'train'), transform=transform, download=True)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image, label = super().__getitem__(index)
        image: np.ndarray = np.array(image)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']
        return {'image': image, 'label': label}
