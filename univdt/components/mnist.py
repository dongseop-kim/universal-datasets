from typing import Any

import numpy as np
import torch
from torchvision.datasets import MNIST as TorchMNIST

from univdt.components.base import BaseComponent


class MNIST(TorchMNIST, BaseComponent):
    _AVAILABLE_SPLITS = ['train', 'test']

    def __init__(self, root_dir: str, split: str, transform=None):
        BaseComponent.__init__(self, root_dir, split, transform=transform)
        TorchMNIST.__init__(self, root_dir, train=(split == 'train'), transform=transform, download=True)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # _load_data is not needed for MNIST and already implemented in TorchMNIST
        image: torch.Tensor = self.data[index]  # 0 ~ 255
        image = torch.unsqueeze(image, 2)  # add channel dimension
        image: np.ndarray = np.array(image)

        label: torch.Tensor = self.targets[index]
        label: np.ndarray = np.array(label)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']

        image = self._to_tensor(image)  # convert image to pytorch tensor
        label = torch.from_numpy(label).long()  # convert label to pytorch tensor

        return {'image': image, 'label': label}
