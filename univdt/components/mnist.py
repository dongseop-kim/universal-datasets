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
        image, label = self.data[index], int(self.targets[index])
        image: np.ndarray = image.numpy()

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']
        # convert image to pytorch tensor
        image = image.unsqueeze(2)
        image = image.transpose(2, 0, 1)
        image = image.astype('float32') / 255.0
        image = torch.Tensor(image)
        label = torch.LongTensor([label])
        return {'image': image, 'label': label}
