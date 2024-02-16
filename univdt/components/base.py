from abc import abstractmethod
from pathlib import Path
from typing import Any,  Optional

from torch.utils.data import Dataset


class BaseComponent(Dataset):
    def __init__(self, root_dir: str, split: str, transform=None,
                 additional_keys: Optional[list[str]] = None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.additional_keys = additional_keys if additional_keys else []
        self.collate_fn = None

    @abstractmethod
    def load_data(self, index) -> dict[str, Any]:
        pass

    def check_split(self, split: list[str]) -> bool:
        assert self.split in split, f'Invalid split: {self.split}, must be one of {split}'
