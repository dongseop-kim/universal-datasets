from pathlib import Path
from typing import List

from torch.utils.data import Dataset


class BaseComponent(Dataset):
    TASK: List[str] = None

    def __init__(self, root_dir: str, split: str, transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        assert self.TASK is not None, "task must be defined"
