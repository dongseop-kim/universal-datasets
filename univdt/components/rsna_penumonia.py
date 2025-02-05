from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from univdt.components.base import BaseComponent
from univdt.utils.image import load_image

class RSNAPneumonia(BaseComponent):
    """
    RSNA Pneumonia dataset, which is a collection of 4 datasets:
    - DADB or TBXPredict
    - Shenzhen 
    - Montgomery
    - TBX11K 

    Args:
        root : root folder for dataset
        path_annotation : path to annotation file
        split : 'train', 'val', 'test' and 'trainval'
        transform : Composed transforms
    """
    _AVAILABLE_SPLITS = ['train', 'val', 'test', 'trainval']

    def __init__(self, root_dir: str, split: str,
                 path_annotation:str, transform=None):
        super().__init__(root_dir, split, transform)
        self._check_split(self._AVAILABLE_SPLITS)