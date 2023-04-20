from enum import Enum
from typing import Any, Dict, Optional, Union, List, Callable
from utils.misc import load_cxr_image


class CollateFunc(Enum):
    """Enum for collate functions."""
    ALL = "all"
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
