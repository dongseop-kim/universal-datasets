from typing import Any

import albumentations as A
from omegaconf import DictConfig, ListConfig

from .pixel import AVAILABLE_TRANSFORMS as pixel_transforms
from .pixel import RandAugmentPixel
from .ratio import RandomRatio
from .resize import Letterbox, RandomResize
from .windowing import RandomWindowing
from .zoom import RandomZoom

AVAILABLE_TRANSFORMS = {'resize': A.Resize,
                        'letterbox': Letterbox,
                        'random_flip': A.HorizontalFlip,
                        'random_ratio': RandomRatio,
                        'random_resize': RandomResize,
                        'random_windowing': RandomWindowing,
                        'random_zoom': RandomZoom,
                        'random_pixel_aug': RandAugmentPixel}

AVAILABLE_TRANSFORMS.update(pixel_transforms)


def build_transforms(transforms: dict[str, Any]):
    def recursive_check(config):
        if isinstance(config, DictConfig):
            return {key: recursive_check(val) for key, val in config.items()}
        elif isinstance(config, ListConfig):
            return tuple(map(recursive_check, config))
        else:
            return config
    return A.Compose([AVAILABLE_TRANSFORMS[str(key)](**recursive_check(val) if val else {})
                      for key, val in transforms.items()])
