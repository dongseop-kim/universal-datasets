from typing import Any
import albumentations as A
from omegaconf import DictConfig, ListConfig

from .pixel import (random_blur, random_brightness, random_clahe,
                    random_compression, random_contrast, random_gamma,
                    random_hist_equal, random_noise, random_windowing)
from .resize import random_resize
from .zoom import random_zoom

AVAILABLE_TRANSFORMS = {'resize': A.Resize,
                        'random_blur': random_blur,
                        'random_brightness': random_brightness,
                        'random_clahe': random_clahe,
                        'random_compression': random_compression,
                        'random_contrast': random_contrast,
                        'random_flip': A.HorizontalFlip,
                        'random_gamma': random_gamma,
                        'random_histequal': random_hist_equal,
                        'random_noise': random_noise,
                        'random_resize': random_resize,
                        'random_windowing': random_windowing,
                        'random_zoom': random_zoom, }


def build_transforms(transforms: dict[str, Any]):

    def recursive_check(config):
        if isinstance(config, DictConfig):
            return {key: recursive_check(val) for key, val in config.items()}
        elif isinstance(config, ListConfig):
            return tuple(map(recursive_check, config))
        else:
            return config

    return [AVAILABLE_TRANSFORMS[str(key)](**recursive_check(val) if val else {})
            for key, val in transforms.items()]
