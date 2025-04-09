from typing import Any, Union

import albumentations as A
from omegaconf import DictConfig, ListConfig

from .pixel import AVAILABLE_TRANSFORMS as PIXEL_TRANSFORMS
from .pixel import (RandAugmentPixel, random_blur, random_brightness,
                    random_clahe, random_compression, random_contrast,
                    random_gamma, random_hist_equal, random_inverse,
                    random_noise)
from .ratio import RandomRatio
from .resize import Letterbox, RandomResize
from .spatial import RandomTranslation
from .windowing import RandomWindowing

# 통합 transform registry
AVAILABLE_TRANSFORMS: dict[str, Any] = {'resize': A.Resize,
                                        'letterbox': Letterbox,
                                        'random_flip': A.HorizontalFlip,
                                        'random_ratio': RandomRatio,
                                        'random_resize': RandomResize,
                                        'random_windowing': RandomWindowing,
                                        'random_pixel_aug': RandAugmentPixel,
                                        'random_translation': RandomTranslation,
                                        }

# 픽셀 기반 transform 추가
AVAILABLE_TRANSFORMS.update(PIXEL_TRANSFORMS)


def recursive_check(config: Union[dict, list, DictConfig, ListConfig, Any]) -> Any:
    """Config의 타입에 따라 재귀적으로 일반 dict/list로 변환"""
    if isinstance(config, (DictConfig, dict)):
        return {k: recursive_check(v) for k, v in config.items()}
    elif isinstance(config, (ListConfig, list)):
        return [recursive_check(v) for v in config]
    else:
        return config


def build_transforms(transform_configs: dict[str, Any]) -> A.Compose:
    """
    주어진 transform 설정을 기반으로 Albumentations Compose 객체 생성
    :param transform_configs: transform 이름과 설정값의 dict
    :return: Albumentations Compose object
    """
    transforms = []
    for name, params in transform_configs.items():
        if name not in AVAILABLE_TRANSFORMS:
            raise ValueError(f"Unknown transform: {name}")
        transform_class = AVAILABLE_TRANSFORMS[name]
        kwargs = recursive_check(params) if params else {}
        transforms.append(transform_class(**kwargs))
    return A.Compose(transforms)
