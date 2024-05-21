import random

import numpy as np
from typing import Callable

from univdt.transforms.pixel import AVAILABLE_TRANSFORMS as PIXEL_TRANSFORMS
from univdt.transforms.pixel import RandAugmentPixel

# set seed for reproducibility
np.random.seed(42)
random.seed(42)

TEST_IMAGE1 = np.random.randint(0, 255, (768, 512, 3), dtype=np.uint8)
TEST_IMAGE2 = np.random.randint(0, 255, (512, 768, 1), dtype=np.uint8)
TEST_IMAGES = [TEST_IMAGE1, TEST_IMAGE2]

_TRANSFORMS = {'random_windowing': {'magnitude': 0.5, 'p': 1.0},
               'random_blur': {'magnitude': 0.5, 'p': 1.0},
               'random_gamma': {'magnitude': 0.5, 'p': 1.0},
               'random_brightness': {'magnitude': 0.5, 'p': 1.0},
               'random_clahe': {'magnitude': 0.5, 'p': 1.0},
               'random_contrast': {'magnitude': 0.5, 'p': 1.0},
               'random_compression': {'magnitude': 0.5, 'p': 1.0},
               'random_noise': {'magnitude': 0.5, 'p': 1.0},
               'random_histequal': {'magnitude': 0.5, 'p': 1.0}}


def apply_transform_and_assert_shape(transform: Callable):
    for tester in TEST_IMAGES:
        transformed: np.ndarray = transform(image=tester)['image']
        assert transformed.shape == tester.shape


def test_pixel_transforms():
    for transform_key, params in _TRANSFORMS.items():
        transform = PIXEL_TRANSFORMS[transform_key](**params)
        apply_transform_and_assert_shape(transform)


def test_rand_aug_pixel():
    transform = RandAugmentPixel(min_n=2, max_n=5, transforms=_TRANSFORMS)
    apply_transform_and_assert_shape(transform)
