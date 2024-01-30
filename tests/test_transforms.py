import numpy as np
import pytest
import copy

import random
from univdt.transforms.builder import AVAILABLE_TRANSFORMS

# set seed for reproducibility
np.random.seed(42)
random.seed(42)

TEST_IMAGE = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)


def test_random_windowing():
    windower = AVAILABLE_TRANSFORMS['random_windowing'](magnitude=1.0, p=1.0)
    augmented: np.ndarray = windower(image=copy.deepcopy(TEST_IMAGE))['image']
    assert augmented.shape == (512, 512, 3)
    assert augmented.dtype == np.uint8
    print("random windowing")
    print(TEST_IMAGE.mean(), TEST_IMAGE.std())
    print(augmented.mean(), augmented.std())
    np.testing.assert_allclose(augmented.mean(), 127.01, atol=0.1)
    np.testing.assert_allclose(augmented.std(), 73.62, atol=0.1)


def test_random_blur():
    blur = AVAILABLE_TRANSFORMS['random_blur'](magnitude=0.5, p=1.0)
    augmented: np.ndarray = blur(image=copy.deepcopy(TEST_IMAGE))['image']
    assert augmented.shape == (512, 512, 3)
    assert augmented.dtype == np.uint8
    print("random blur")
    print(TEST_IMAGE.mean(), TEST_IMAGE.std())
    print(augmented.mean(), augmented.std())
    np.testing.assert_allclose(augmented.mean(), 127.02, atol=0.1)
    np.testing.assert_allclose(augmented.std(), 23.30, atol=0.1)


def test_random_gamma():
    gamma = AVAILABLE_TRANSFORMS['random_gamma'](magnitude=0.5, p=1.0)
    augmented: np.ndarray = gamma(image=copy.deepcopy(TEST_IMAGE))['image']
    assert augmented.shape == (512, 512, 3)
    assert augmented.dtype == np.uint8
    print("random gamma")
    print(TEST_IMAGE.mean(), TEST_IMAGE.std())
    print(augmented.mean(), augmented.std())
    np.testing.assert_allclose(augmented.mean(), 131.80, atol=0.1)
    np.testing.assert_allclose(augmented.std(), 72.58, atol=0.1)
