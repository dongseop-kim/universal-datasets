import random

import albumentations as A
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform


def random_gamma(magnitude: float = 0.2, p=1.0):
    """
    Randomly change the gamma of the input image.

    Args:
        magnitude (float): maximum change of gamma. Default: 0.1. The range of magnitude is (0.0~ 1.0]
        always_apply (bool): always apply the transform.
        p (float): probability of applying the transform. Default: 1.0.
    """
    # set gamma limit to (100-m, 100+m). max gamma limit is (0, 200).
    # default gamma limit is (90, 110) with 0.1 magnitude.
    assert 0.0 < magnitude <= 1.0
    magnitude = min(magnitude, 1.0) * 100
    gamma_limit = (100 - magnitude, 100 + magnitude)
    return A.RandomGamma(gamma_limit=gamma_limit, p=p)


def random_blur(magnitude: float = 0.2, p=1.0):
    """
    Randomly blur the input image. The blur algorithm is chosen from Blur, GaussianBlur, MotionBlur, MedianBlur.

    Args:
        magnitude (float): maximum change of gamma. Default: 0.2. The range of magnitude is (0.0~ 1.0]

        p (float): probability of applying the transform. Default: 1.0.
    """
    # set blur limit to (3, m). max blur limit is (3, 21).
    # default blur limit is (3, 7) with 0.2 magnitude.
    assert 0.0 < magnitude <= 1.0
    m = min(round(magnitude * 19)+2, 21)
    m = m if m % 2 == 1 else m + 1  # make sure m is odd
    blur_limit = sorted((3, m))
    blur = A.Blur(blur_limit=blur_limit)
    gaussian_blur = A.GaussianBlur(blur_limit=blur_limit)
    motion_blur = A.MotionBlur(blur_limit=blur_limit)
    median_blur = A.MedianBlur(blur_limit=blur_limit)

    return A.OneOf([blur, gaussian_blur, motion_blur, median_blur], p=p)


def random_brightness(magnitude: float = 0.2, p=1.0):
    """
    Randomly change the brightness of the input image.

    Args:
        magnitude (float): maximum change of brightness. Default: 0.2. The range of magnitude is (0.0~ 1.0]
        always_apply (bool): always apply the transform.
        p (float): probability of applying the transform. Default: 1.0.
    """
    # set brightness limit to (-m, m). max brightness limit is (-1, 1).
    # default brightness limit is (-0.2, 0.2) with 0.2 magnitude.
    assert 0.0 < magnitude <= 1.0

    brightness_limit = sorted((-magnitude, magnitude))
    return A.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=0, p=p)


def random_contrast(magnitude: float = 0.2, p=1.0):
    """
    Randomly change the contrast of the input image.

    Args:
        magnitude (float): maximum change of contrast. Default: 0.2. The range of magnitude is (0.0~ 1.0]
        always_apply (bool): always apply the transform.
        p (float): probability of applying the transform. Default: 1.0.
    """
    # set contrast limit to (-m, m). max contrast limit is (-1, 1).
    # default contrast limit is (-0.2, 0.2) with 0.2 magnitude.
    assert 0.0 < magnitude <= 1.0

    contrast_limit = sorted((-magnitude, magnitude))
    return A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=contrast_limit, p=p)


def random_noise(magnitude: float = 0.2, p=1.0):
    """
    Randomly add noise to the input image.

    Args:
        magnitude (float): maximum change of noise. Default: 0.2. The range of magnitude is (0.0~ 1.0]
        always_apply (bool): always apply the transform.
        p (float): probability of applying the transform. Default: 1.0.
    """
    # set noise limit to (0, m). max noise limit is (0, 1).
    # default noise limit is (0, 0.2) with 0.2 magnitude.
    assert 0.0 < magnitude <= 1.0

    gaussian_noise = A.GaussNoise(var_limit=(0, magnitude), p=p)
    multiplicative_noise = A.MultiplicativeNoise(multiplier=(1-magnitude, 1+magnitude), p=p)
    return A.OneOf([gaussian_noise, multiplicative_noise], p=p)


def random_compression(magnitude: float = 0.2, p=1.0):
    """
    Randomly compress the input image.

    Args:
        magnitude (float): maximum change of compression. Default: 0.2. The range of magnitude is (0.0~ 1.0]
        p (float): probability of applying the transform. Default: 1.0.
    """
    # set compression limit to (1-m, 1+m). max compression limit is (0, 2).
    # default compression limit is (0.8, 1.2) with 0.2 magnitude.
    assert 0.0 < magnitude <= 1.0
    # compression_types = [A.ImageCompression.ImageCompressionType.JPEG, A.ImageCompression.ImageCompressionType.WEBP]
    # compression_type = random.choice(compression_types)
    magnitude = min(magnitude, 1.0) * 100
    compression_lower = max(1, 100 - magnitude)
    return A.ImageCompression(quality_lower=compression_lower, quality_upper=100,
                              compression_type=A.ImageCompression.ImageCompressionType.JPEG, p=p)


def random_hist_equal(magnitude: float = 0.2, p=1.0):
    """
    Randomly apply histogram equalization to the input image.

    Args:
        magnitude (float): None
        p (float): probability of applying the transform. Default: 1.0.
    """
    t1 = A.Equalize(mode='cv')
    t2 = A.Equalize(mode='pil')
    return A.OneOf([t1, t2], p=p)


def random_clahe(magnitude: float = 0.2, p=1.0):
    """
    Randomly apply CLAHE to the input image.

    Args:
        magnitude (float): None
        p (float): probability of applying the transform. Default: 1.0.
    """
    tile_grid_size = (8, 8)
    clip_limit = 2.0 + magnitude * 10
    return A.CLAHE(clip_limit=(1, clip_limit), tile_grid_size=tile_grid_size, p=p)


AVAILABLE_TRANSFORMS = {'random_blur': random_blur,
                        'random_brightness': random_brightness,
                        'random_clahe': random_clahe,
                        'random_compression': random_compression,
                        'random_contrast': random_contrast,
                        'random_gamma': random_gamma,
                        'random_histequal': random_hist_equal,
                        'random_noise': random_noise}


class RandAugmentPixel(ImageOnlyTransform):
    '''RandAugment for pixel-level transforms
    Args:
        transforms (Dict[str, Dict]): dictionary of transforms to apply
        min_n (int): minimum number of transforms to apply
        max_n (int): maximum number of transforms to apply
        replace (bool): Whether the sampled transforms are with or without replacement. Default: True.
        p (float): probability of applying the transform. (default: 1.0).  the p of each transform is normalized.
    '''

    def __init__(self, transforms: dict[str, dict],
                 min_n: int = 1, max_n: int = 3,
                 replace: bool = True, p: float = 1.0):
        super(ImageOnlyTransform, self).__init__(always_apply=False, p=p)
        if not transforms:
            raise ValueError('transforms must be specified for training')
        self.min_n, self.max_n = min_n, max_n
        self.replace = replace
        t = [AVAILABLE_TRANSFORMS[key](**val) for key, val in transforms.items()]
        self.transforms = [A.SomeOf(t, n=i, p=p, replace=self.replace) for i in range(min_n, max_n+1)]

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        if random.random() < self.p:
            t = random.choice(self.transforms)
            img = t(image=img)['image']
        return img

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ('min_n', 'max_n', 'transforms', 'p', )
