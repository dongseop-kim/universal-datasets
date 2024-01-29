import albumentations as A
import numpy as np


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


def windowing(image: np.ndarray, use_median: bool = False, width_param: float = 4.0) -> np.ndarray:
    """
    Windowing function that clips the values based on the given params.
    Args:
        image (str): the image to do the windowing
        use_median (bool): use median as center if True, mean otherwise
        width_param (float): the width of the value range for windowing.
        brightness (float) : brightness_rate. a value between 0 and 1 and indicates the amount to subtract.

    Returns:
        image that was windowed.
    """
    center = np.median(image) if use_median else image.mean()
    range_width_half = (image.std() * width_param) / 2.0
    low, high = center - range_width_half, center + range_width_half
    return np.clip(image, low, high)


class RandomWindowing(A.ImageOnlyTransform):
    """
    Apply random windowing
    Args:
        width_param (float): width parameter
        width_range (float): width range. width_param - width_range/2 ~ width_param + width_range/2
        use_median (bool): use median or not
        always_apply (bool): always apply or not
        p (float): probability
    """

    def __init__(self, width_param: float = 4.0, width_range: float = 1.0,  use_median: bool = True,
                 always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.use_median = use_median
        self.width_param = width_param
        self.width_range = width_range

    # TODO: width_param -> update_params에서 random하게 설정하도록 변경
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        width_param = (self.width_param - (self.width_range/2)) + (np.random.rand(1) * (self.width_range))
        return windowing(img, use_median=self.use_median, width_param=width_param).astype(np.uint8)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ('width_param', 'width_range', 'use_median',)


def random_windowing(magnitude: float = 0.2, p=1.0):
    """
    Randomly change the windowing of the input image.

    Args:
        magnitude (float): 
        p (float): probability of applying the transform. Default: 1.0.
    """
    # The range of magnitude is (0.0~ 1.0]
    # when magnitude is 1.0, width_range
    assert 0.0 < magnitude <= 1.0
    return RandomWindowing(width_param=4.0, width_range=magnitude*2, use_median=True, p=p)
