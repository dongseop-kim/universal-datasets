import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform


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


class RandomWindowing(ImageOnlyTransform):
    """
    Apply random windowing
    Args:
        width_param (float): width parameter
        width_range (float): width range. width_param - width_range/2 ~ width_param + width_range/2 
                             if width_param = 4.0, width_range = 1.0, then width_param = 3.5 ~ 4.5
        use_median (bool): use median or not
        always_apply (bool): always apply or not
        p (float): probability
    """

    def __init__(self,
                 width_param: float = 4.0,
                 width_range: float = 1.0,
                 use_median: bool = True,
                 always_apply: bool = False,
                 p: float = 0.5):
        super().__init__(always_apply, p)
        self.use_median = use_median
        self.width_param = width_param
        self.width_range = width_range

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        width_param = self._get_random_width_param()
        return windowing(img, use_median=self.use_median, width_param=width_param).astype(np.uint8)

    def _get_random_width_param(self) -> float:
        # Generate a random width parameter within the specified range.
        return self.width_param + np.random.uniform(-self.width_range/2, self.width_range/2)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ('width_param', 'width_range', 'use_median',)
