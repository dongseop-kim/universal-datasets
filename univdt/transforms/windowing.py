import logging

import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform

from univdt.transforms.pixel import Invert

# 모듈 수준에서 로거 설정
logger = logging.getLogger(__name__)


def configure_logging(debug=False):
    """전역 로깅 설정을 구성합니다."""
    level = logging.DEBUG if debug else logging.INFO

    # 루트 로거 설정
    logging.basicConfig(level=level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def min_max_normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalize the image to the range [0, 1].
    Args:
        image (np.ndarray): The input image.

    Returns:
        np.ndarray: The normalized image.
    """
    min_val = image.min()
    max_val = image.max()
    return (image - min_val) / (max_val - min_val + 1e-6)


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
        p (float): probability
        debug (bool): Enable debug logging
    """

    def __init__(self,
                 width_param: float = 4.0,
                 width_range: float = 1.0,
                 use_median: bool = True,
                 p: float = 0.5,
                 debug: bool = False):
        super().__init__(p)
        self.use_median = use_median
        self.width_param = width_param
        self.width_range = width_range
        self.debug = debug

        # 모듈 로거의 레벨만 조정 (필요한 경우)
        if self.debug and logger.level > logging.DEBUG:
            logger.setLevel(logging.DEBUG)

        if self.debug:
            logger.debug(
                f"Initialized RandomWindowing: width_param={width_param}, width_range={width_range}, use_median={use_median}, p={p}")

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        if self.debug:
            logger.debug(f"Processing image with shape {img.shape}, dtype {img.dtype}")
            logger.debug(
                f"Image stats before transform - min: {img.min()}, max: {img.max()}, mean: {img.mean():.2f}, std: {img.std():.2f}")

        width_param = self._get_random_width_param()

        if self.debug:
            logger.debug(f"Using width_param: {width_param:.4f}")
            logger.debug(f"Using median as center: {self.use_median}")

        # Calculate statistical values for logging if debug is enabled
        if self.debug:
            center = np.median(img) if self.use_median else img.mean()
            std = img.std()
            range_width_half = (std * width_param) / 2.0
            low, high = center - range_width_half, center + range_width_half
            logger.debug(f"Windowing parameters - center: {center:.2f}, std: {std:.2f}")
            logger.debug(f"Clipping range - low: {low:.2f}, high: {high:.2f}")

        result = windowing(img, use_median=self.use_median, width_param=width_param)

        if self.debug:
            logger.debug(
                f"Image stats after transform - min: {result.min()}, max: {result.max()}, mean: {result.mean():.2f}, std: {result.std():.2f}")
            clipped_pixels = np.sum((img < low) | (img > high))
            total_pixels = img.size
            clip_percentage = (clipped_pixels / total_pixels) * 100
            logger.debug(f"Clipped {clipped_pixels} pixels ({clip_percentage:.2f}% of total)")

        return result.astype(np.uint8)

    def _get_random_width_param(self) -> float:
        # Generate a random width parameter within the specified range.
        random_width = self.width_param + np.random.uniform(-self.width_range/2, self.width_range/2)

        if self.debug:
            logger.debug(
                f"Generated random width_param: {random_width:.4f} (base: {self.width_param}, range: ±{self.width_range/2:.2f})")

        return random_width

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ('width_param', 'width_range', 'use_median', 'debug')

    def __call__(self, *args, **kwargs):
        if self.debug:
            logger.debug(f"RandomWindowing.__call__ invoked with p={self.p}")
            if np.random.random() < self.p:
                logger.debug(f"Transform will be applied (random < p)")
            else:
                logger.debug(f"Transform will be skipped (random >= p)")

        return super().__call__(*args, **kwargs)


class RandomWindowingInvert(ImageOnlyTransform):
    """
    Apply RandomWindowing and/or Invert with independent probabilities.

    Args:
        windowing_prob (float): Probability to apply RandomWindowing.
        invert_prob (float): Probability to apply Invert.
        windowing_kwargs (dict): Arguments to initialize RandomWindowing.
        invert_kwargs (dict): Arguments to initialize Invert.
        p (float): Overall probability of applying this transform.
        debug (bool): Enable debug logging.
    """

    def __init__(self,
                 windowing_prob: float = 0.5,
                 invert_prob: float = 0.5,
                 width_param: float = 4.0,
                 width_range: float = 1.0,
                 use_median: bool = True,
                 p: float = 1.0,
                 debug: bool = False):
        super().__init__(p=p)
        self.windowing_prob = windowing_prob
        self.invert_prob = invert_prob

        self.width_param = width_param
        self.width_range = width_range
        self.use_median = use_median

        self.debug = debug

        if debug:
            logger.debug(f"Initializing RandomWindowingAndInvert with:")
            logger.debug(f"  windowing_prob: {windowing_prob}")
            logger.debug(f"  invert_prob: {invert_prob}")
            logger.debug(f"  width_param: {width_param}")
            logger.debug(f"  width_range: {width_range}")
            logger.debug(f"  use_median: {use_median}")

        self.windowing = RandomWindowing(p=1.0, debug=debug,
                                         width_param=width_param,
                                         width_range=width_range,
                                         use_median=use_median)
        self.invert = Invert(p=1.0, debug=debug)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        if self.debug:
            logger.debug(f"Applying RandomWindowingAndInvert to image with shape {img.shape}")

        if np.random.rand() < self.windowing_prob:
            if self.debug:
                logger.debug("Applying RandomWindowing")
            img = self.windowing.apply(img, **params)

        if np.random.rand() < self.invert_prob:
            if self.debug:
                logger.debug("Applying Invert")
            img = self.invert.apply(img, **params)

        return img

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ('windowing_prob', 'invert_prob', 'debug')

class HighlightTripleView(ImageOnlyTransform):
    """
    Create a 3-channel visualization:
        - Channel 0: bright-enhanced view using log scaling
        - Channel 1: original image (or pre_aug-applied image)
        - Channel 2: dark-enhanced view using gamma

    Args:
        scale (float): Base scaling factor for bright-enhanced view.
        gamma (float): Base gamma value for dark-enhanced view.
        is_train (bool): Whether the transform is used in training mode (controls randomness).
        p (float): Probability of applying the whole transform.
        debug (bool): Enable debug logging.
    """

    def __init__(self,
                 scale: float = 1.0,
                 gamma: float = 2.0,
                 is_train: bool = False,
                 p: float = 1.0,
                 debug: bool = False):
        super().__init__(p=p)
        self.scale = scale
        self.gamma = gamma
        self.is_train = is_train
        self.debug = debug
        if self.debug:
            logger.debug(f"HighlightTripleView initialized (train={self.is_train})")

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        if self.debug:
            logger.debug(
                f"Input image shape: {img.shape}, dtype: {img.dtype}, "
                f"min: {img.min():.2f}, max: {img.max():.2f}"
            )

        img = min_max_normalize(img)
        img = (img * 255.0).astype(np.uint8)

        # Apply random variation in training mode
        scale = self.scale
        gamma = self.gamma
        if self.is_train:
            # 80% chance to apply random scaling and gamma
            if np.random.rand() < 0.8:
                scale = np.random.uniform(self.scale * 0.75, self.scale * 1.25)
                gamma = np.random.uniform(self.gamma * 0.75, self.gamma * 1.25)
            else:
                # 20% chance to use original scale and gamma
                scale = self.scale
                gamma = self.gamma

            if self.debug:
                logger.debug(f"Randomized scale={scale:.2f}, gamma={gamma:.2f}")

        bright_view = self._highlight_bright_regions(img, scale=scale)
        dark_view = self._highlight_dark_regions(img, gamma=gamma)

        stacked = np.stack([bright_view, img, dark_view], axis=-1)

        if self.debug:
            logger.debug(f"Stacked output shape: {stacked.shape}, dtype: {stacked.dtype}")

        return stacked

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ('scale', 'gamma', 'is_train', 'debug')

    def _highlight_bright_regions(self, x: np.ndarray, scale=1.0) -> np.ndarray:
        x_norm = x.astype(np.float32) / 255.0
        result = np.log1p(x_norm * scale)
        result = min_max_normalize(result)
        return (result * 255).astype(np.uint8)

    def _highlight_dark_regions(self, x: np.ndarray, gamma=2.0) -> np.ndarray:
        x_safe = np.clip(x, 1, 255)
        result = np.power(x_safe, gamma)
        result = min_max_normalize(result)
        return (result * 255).astype(np.uint8)
